"""Feature engineering: split, feature selection and preprocessing (pipeline stage 4).

Leakage rules enforced here:
    * the train/test split happens on the de-duplicated data, stratified by family;
    * every filter (variance, correlation, importance) is fitted on the
      training split only;
    * every fitted statistic used at prediction time (median imputation,
      outlier caps, scaling) lives inside an sklearn ``Pipeline`` so that CV,
      tuning and the final fit only ever see training rows;
    * identifiers (Flow ID, IPs, ports, Timestamp) never enter the feature list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config


# --------------------------------------------------------------------------
# Targets and split
# --------------------------------------------------------------------------
def encode_target(family: pd.Series) -> np.ndarray:
    """Map family names to integer codes 0..7 following ``config.CLASS_ORDER``.

    Args:
        family: Series of family names.

    Returns:
        ``int64`` array of class codes (BENIGN = 0).
    """
    return family.map(config.CLASS_TO_CODE).to_numpy(dtype=np.int64)


def decode_target(codes: np.ndarray) -> np.ndarray:
    """Map integer class codes back to family names.

    Args:
        codes: Array of class codes.

    Returns:
        Array of family-name strings.
    """
    return np.asarray(config.CLASS_ORDER, dtype=object)[np.asarray(codes, dtype=np.int64)]


def split_train_test(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
                     seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split by family (run after de-duplication).

    Args:
        df: Clean, de-duplicated flows.
        test_size: Fraction held out for the final test.
        seed: Random seed.

    Returns:
        ``(train_df, test_df)`` with fresh indices.
    """
    train, test = train_test_split(df, test_size=test_size, stratify=df[config.TARGET_COL],
                                   random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, keep_all_below: int = config.MINORITY_KEEP_ALL_THRESHOLD,
                      seed: int = config.SEED) -> pd.DataFrame:
    """Draw a sample of ~``n`` rows that keeps every row of the small classes.

    Classes with fewer than ``keep_all_below`` rows are kept in full (so rare
    attacks are not thinned out); the remaining budget is filled from the
    large classes proportionally to their size.

    Args:
        df: Training flows (must contain ``family``).
        n: Target sample size.
        keep_all_below: Class-size threshold for keeping all rows.
        seed: Random seed.

    Returns:
        Sampled frame (shuffled, fresh index).
    """
    counts = df[config.TARGET_COL].value_counts()
    small = counts[counts < keep_all_below].index
    keep = df[df[config.TARGET_COL].isin(small)]
    big = df[~df[config.TARGET_COL].isin(small)]
    budget = max(n - len(keep), 0)
    if budget >= len(big):
        sampled_big = big
    else:
        sampled_big, _ = train_test_split(big, train_size=budget, stratify=big[config.TARGET_COL],
                                          random_state=seed)
    out = pd.concat([keep, sampled_big])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------
# Feature selection (fitted on train only)
# --------------------------------------------------------------------------
def variance_filter(X: pd.DataFrame, threshold: float = config.VARIANCE_THRESHOLD) -> list[str]:
    """Return columns whose variance on the training data exceeds ``threshold``.

    Args:
        X: Training features.
        threshold: Minimum variance (0 removes constant columns).

    Returns:
        Kept column names.
    """
    vt = VarianceThreshold(threshold).fit(X.fillna(0))
    return list(X.columns[vt.get_support()])


def correlation_filter(X: pd.DataFrame, threshold: float = config.CORRELATION_THRESHOLD
                       ) -> tuple[list[str], pd.DataFrame]:
    """Greedy correlation filter: keep a column unless it correlates > ``threshold`` with a kept one.

    Columns are visited in CSV order, so the more basic statistic (e.g.
    ``Total Fwd Packets``) is kept and its derived twin (``Subflow Fwd Packets``)
    is dropped.

    Args:
        X: Training features (a sample is enough).
        threshold: Absolute Pearson correlation above which a column is redundant.

    Returns:
        ``(kept columns, table of dropped columns with the kept partner and r)``.
    """
    corr = X.corr().abs().fillna(0)
    kept: list[str] = []
    dropped = []
    for col in X.columns:
        partners = corr.loc[col, kept] if kept else pd.Series(dtype=float)
        if len(partners) and partners.max() > threshold:
            dropped.append({"dropped": col, "kept_partner": partners.idxmax(), "abs_r": partners.max()})
        else:
            kept.append(col)
    return kept, pd.DataFrame(dropped)


def importance_filter(X: pd.DataFrame, y: np.ndarray, cumulative: float = config.IMPORTANCE_CUMULATIVE,
                      min_features: int = 15, seed: int = config.SEED) -> tuple[list[str], pd.DataFrame]:
    """Keep the most important features according to a class-weighted Random Forest.

    Args:
        X: Training features (sample).
        y: Encoded training labels.
        cumulative: Keep the top features whose importances sum to this share.
        min_features: Lower bound on the number of features kept.
        seed: Random seed.

    Returns:
        ``(kept columns, importance table sorted descending with cumulative share)``.
    """
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=config.N_JOBS,
                                random_state=seed, min_samples_leaf=2)
    rf.fit(X.fillna(X.median()), y)
    imp = (pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    imp["cumulative"] = imp["importance"].cumsum()
    n_keep = max(min_features, int((imp["cumulative"] < cumulative).sum()) + 1)
    imp["selected"] = imp.index < n_keep
    return list(imp.loc[imp["selected"], "feature"]), imp


def select_features(train: pd.DataFrame, candidates: list[str], sample_size: int = config.SELECTION_SAMPLE_SIZE,
                    seed: int = config.SEED) -> dict:
    """Run variance -> correlation -> importance selection on the training split.

    Args:
        train: Training flows (never the test split).
        candidates: Candidate feature columns (identifiers already excluded).
        sample_size: Rows (minority-preserving sample) for the correlation and
            importance steps; the variance step uses all training rows.
        seed: Random seed.

    Returns:
        Dict with ``selected`` (final list, CSV order), ``constant`` (dropped),
        ``correlated`` (DataFrame), ``importance`` (DataFrame) and the count after
        each step in ``counts``.
    """
    after_var = variance_filter(train[candidates])
    constant = [c for c in candidates if c not in after_var]
    sample = stratified_sample(train, sample_size, seed=seed)
    after_corr, correlated = correlation_filter(sample[after_var])
    after_imp, importance = importance_filter(sample[after_corr], encode_target(sample[config.TARGET_COL]),
                                              seed=seed)
    selected = [c for c in candidates if c in after_imp]
    return {
        "selected": selected,
        "constant": constant,
        "correlated": correlated,
        "importance": importance,
        "counts": {"candidates": len(candidates), "after_variance": len(after_var),
                   "after_correlation": len(after_corr), "after_importance": len(selected)},
    }


# --------------------------------------------------------------------------
# Preprocessing transformers
# --------------------------------------------------------------------------
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed list of columns from a DataFrame.

    Putting the selection inside the saved pipeline lets the dashboard feed a
    raw CIC-IDS2017 row (with identifiers and all 80+ columns) straight in.

    Args:
        columns: Feature names to keep, in order.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnSelector":
        """No-op fit (records the output feature names)."""
        self.feature_names_out_ = np.asarray(self.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X[columns]`` as float32, with +/-inf turned into NaN."""
        out = X[self.columns].astype(np.float32)
        return out.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the selected column names."""
        return np.asarray(self.columns, dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Cap each feature at an upper quantile learned on the training data.

    Used only in the scale-sensitive Logistic Regression pipeline: extreme
    magnitudes (up to 1e8 us durations, 1e9 B/s rates) would otherwise
    dominate the standardised linear model. Rows are never deleted.

    Args:
        upper: Upper quantile used as the cap (e.g. 0.999).
    """

    def __init__(self, upper: float = config.OUTLIER_QUANTILE):
        self.upper = upper

    def fit(self, X, y=None) -> "QuantileClipper":
        """Learn per-column caps (NaN-aware) from the training data."""
        arr = np.asarray(X, dtype=np.float64)
        self.caps_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        """Clip values above the learned caps."""
        return np.minimum(np.asarray(X, dtype=np.float64), self.caps_)

    def get_feature_names_out(self, input_features=None):
        """Pass feature names through unchanged."""
        return input_features


def signed_log1p(X) -> np.ndarray:
    """Apply ``sign(x) * log1p(|x|)`` - compresses heavy tails, keeps the -1 sentinel."""
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def make_preprocessor(features: list[str], scale: bool) -> list[tuple[str, object]]:
    """Build the preprocessing steps that precede a model.

    Args:
        features: Selected feature names.
        scale: True for scale-sensitive models (Logistic Regression): adds
            outlier capping, signed-log transform and standardisation. Trees are
            invariant to monotone transforms, so they only get imputation and
            keep extreme values (49% of Rare-family rows have a value above the
            global 99.9th percentile - capping would blur them).

    Returns:
        List of ``(name, transformer)`` steps for an sklearn Pipeline.
    """
    steps: list[tuple[str, object]] = [
        ("select", ColumnSelector(features)),
        # Median imputation of the NaN Flow Bytes/s / Flow Packets/s (zero-duration flows).
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps += [
            ("clip", QuantileClipper()),
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    return steps


def preprocessing_only(pipeline: Pipeline) -> Pipeline:
    """Return the fitted preprocessing part (all steps except ``model``) of a pipeline."""
    return Pipeline(pipeline.steps[:-1])
