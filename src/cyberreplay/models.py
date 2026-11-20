"""The four Track A classifiers as leakage-safe sklearn Pipelines (pipeline stage 5).

Each pipeline = ``ColumnSelector -> SimpleImputer [-> clip -> log -> scale] -> model``.
Imbalance is handled with class weights (no SMOTE):
    * LogisticRegression / DecisionTree: ``class_weight="balanced"``
    * RandomForest: ``"balanced"`` or ``"balanced_subsample"`` (tuned)
    * XGBoost: per-row sample weights computed inside ``WeightedXGBClassifier``
      from the labels it is fitted on (so every CV fold gets its own weights).

The search spaces for RandomizedSearchCV (step 7) are defined next to the
models so names stay consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config
from .features import make_preprocessor


def weights_for(y: np.ndarray, mode: str) -> np.ndarray:
    """Per-row sample weights that rebalance classes.

    Args:
        y: Encoded labels.
        mode: ``"balanced"`` (n / (k * count_c)) or ``"sqrt"`` (square root of the
            balanced weight - softer, fewer false positives on BENIGN).

    Returns:
        Array of sample weights (mean ~1).
    """
    w = compute_sample_weight("balanced", y)
    if mode == "sqrt":
        w = np.sqrt(w)
    elif mode != "balanced":
        raise ValueError(f"unknown weighting mode {mode}")
    return w / w.mean()


class WeightedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost multi-class classifier that applies class-balancing sample weights itself.

    Wrapping the weights inside ``fit`` (instead of passing ``sample_weight``
    from outside) means RandomizedSearchCV computes them on each training
    fold only and can tune the weighting mode like any hyperparameter.

    Args:
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
        min_child_weight, reg_lambda, gamma: Standard XGBoost hyperparameters.
        class_weighting: ``"balanced"`` or ``"sqrt"`` (see ``weights_for``).
        n_jobs: Threads used by XGBoost.
        random_state: Seed.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 8, learning_rate: float = 0.1,
                 subsample: float = 0.9, colsample_bytree: float = 0.8, min_child_weight: float = 1.0,
                 reg_lambda: float = 1.0, gamma: float = 0.0, class_weighting: str = "balanced",
                 n_jobs: int = config.N_JOBS, random_state: int = config.SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.class_weighting = class_weighting
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y) -> "WeightedXGBClassifier":
        """Fit XGBoost (hist, multi:softprob) with class-balancing sample weights.

        Args:
            X: Preprocessed features.
            y: Encoded labels 0..K-1.

        Returns:
            self (fitted; ``model_`` holds the underlying ``XGBClassifier``).
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight, reg_lambda=self.reg_lambda, gamma=self.gamma,
            objective="multi:softprob", tree_method="hist", eval_metric="mlogloss",
            n_jobs=self.n_jobs, random_state=self.random_state, verbosity=0,
        )
        self.model_.fit(X, y, sample_weight=weights_for(y, self.class_weighting))
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, columns ordered as ``classes_``."""
        return self.model_.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """Most probable class code per row."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Gain-based feature importances of the fitted booster."""
        return self.model_.feature_importances_


def make_estimator(name: str, **params):
    """Instantiate one of the four classifiers with the project defaults.

    Args:
        name: One of ``config.MODEL_NAMES``.
        **params: Overrides of the default hyperparameters.

    Returns:
        Unfitted estimator.
    """
    seed = config.SEED
    if name == "LogisticRegression":
        base = dict(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, tol=1e-4,
                    random_state=seed)
        return LogisticRegression(**{**base, **params})
    if name == "DecisionTree":
        base = dict(class_weight="balanced", random_state=seed)
        return DecisionTreeClassifier(**{**base, **params})
    if name == "RandomForest":
        base = dict(n_estimators=100, class_weight="balanced_subsample", n_jobs=config.N_JOBS,
                    random_state=seed)
        return RandomForestClassifier(**{**base, **params})
    if name == "XGBoost":
        base = dict(n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0)
        return WeightedXGBClassifier(**{**base, **params})
    raise ValueError(f"unknown model {name}")


def build_pipeline(name: str, features: list[str], **params) -> Pipeline:
    """Build the full preprocessing + model pipeline for one model.

    Args:
        name: Model name (``config.MODEL_NAMES``).
        features: Selected feature columns.
        **params: Model hyperparameters (without the ``model__`` prefix).

    Returns:
        Unfitted ``Pipeline`` whose last step is named ``model``.
    """
    scale = name == "LogisticRegression"
    return Pipeline(make_preprocessor(features, scale=scale) + [("model", make_estimator(name, **params))])


def search_space(name: str) -> dict:
    """RandomizedSearchCV distributions for one model (keys prefixed ``model__``).

    Args:
        name: Model name.

    Returns:
        Dict of parameter distributions.
    """
    spaces = {
        "LogisticRegression": {
            "C": loguniform(1e-2, 1e3),
            "tol": [1e-4, 1e-3],
        },
        "DecisionTree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 15, 20, 30, 40],
            "min_samples_split": randint(2, 21),
            "min_samples_leaf": randint(1, 11),
            "max_features": [None, "sqrt", 0.5, 0.8],
        },
        "RandomForest": {
            "n_estimators": [100, 150, 200],
            "max_depth": [None, 20, 30, 40],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.3, 0.5],
            "class_weight": ["balanced", "balanced_subsample"],
            "criterion": ["gini", "entropy"],
        },
        "XGBoost": {
            "n_estimators": [200, 300, 400],
            "max_depth": [6, 8, 10, 12],
            "learning_rate": loguniform(0.03, 0.3),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "class_weighting": ["balanced", "sqrt"],
        },
    }
    return {f"model__{k}": v for k, v in spaces[name].items()}


# Number of RandomizedSearchCV candidates per model (bounded by fit cost).
N_ITER: dict[str, int] = {"LogisticRegression": 10, "DecisionTree": 30, "RandomForest": 20, "XGBoost": 20}
