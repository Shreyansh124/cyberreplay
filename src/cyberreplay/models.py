"""The four Track A classifiers as leakage-safe sklearn Pipelines (pipeline stage 5).

Each pipeline = ``ColumnSelector -> SimpleImputer [-> clip -> log -> scale] -> model``.
Imbalance is handled with class weights (no SMOTE). Each model is a thin
subclass/wrapper that turns a ``class_weighting`` hyperparameter
("balanced" = n / (k * count_c), or its square root "sqrt") into per-row
sample weights inside ``fit``. Weights are therefore computed from the rows
each CV fold trains on, and the weighting strength is tuned like any other
hyperparameter (fully balanced weights make the linear model flag ~12% of
BENIGN flows as attacks, see DECISIONS.md).

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


class WeightedLogisticRegression(LogisticRegression):
    """Multinomial Logistic Regression with a tunable class-weighting strength.

    ``class_weighting`` ("balanced" or "sqrt") is turned into per-row sample
    weights inside ``fit``, so each CV fold computes weights from its own
    training rows only.
    """

    def __init__(self, C: float = 1.0, tol: float = 1e-4, max_iter: int = 1000, solver: str = "lbfgs",
                 class_weighting: str = "balanced", random_state: int = config.SEED):
        super().__init__(C=C, tol=tol, max_iter=max_iter, solver=solver, random_state=random_state)
        self.class_weighting = class_weighting

    def fit(self, X, y, sample_weight=None):
        """Fit with class-balancing sample weights (``sample_weight`` is ignored)."""
        return super().fit(X, y, sample_weight=weights_for(np.asarray(y), self.class_weighting))


class WeightedDecisionTree(DecisionTreeClassifier):
    """CART decision tree with a tunable class-weighting strength (see ``weights_for``)."""

    def __init__(self, criterion: str = "gini", max_depth: int | None = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features=None, class_weighting: str = "balanced",
                 random_state: int = config.SEED):
        super().__init__(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=random_state)
        self.class_weighting = class_weighting

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit with class-balancing sample weights (``sample_weight`` is ignored)."""
        return super().fit(X, y, sample_weight=weights_for(np.asarray(y), self.class_weighting),
                           check_input=check_input)


class WeightedRandomForest(RandomForestClassifier):
    """Random Forest with a tunable class-weighting strength (see ``weights_for``)."""

    def __init__(self, n_estimators: int = 100, criterion: str = "gini", max_depth: int | None = None,
                 min_samples_leaf: int = 1, max_features="sqrt", class_weighting: str = "balanced",
                 n_jobs: int = config.N_JOBS, random_state: int = config.SEED):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=n_jobs,
                         random_state=random_state)
        self.class_weighting = class_weighting

    def fit(self, X, y, sample_weight=None):
        """Fit with class-balancing sample weights (``sample_weight`` is ignored)."""
        return super().fit(X, y, sample_weight=weights_for(np.asarray(y), self.class_weighting))


def make_estimator(name: str, **params):
    """Instantiate one of the four classifiers with the project defaults.

    Every model starts from fully balanced class weights; the search may soften
    them to their square root (``class_weighting="sqrt"``).

    Args:
        name: One of ``config.MODEL_NAMES``.
        **params: Overrides of the default hyperparameters.

    Returns:
        Unfitted estimator.
    """
    if name == "LogisticRegression":
        return WeightedLogisticRegression(**params)
    if name == "DecisionTree":
        return WeightedDecisionTree(**params)
    if name == "RandomForest":
        return WeightedRandomForest(**params)
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
            "class_weighting": ["balanced", "sqrt"],
        },
        "DecisionTree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 15, 20, 30, 40],
            "min_samples_split": randint(2, 21),
            "min_samples_leaf": randint(1, 11),
            "max_features": [None, "sqrt", 0.5, 0.8],
            "class_weighting": ["balanced", "sqrt"],
        },
        "RandomForest": {
            "n_estimators": [100, 150, 200],
            "max_depth": [None, 20, 30, 40],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.3, 0.5],
            "class_weighting": ["balanced", "sqrt"],
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
