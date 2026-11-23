"""Unit tests for the report-extension diagnostics (split table, ROC, baseline, fit diagnosis, curves)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cyberreplay import config, diagnostics, evaluate  # noqa: E402


@pytest.fixture(autouse=True)
def no_files(monkeypatch):
    """Keep tests from writing tables or figures into reports/."""
    monkeypatch.setattr(diagnostics, "save_table", lambda *a, **k: None)
    monkeypatch.setattr(diagnostics, "save", lambda *a, **k: None)


def synthetic(n_per_class: int = 60, classes: tuple[str, ...] = ("BENIGN", "DoS", "DDoS"), seed: int = 0):
    """Small separable data set: one informative feature per class plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, fam in enumerate(classes):
        x = rng.normal(size=(n_per_class, 3))
        x[:, 0] += 4 * k
        rows.append(pd.DataFrame(x, columns=["f0", "f1", "f2"]).assign(**{config.TARGET_COL: fam}))
    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_diagnose_thresholds():
    """Low training F1 is underfitting; otherwise the train-validation gap decides."""
    assert diagnostics.diagnose(0.72, 0.71) == "underfit (high bias)"
    assert diagnostics.diagnose(0.999, 0.90) == "overfit (high variance)"
    assert diagnostics.diagnose(0.99, 0.96) == "slight overfit"
    assert diagnostics.diagnose(0.97, 0.96) == "good fit"


def test_split_table_counts_and_total():
    """Per-family counts add up and the Total row carries the test share."""
    train = pd.DataFrame({config.TARGET_COL: ["BENIGN"] * 8 + ["Rare"] * 4})
    test = pd.DataFrame({config.TARGET_COL: ["BENIGN"] * 2 + ["Rare"]})
    out = diagnostics.split_table(train, test, pd.Series({"BENIGN": 15, "Rare": 6})).set_index("family")
    assert out.loc["Rare", ["raw_flows", "unique_flows", "train", "test"]].tolist() == [6, 5, 4, 1]
    assert out.loc["Rare", "raw_labels"] == "Infiltration, Heartbleed"
    assert out.loc["Total", "unique_flows"] == 15 and out.loc["Total", "test_share"] == pytest.approx(0.2)
    assert out.loc["DoS", "train"] == 0


def test_check_reproduction_accepts_equal_and_rejects_changed():
    """Identical metrics pass; a changed macro F1 raises."""
    stored = pd.DataFrame({c: [0.5] for c in diagnostics.REPRO_COLS}, index=pd.Index(["RandomForest"], name="model"))
    metrics = {"RandomForest": {c: 0.5 for c in diagnostics.REPRO_COLS}}
    assert diagnostics.check_reproduction(metrics, stored).to_numpy().max() == 0
    metrics["RandomForest"]["f1_macro"] = 0.51
    with pytest.raises(RuntimeError):
        diagnostics.check_reproduction(metrics, stored)


def test_extend_comparison_keeps_existing_values():
    """New columns are inserted after accuracy and f1_macro; old values are untouched."""
    stored = pd.DataFrame({"accuracy": [0.9], "precision_macro": [0.8], "f1_macro": [0.7], "train_time_s": [12.3]},
                          index=pd.Index(["DecisionTree"], name="model"))
    out = diagnostics.extend_comparison(stored, {"DecisionTree": {"balanced_accuracy": 0.6, "roc_auc_ovr": 0.95}})
    assert list(out.columns) == ["accuracy", "balanced_accuracy", "precision_macro", "f1_macro", "roc_auc_ovr",
                                 "train_time_s"]
    assert out.loc["DecisionTree", "train_time_s"] == 12.3 and out.loc["DecisionTree", "roc_auc_ovr"] == 0.95


def test_roc_curves_and_macro_auc():
    """Per-class AUCs are 1 for perfect scores and the macro AUC matches roc_auc_ovr."""
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(len(config.CLASS_ORDER)), 20)
    perfect = np.eye(len(config.CLASS_ORDER))[y]
    curves = diagnostics.roc_curves(y, perfect)
    assert all(a == pytest.approx(1.0) for _, _, a in curves.values())
    noisy = rng.dirichlet(np.ones(len(config.CLASS_ORDER)), size=len(y)) + 0.5 * perfect
    noisy /= noisy.sum(axis=1, keepdims=True)
    _, tpr, auc = diagnostics.macro_roc(diagnostics.roc_curves(y, noisy))
    assert auc == pytest.approx(evaluate.roc_auc_ovr(y, noisy))
    assert tpr[0] >= 0 and tpr[-1] == pytest.approx(1.0)


def test_dummy_predicts_majority_and_has_chance_auc():
    """The baseline always predicts the most frequent class; its ROC-AUC is 0.5."""
    df = synthetic(n_per_class=10, classes=tuple(config.CLASS_ORDER))
    df = pd.concat([df, df[df[config.TARGET_COL] == "BENIGN"]], ignore_index=True)
    y = df[config.TARGET_COL].map(config.CLASS_TO_CODE).to_numpy()
    pipe = diagnostics.make_dummy(["f0", "f1", "f2"]).fit(df, y)
    pred = pipe.predict(df)
    assert (pred == 0).all()
    assert evaluate.roc_auc_ovr(y, evaluate.full_proba(pipe, df)) == pytest.approx(0.5)


def test_full_proba_fills_unseen_classes_with_zeros():
    """Columns follow CLASS_ORDER even when the model saw only some classes."""
    class TwoClass:
        classes_ = np.array([0, 3])

        def predict_proba(self, X):
            return np.tile([0.25, 0.75], (len(X), 1))

    out = evaluate.full_proba(TwoClass(), pd.DataFrame({"x": [1, 2]}))
    assert out.shape == (2, len(config.CLASS_ORDER))
    assert out[0, 3] == 0.75 and out[:, [1, 2, 4, 5, 6, 7]].sum() == 0


def test_balanced_accuracy_equals_macro_recall():
    """For multi-class labels the mean per-class recall is the balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    p = np.array([0, 1, 0, 1, 1, 2, 0, 2, 2])
    assert balanced_accuracy_score(y, p) == pytest.approx(recall_score(y, p, average="macro"))


def test_learning_curve_rows_grow_with_fraction():
    """One score per fold and fraction; larger fractions fit on more rows."""
    df = synthetic()
    lc = diagnostics.learning_curve_scores("DecisionTree", {"max_depth": 3}, df, ["f0", "f1", "f2"],
                                           fractions=[0.5, 1.0], folds=3)
    assert len(lc) == 6
    rows = lc.groupby("fraction")["train_rows"].mean()
    assert rows[0.5] < rows[1.0]
    assert lc[["train_f1", "val_f1"]].stack().between(0, 1).all()


def test_validation_and_boosting_curves_shape():
    """Validation curves give one row per value and fold; boosting reuses one booster per fold."""
    df = synthetic()
    feats = ["f0", "f1", "f2"]
    vc = diagnostics.validation_curve_scores("DecisionTree", {}, df, feats, "max_depth", [1, None], folds=2)
    assert len(vc) == 4 and set(vc["value"]) == {"1", "none"}
    depth1 = vc[vc["value"] == "1"]["train_f1"].mean()
    assert depth1 < vc[vc["value"] == "none"]["train_f1"].mean()
    bc = diagnostics.boosting_curve_scores({"max_depth": 2, "n_jobs": 1}, df, feats, rounds=[1, 5], folds=2)
    assert len(bc) == 4 and set(bc["value"]) == {"1", "5"}
