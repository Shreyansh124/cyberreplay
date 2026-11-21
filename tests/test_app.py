"""Headless smoke test of the Streamlit dashboard (steps 11-14) with ``AppTest``.

Requires the saved model (``models/best_model.joblib``) and the committed
dataset sample. Run with ``pytest tests/test_app.py -s``.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(not (ROOT / "models" / "best_model.joblib").exists(),
                                reason="trained model not available")


def _labels(elements) -> list[str]:
    """Return the labels of a list of AppTest widgets."""
    return [e.label for e in elements]


def test_dashboard_renders_and_replays():
    """The app renders every tab, explains a detection and completes an unpaced replay."""
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4
    assert "Flows analysed" in _labels(at.metric)

    # Step 14: explanation for the last detected family.
    fam = next(s for s in at.selectbox if s.label == "Detected family")
    fam.set_value(fam.options[-1]).run()
    assert not at.exception, [e.value for e in at.exception]
    why = [m.value for m in at.markdown if m.value.startswith("**Why?**")]
    assert why, "no plain-language explanation rendered"
    print("\nExplanation:", why[0])

    # Step 12: full replay, as fast as possible, large ticks.
    next(c for c in at.checkbox if c.label.startswith("As fast")).check().run()
    next(s for s in at.slider if s.label == "Flows per replay tick").set_value(2000).run()
    next(b for b in at.button if b.label == "Start / resume").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Replay finished" in s.value for s in at.success)
    metrics = dict(zip(_labels(at.metric), [m.value for m in at.metric]))
    print("Replay metrics:", {k: metrics[k] for k in metrics if k in ("Flows replayed", "Alerts (predicted attacks)")})
