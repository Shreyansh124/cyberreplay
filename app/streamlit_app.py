"""CyberReplay Streamlit dashboard (build steps 11-14).

Tabs:
    1. Overview          - loads the saved best model + preprocessing pipeline,
                           predicts every flow and shows attack frequencies (step 11).
    2. Live replay       - streams flows in capture-time order with adjustable
                           speed and live predictions / alerts (step 12).
    3. Attack patterns   - interactive Plotly views: per-attack feature
                           distributions, detection timeline, frequencies and
                           model comparison (step 13).
    4. Explain detection - pick a detected attack and see the predicted class,
                           SHAP top contributing features and a plain-language
                           reason (step 14).

Run locally:  ``streamlit run app/streamlit_app.py``
Data: the committed sample ``data/sample/cicids2017_sample.csv.gz`` (drawn from
the test split, i.e. unseen by the model) or any uploaded CIC-IDS2017 CSV.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyberreplay import config, explain, replay  # noqa: E402
from cyberreplay.data import prepare_external  # noqa: E402
from cyberreplay.eda import FAMILY_COLORS  # noqa: E402

st.set_page_config(page_title="CyberReplay - intrusion detection", page_icon=":shield:", layout="wide")


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the saved best pipeline and its metadata (cached for the session).

    Returns:
        ``(pipeline, metadata dict)`` or ``(None, None)`` if the model is missing.
    """
    if not config.BEST_MODEL_PATH.exists():
        return None, None
    pipe = joblib.load(config.BEST_MODEL_PATH)
    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    return pipe, meta


@st.cache_resource(show_spinner="Building SHAP explainer...")
def load_explainer(_pipe):
    """Create the SHAP TreeExplainer once per session (``_pipe`` is not hashed)."""
    return explain.make_explainer(_pipe)


@st.cache_data(show_spinner="Loading flows...")
def load_flows(source: str, raw_bytes: bytes | None = None) -> pd.DataFrame:
    """Read and prepare the flows to analyse.

    Args:
        source: ``"sample"`` or an uploaded file name (cache key).
        raw_bytes: Uploaded CSV content, if any.

    Returns:
        Prepared flows in capture order.
    """
    if raw_bytes is not None:
        import io
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1", low_memory=False)
    else:
        df = pd.read_csv(config.SAMPLE_PATH, low_memory=False)
    return replay.order_stream(prepare_external(df))


@st.cache_data(show_spinner="Predicting all flows...")
def predict_all(_pipe, df: pd.DataFrame, features: tuple[str, ...], key: str) -> pd.DataFrame:
    """Predict every flow once (cached per data source ``key``)."""
    return pd.concat([df, replay.predict_batch(_pipe, df, list(features))], axis=1)


def color_map() -> dict[str, str]:
    """Plotly colour map shared by every chart."""
    return dict(FAMILY_COLORS)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
pipe, meta = load_model()
st.title("CyberReplay - ML-based network intrusion detection")
st.caption("CIC-IDS2017 flows classified into 8 attack families; `is_malicious` = predicted family != BENIGN.")

if pipe is None:
    st.error("No trained model found at `models/best_model.joblib`. Regenerate it with "
             "`python -m cyberreplay.pipeline --download` (see README).")
    st.stop()

features: list[str] = meta["features"]

with st.sidebar:
    st.header("Data")
    source = st.radio("Flows to analyse", ["Bundled sample (test split)", "Upload CIC-IDS2017 CSV"], index=0)
    upload = None
    if source.startswith("Upload"):
        upload = st.file_uploader("CSV with CIC-IDS2017 columns", type=["csv"])
    st.header("Replay")
    speed = st.select_slider("Replay speed (capture seconds per real second)",
                             options=[1, 5, 15, 60, 300, 900, 3600], value=300)
    unpaced = st.checkbox("As fast as possible (ignore capture timing)", value=False)
    batch_size = st.slider("Flows per replay tick", 50, 2000, 300, step=50)
    st.header("Model")
    m = meta["test_metrics"]
    st.markdown(f"**{meta['model']}** trained on {meta['train_rows']:,} flows  \n"
                f"Test macro F1 **{m['f1_macro']:.4f}**, accuracy {m['accuracy']:.4f}  \n"
                f"BENIGN false-positive rate {m['benign_fpr']:.3%}")

if upload is not None:
    flows = load_flows(upload.name, upload.getvalue())
    key = upload.name
else:
    flows = load_flows("sample")
    key = "sample"

missing = [f for f in features if f not in flows.columns]
if missing:
    st.error(f"The data is missing {len(missing)} model features, e.g. {missing[:5]}.")
    st.stop()

pred = predict_all(pipe, flows, tuple(features), key)
has_truth = config.TARGET_COL in pred.columns

tab_overview, tab_replay, tab_patterns, tab_explain = st.tabs(
    ["Overview", "Live replay", "Attack patterns", "Explain a detection"])

# --------------------------------------------------------------------------
# 1. Overview (step 11)
# --------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows analysed", f"{len(pred):,}")
    c2.metric("Predicted malicious", f"{pred['pred_is_malicious'].mean():.1%}")
    c3.metric("Attack families detected", int(pred.loc[pred["pred_is_malicious"] == 1, "predicted"].nunique()))
    if has_truth:
        acc = (pred["predicted"] == pred[config.TARGET_COL]).mean()
        c4.metric("Accuracy on these flows", f"{acc:.2%}")
    freq = pred["predicted"].value_counts().reindex(config.CLASS_ORDER, fill_value=0).rename("predicted")
    left, right = st.columns([3, 2])
    with left:
        if has_truth:
            truth = pred[config.TARGET_COL].value_counts().reindex(config.CLASS_ORDER, fill_value=0).rename("true")
            fdf = pd.concat([freq, truth], axis=1).reset_index(names="family").melt(
                id_vars="family", var_name="source", value_name="flows")
            fig = px.bar(fdf, x="family", y="flows", color="source", barmode="group", log_y=True,
                         title="Attack frequencies: predicted vs true (log scale)",
                         color_discrete_map={"predicted": "#c44e52", "true": "#4c72b0"})
        else:
            fig = px.bar(freq.rename_axis("family").reset_index(), x="family", y="predicted", color="family",
                         color_discrete_map=color_map(), log_y=True, title="Predicted attack frequencies")
        st.plotly_chart(fig, width="stretch")
    with right:
        attacks = freq.drop(config.BENIGN)
        fig = px.pie(attacks.rename_axis("family").reset_index(), names="family", values="predicted", hole=0.45,
                     color="family", color_discrete_map=color_map(), title="Share of detected attacks")
        st.plotly_chart(fig, width="stretch")
    if has_truth:
        from sklearn.metrics import f1_score, precision_score, recall_score
        yb, pb = pred[config.BINARY_COL].astype(int), pred["pred_is_malicious"]
        st.markdown(
            f"**Binary is_malicious on these flows** - precision {precision_score(yb, pb, zero_division=0):.4f}, "
            f"recall {recall_score(yb, pb, zero_division=0):.4f}, F1 {f1_score(yb, pb, zero_division=0):.4f}; "
            f"multi-class macro F1 "
            f"{f1_score(pred[config.TARGET_COL], pred['predicted'], average='macro', zero_division=0):.4f}")
    st.subheader("Most recent detections")
    show_cols = [c for c in [config.TIMESTAMP_COL, "Source IP", "Destination IP", "Destination Port", "predicted",
                             "confidence", config.RAW_LABEL_COL] if c in pred.columns]
    st.dataframe(pred.loc[pred["pred_is_malicious"] == 1, show_cols].tail(200), width="stretch", height=300)

# --------------------------------------------------------------------------
# 2. Live replay (step 12)
# --------------------------------------------------------------------------
with tab_replay:
    st.markdown("Flows are replayed in **capture-time order**. The speed slider maps recorded time to real "
                "time (e.g. 300 = five minutes of traffic per second); idle gaps are capped at one second.")
    if "replay_pos" not in st.session_state:
        st.session_state.replay_pos = 0
        st.session_state.replay_log = []
    b1, b2, b3, _ = st.columns([1, 1, 1, 5])
    start = b1.button("Start / resume", type="primary")
    b2.button("Pause")  # any interaction reruns the script, which stops the loop
    if b3.button("Reset"):
        st.session_state.replay_pos = 0
        st.session_state.replay_log = []
    progress = st.progress(st.session_state.replay_pos / len(flows))
    k1, k2, k3, k4 = st.columns(4)
    kpi = [k1.empty(), k2.empty(), k3.empty(), k4.empty()]
    chart_box = st.empty()
    alert_box = st.empty()

    def render_live(log: list[pd.DataFrame], capture_time) -> None:
        """Redraw the live KPIs, timeline and alert table from the replayed batches."""
        if not log:
            chart_box.info("Press **Start / resume** to begin the replay.")
            return
        seen = pd.concat(log)
        kpi[0].metric("Capture time", "-" if pd.isna(capture_time) else str(capture_time))
        kpi[1].metric("Flows replayed", f"{len(seen):,}")
        kpi[2].metric("Alerts (predicted attacks)", f"{int(seen['pred_is_malicious'].sum()):,}")
        kpi[3].metric("Alert rate", f"{seen['pred_is_malicious'].mean():.1%}")
        tl = replay.detection_timeline(seen, freq="1min")
        fig = px.bar(tl, x="time", y="flows", color="predicted", color_discrete_map=color_map(),
                     title="Live detection timeline (flows per minute by predicted family)")
        fig.update_layout(height=330, margin=dict(t=40, b=10), bargap=0)
        chart_box.plotly_chart(fig, width="stretch", key=f"live_{len(seen)}")
        alerts = seen.loc[seen["pred_is_malicious"] == 1]
        cols = [c for c in [config.TIMESTAMP_COL, "Source IP", "Destination IP", "Destination Port", "predicted",
                            "confidence", config.RAW_LABEL_COL] if c in alerts.columns]
        alert_box.dataframe(alerts[cols].tail(15).iloc[::-1], width="stretch", height=300)

    last_time = st.session_state.replay_log[-1][config.TIMESTAMP_COL].iloc[-1] if st.session_state.replay_log \
        else pd.NaT
    render_live(st.session_state.replay_log, last_time)
    if start:
        stream = replay.replay(flows, pipe, features, batch_size=batch_size, speed=0 if unpaced else speed,
                               start=st.session_state.replay_pos)
        for step in stream:
            st.session_state.replay_log.append(step.batch)
            st.session_state.replay_pos = min(st.session_state.replay_pos + batch_size, len(flows))
            progress.progress(step.progress)
            render_live(st.session_state.replay_log, step.capture_time)
            if unpaced:
                time.sleep(0.02)  # keep the UI responsive
        st.success("Replay finished.")

# --------------------------------------------------------------------------
# 3. Attack patterns (step 13)
# --------------------------------------------------------------------------
with tab_patterns:
    label_col = config.TARGET_COL if has_truth else "predicted"
    st.markdown(f"Colour = **{'true family' if has_truth else 'predicted family'}**. All charts are interactive "
                "(zoom, hover, click legend entries to hide families).")
    default_feats = [f for f in ["Fwd Packet Length Max", "Init_Win_bytes_backward", "Flow Duration",
                                 "Total Length of Fwd Packets"] if f in features] or features[:4]
    pc1, pc2 = st.columns(2)
    with pc1:
        feat = st.selectbox("Feature distribution per attack family", features,
                            index=features.index(default_feats[0]))
        fig = px.box(pred, x=label_col, y=feat, color=label_col, color_discrete_map=color_map(), log_y=True,
                     category_orders={label_col: config.CLASS_ORDER}, points=False,
                     title=f"{feat} by family (log scale)")
        st.plotly_chart(fig, width="stretch")
    with pc2:
        fx = st.selectbox("Scatter: x feature", features, index=features.index(default_feats[1]))
        fy = st.selectbox("Scatter: y feature", features, index=features.index(default_feats[2]))
        sc = pred.groupby(label_col, group_keys=False).head(1500)
        fig = px.scatter(sc, x=fx, y=fy, color=label_col, color_discrete_map=color_map(), log_x=True, log_y=True,
                         opacity=0.6, title=f"{fy} vs {fx}", hover_data=["predicted", "confidence"])
        st.plotly_chart(fig, width="stretch")
    tl = replay.detection_timeline(pred, freq="5min")
    tl = tl[tl["predicted"] != config.BENIGN]
    fig = px.scatter(tl, x="time", y="predicted", size="flows", color="predicted", color_discrete_map=color_map(),
                     category_orders={"predicted": config.CLASS_ORDER},
                     title="Detection timeline across the capture week (bubble = detected flows per 5 min)")
    st.plotly_chart(fig, width="stretch")
    if has_truth:
        ct = pd.crosstab(pred[config.TARGET_COL], pred["predicted"]).reindex(
            index=config.CLASS_ORDER, columns=config.CLASS_ORDER, fill_value=0)
        norm = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
        fig = px.imshow(norm, text_auto=".2f", color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="predicted", y="true", color="recall"),
                        title="Confusion matrix on the loaded flows (row-normalised)")
        st.plotly_chart(fig, width="stretch")
    comp_path = config.RESULTS_DIR / "test_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        long = comp.melt(id_vars="model", value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                      "minority_recall"], var_name="metric", value_name="score")
        fig = px.bar(long, x="metric", y="score", color="model", barmode="group",
                     title="Model comparison on the full held-out test split (from reports/results)")
        fig.update_yaxes(range=[max(0, long["score"].min() - 0.05), 1])
        st.plotly_chart(fig, width="stretch")
    f1_path = config.RESULTS_DIR / "per_class_f1.csv"
    if f1_path.exists():
        f1 = pd.read_csv(f1_path, index_col=0)
        fig = px.imshow(f1, text_auto=".3f", color_continuous_scale="RdYlGn", zmin=0, zmax=1, aspect="auto",
                        title="Per-class test F1 by model")
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
# 4. Explain a detection (step 14)
# --------------------------------------------------------------------------
with tab_explain:
    detected = pred[pred["pred_is_malicious"] == 1]
    if detected.empty:
        st.info("No attacks detected in the loaded flows.")
    else:
        fams = [f for f in config.CLASS_ORDER if f in set(detected["predicted"])]
        ec1, ec2 = st.columns([1, 2])
        fam = ec1.selectbox("Detected family", fams)
        cand = detected[detected["predicted"] == fam]
        options = cand.index[:500].tolist()

        def describe(i: int) -> str:
            """Label for a detection in the picker."""
            r = pred.loc[i]
            ts = r[config.TIMESTAMP_COL]
            src = f" {r['Source IP']} -> {r['Destination IP']}:{int(r['Destination Port'])}" \
                if "Source IP" in pred.columns else ""
            return f"#{i} {'' if pd.isna(ts) else ts}{src} ({r['confidence']:.0%})"

        idx = ec2.selectbox("Detection", options, format_func=describe)
        explainer = load_explainer(pipe)
        benign_med = pd.Series(meta["family_medians"][config.BENIGN])
        res = explain.explain_flow(pipe, explainer, pred.loc[[idx]], features, benign_med, top_k=8)
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted class", res["predicted"])
        r2.metric("Confidence", f"{res['confidence']:.1%}")
        if has_truth:
            r3.metric("True label", f"{pred.loc[idx, config.RAW_LABEL_COL]}")
        st.markdown(f"**Why?** {res['reason']}")
        x1, x2 = st.columns([3, 2])
        with x1:
            con = res["contributions"].iloc[::-1].copy()
            con["value_txt"] = [explain.pretty_value(f, v) for f, v in zip(con["feature"], con["value"])]
            con["benign_txt"] = [explain.pretty_value(f, v) for f, v in zip(con["feature"], con["benign_median"])]
            fig = go.Figure(go.Bar(
                x=con["shap"], y=con["feature"], orientation="h",
                marker_color=np.where(con["shap"] > 0, "#c44e52", "#4c72b0"),
                customdata=np.stack([con["value_txt"], con["benign_txt"]], axis=1),
                hovertemplate="%{y}<br>value %{customdata[0]}<br>benign median %{customdata[1]}"
                              "<br>SHAP %{x:.3f}<extra></extra>"))
            fig.update_layout(title=f"Top SHAP contributions towards '{res['predicted']}' "
                                    "(red pushes towards it, blue away)", height=420, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
        with x2:
            probs = res["probabilities"].sort_values(ascending=True)
            fig = px.bar(x=probs.values, y=probs.index, orientation="h", title="Class probabilities",
                         color=probs.index, color_discrete_map=color_map())
            fig.update_layout(showlegend=False, height=420, margin=dict(t=50), xaxis_title="probability",
                              yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        table = res["contributions"].assign(
            value=lambda d: [explain.pretty_value(f, v) for f, v in zip(d["feature"], d["value"])],
            benign_median=lambda d: [explain.pretty_value(f, v) for f, v in zip(d["feature"], d["benign_median"])])
        st.dataframe(table, width="stretch", hide_index=True)
