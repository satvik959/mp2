from __future__ import annotations

import html
import pickle
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from dataset.dataset_builder import DatasetBuilder

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    from agents import run_agents
    AGENTS_AVAILABLE = True
except Exception:
    AGENTS_AVAILABLE = False


def _to_abs_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).parent / path


@st.cache_resource
def load_tf_model(model_path: str):
    if tf is None:
        raise RuntimeError("TensorFlow is not available. Install tensorflow to load .keras models.")
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _build_feature_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    timestamp_source = (
        pd.to_numeric(raw_df["timestamp"], errors="coerce")
        if "timestamp" in raw_df.columns
        else pd.to_numeric(raw_df["Time"], errors="coerce")
        if "Time" in raw_df.columns
        else pd.Series(np.arange(len(raw_df), dtype=float))
    )
    src_source = (
        raw_df["src_ip"]
        if "src_ip" in raw_df.columns
        else raw_df["Source"]
        if "Source" in raw_df.columns
        else pd.Series(["0.0.0.0"] * len(raw_df))
    )
    dst_source = (
        raw_df["dst_ip"]
        if "dst_ip" in raw_df.columns
        else raw_df["Destination"]
        if "Destination" in raw_df.columns
        else pd.Series(["0.0.0.0"] * len(raw_df))
    )
    protocol_source = (
        raw_df["protocol"]
        if "protocol" in raw_df.columns
        else raw_df["Protocol"]
        if "Protocol" in raw_df.columns
        else pd.Series(["TCP"] * len(raw_df))
    )
    packet_size_source = (
        raw_df["packet_size"]
        if "packet_size" in raw_df.columns
        else raw_df["Length"]
        if "Length" in raw_df.columns
        else pd.Series([0] * len(raw_df))
    )
    src_port_source = raw_df["src_port"] if "src_port" in raw_df.columns else pd.Series([None] * len(raw_df))
    dst_port_source = raw_df["dst_port"] if "dst_port" in raw_df.columns else pd.Series([None] * len(raw_df))
    flags_source = raw_df["flags"] if "flags" in raw_df.columns else pd.Series(["NONE"] * len(raw_df))

    packet_df = pd.DataFrame(
        {
            "timestamp": timestamp_source,
            "src_ip": src_source,
            "dst_ip": dst_source,
            "protocol": protocol_source,
            "packet_size": packet_size_source,
            "src_port": src_port_source,
            "dst_port": dst_port_source,
            "flags": flags_source,
        }
    )

    builder = DatasetBuilder()
    feature_df = builder.build_dataset(packet_df, window_size=1.0)

    feature_df["label"] = raw_df["label"].values if "label" in raw_df.columns else "benign"
    feature_df["info"] = (
        raw_df["info"].values
        if "info" in raw_df.columns
        else raw_df["Info"].values
        if "Info" in raw_df.columns
        else "packet"
    )
    feature_df["flags"] = raw_df["flags"].values if "flags" in raw_df.columns else "NONE"

    return feature_df


def _engineer_features(feature_df: pd.DataFrame, preprocessors: Any) -> np.ndarray:
    if isinstance(preprocessors, dict):
        struct_features = [
            "packet_size",
            "packet_rate",
            "connection_count",
            "avg_packet_size",
            "src_port",
            "dst_port",
        ]

        x_struct = feature_df[struct_features].fillna(0).values
        x_struct_scaled = preprocessors["scaler_struct"].transform(x_struct)

        x_protocol = preprocessors["protocol_encoder"].transform(feature_df["protocol"]).reshape(-1, 1)
        x_protocol_scaled = preprocessors["scaler_protocol"].transform(x_protocol)

        flags_clean = feature_df["flags"].fillna("PA").astype(str)
        known_flags = set(preprocessors["flags_encoder"].classes_)
        flags_clean = flags_clean.apply(lambda x: x if x in known_flags else "PA")
        x_flags = preprocessors["flags_encoder"].transform(flags_clean).reshape(-1, 1)
        x_flags_scaled = preprocessors["scaler_flags"].transform(x_flags)

        info_text = feature_df["info"].fillna("").astype(str).values
        x_tfidf = preprocessors["vectorizer"].transform(info_text).toarray()

        time_delta = np.diff(feature_df["timestamp"].values, prepend=feature_df["timestamp"].values[0])
        x_stats = np.column_stack(
            [
                np.log1p(feature_df["packet_size"].values),
                time_delta,
                (feature_df["src_port"].fillna(0).values % 256) / 256,
                (feature_df["dst_port"].fillna(0).values % 256) / 256,
            ]
        )
        x_stats_scaled = preprocessors["scaler_stats"].transform(x_stats)

        return np.hstack([x_struct_scaled, x_protocol_scaled, x_flags_scaled, x_tfidf, x_stats_scaled])

    x_combined, _ = preprocessors.preprocess_data(feature_df, fit=False)
    return x_combined


def _prepare_inputs(x_combined: np.ndarray, feature_df: pd.DataFrame, model) -> Tuple[np.ndarray, np.ndarray, int]:
    sequence_input_shape, _graph_input_shape = model.input_shape
    seq_length = int(sequence_input_shape[1] or 1)

    if seq_length <= 1:
        x_seq = x_combined.reshape(len(x_combined), 1, x_combined.shape[1]).astype(np.float32)
    else:
        seqs = []
        for i in range(len(x_combined) - seq_length + 1):
            seqs.append(x_combined[i : i + seq_length])
        x_seq = np.array(seqs, dtype=np.float32)

    src_hash = feature_df["src_ip"].astype(str).apply(lambda x: hash(x) % 1000).to_numpy(dtype=np.float32)
    dst_hash = feature_df["dst_ip"].astype(str).apply(lambda x: hash(x) % 1000).to_numpy(dtype=np.float32)
    edge_activity = feature_df["connection_count"].to_numpy(dtype=np.float32)
    packet_rate = feature_df["packet_rate"].to_numpy(dtype=np.float32)
    graph_all = np.stack([src_hash, dst_hash, edge_activity, packet_rate], axis=1)
    graph_all /= np.array([1000.0, 1000.0, 50.0, 50.0], dtype=np.float32)

    g_seq = graph_all[seq_length - 1 :] if seq_length > 1 else graph_all

    if len(x_seq) != len(g_seq):
        min_len = min(len(x_seq), len(g_seq))
        x_seq = x_seq[:min_len]
        g_seq = g_seq[:min_len]

    return x_seq, g_seq, seq_length


def _is_benign(label: Any) -> bool:
    text = str(label).strip().lower()
    return text == "benign" or text == "1"


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-0: #09121f;
                --bg-1: #0f1d2f;
                --bg-2: #11263a;
                --panel: rgba(12, 24, 39, 0.84);
                --panel-soft: rgba(17, 34, 54, 0.72);
                --text: #e5edf6;
                --muted: #9ab0c7;
                --accent: #00c2a8;
                --accent-warm: #f59e0b;
                --danger: #ef4444;
                --ok: #22c55e;
                --border: rgba(154, 176, 199, 0.2);
            }
            .stApp {
                color: var(--text);
                background:
                    radial-gradient(1200px 540px at 8% -12%, rgba(0, 194, 168, 0.22), transparent 62%),
                    radial-gradient(900px 500px at 90% 0%, rgba(245, 158, 11, 0.19), transparent 64%),
                    linear-gradient(160deg, var(--bg-0), var(--bg-1) 45%, var(--bg-2));
            }
            [data-testid="stAppViewContainer"] { background: transparent; }
            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0d1b2b 0%, #102239 100%);
                border-right: 1px solid var(--border);
            }
            [data-testid="stSidebar"] * { color: var(--text); }
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stMarkdownContainer"] span,
            [data-testid="stText"] {
                color: var(--text);
            }
            .dark-card {
                background: var(--panel);
                backdrop-filter: blur(6px);
                border-radius: 14px;
                padding: 14px 16px;
                margin: 8px 0;
                border: 1px solid var(--border);
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.22);
            }
            .hero {
                position: relative;
                padding: 20px 22px;
                border-radius: 16px;
                margin: 4px 0 14px 0;
                overflow: hidden;
                background: linear-gradient(130deg, rgba(0, 194, 168, 0.18), rgba(245, 158, 11, 0.18));
                border: 1px solid rgba(154, 176, 199, 0.26);
            }
            .hero h2 {
                margin: 0;
                font-size: 1.65rem;
                letter-spacing: 0.3px;
                color: #f2f8ff;
            }
            .hero p {
                margin: 8px 0 0 0;
                color: #d6e3f1;
            }
            .stage-card {
                background: var(--panel-soft);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 12px 14px;
                margin-bottom: 10px;
            }
            .stage-title {
                font-weight: 700;
                color: #f8fcff;
                letter-spacing: 0.2px;
                margin-bottom: 3px;
            }
            .stage-note {
                font-size: 0.92rem;
                color: var(--muted);
                margin: 0;
            }
            .metric-chip {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.8rem;
                margin-right: 6px;
                font-weight: 600;
            }
            .chip-green { background: rgba(34, 197, 94, 0.2); color: #c0ffd8; border: 1px solid rgba(34,197,94,0.35); }
            .chip-red { background: rgba(239, 68, 68, 0.2); color: #ffd4d4; border: 1px solid rgba(239,68,68,0.35); }
            .chip-gray { background: rgba(148, 163, 184, 0.2); color: #d7e2ee; border: 1px solid rgba(148,163,184,0.3); }
            .chip-amber { background: rgba(245, 158, 11, 0.2); color: #ffe5bd; border: 1px solid rgba(245,158,11,0.35); }
            .threat-highlight {
                border: 1px solid rgba(239, 68, 68, 0.72);
                box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.3) inset;
            }
            .log-box {
                background: rgba(8, 16, 30, 0.9);
                color: #c9d6e5;
                padding: 10px;
                border-radius: 10px;
                height: 250px;
                overflow-y: auto;
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 0.85rem;
                border: 1px solid var(--border);
                line-height: 1.45;
            }
            .section-title {
                color: #f9fcff;
                font-weight: 700;
                margin-top: 6px;
            }
            .mini-caption {
                color: var(--muted);
                font-size: 0.88rem;
                margin-top: -4px;
            }
            .agent-card {
                background: linear-gradient(175deg, rgba(12, 25, 40, 0.92), rgba(15, 32, 48, 0.82));
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 14px 14px 10px 14px;
                margin-bottom: 10px;
            }
            .agent-card h4 {
                margin: 0 0 4px 0;
                color: #f8fcff;
                letter-spacing: 0.2px;
            }
            .agent-card p {
                margin: 6px 0;
            }
            div[data-testid="stMetric"] {
                background: rgba(9, 20, 33, 0.86);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 10px 12px;
            }
            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] div {
                color: #e9f2fb;
            }
            .stDataFrame, .stTable {
                background: rgba(9, 20, 33, 0.86);
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _save_uploaded_csv(uploaded_file) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def _render_analysis_card(item: Dict[str, Any]) -> None:
    analyzer = item.get("analyzer_output", {})
    remediation = item.get("remediation_output", {})
    evidence = analyzer.get("evidence", []) or []
    actions = remediation.get("recommended_actions", []) or []

    priority = str(remediation.get("priority", "medium")).lower()
    priority_class = "chip-red" if priority in {"high", "critical"} else "chip-gray"

    evidence_html = "".join(f"<li>{html.escape(str(e))}</li>" for e in evidence) or "<li>No evidence provided</li>"
    actions_html = "".join(f"<li>{html.escape(str(a))}</li>" for a in actions) or "<li>No action provided</li>"

    flow = f"{html.escape(str(item.get('src_ip', 'N/A')))} -> {html.escape(str(item.get('dst_ip', 'N/A')))}"
    fallback_used = bool(item.get("fallback_used", False))
    error_hint = str(item.get("error_hint", "")).strip()
    fallback_chip = '<span class="metric-chip chip-amber">Fallback</span>' if fallback_used else ""
    fallback_note = f"<p><b>Agent Notice:</b> {html.escape(error_hint)}</p>" if error_hint else ""
    st.markdown(
        f"""
        <div class="agent-card">
            <h4>{html.escape(str(item.get('prediction', 'unknown')).upper())}</h4>
            <div>
                <span class="metric-chip chip-red">Threat</span>
                <span class="metric-chip {priority_class}">Priority: {html.escape(priority.upper())}</span>
                <span class="metric-chip chip-amber">Confidence: {html.escape(str(analyzer.get('confidence', 'low')).upper())}</span>
                {fallback_chip}
            </div>
            <p><b>Flow:</b> {flow}</p>
            <p><b>Root Cause:</b> {html.escape(str(analyzer.get('cause', 'Unknown')))}</p>
            {fallback_note}
            <p><b>Evidence:</b></p>
            <ul>{evidence_html}</ul>
            <p><b>Remediation:</b></p>
            <ul>{actions_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_log_box(lines: List[str]) -> str:
    return (
        f'<div class="log-box">{"<br>".join(html.escape(line) for line in lines) if lines else "No events yet."}</div>'
    )


def _render_pipeline_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h2>Hybrid Intrusion Detection + Agentic Response Console</h2>
            <p>
                Stage 1 runs LSTM + GCN inference only. Stage 2 is optional and consumes API tokens only for selected anomalies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_detection(
    csv_path: Path,
    model_path: Path,
    label_encoder_path: Path,
    preprocessor_path: Path,
    batch_size: int,
    run_agent_analysis: bool,
    llm_model: str,
    max_agent_items: int,
) -> Dict[str, Any]:
    raw_df = pd.read_csv(csv_path)

    model = load_tf_model(str(model_path))
    label_encoder = load_pickle(str(label_encoder_path))
    preprocessors = load_pickle(str(preprocessor_path))

    feature_df = _build_feature_df(raw_df)
    x_combined = _engineer_features(feature_df, preprocessors)
    x_seq, g_seq, seq_length = _prepare_inputs(x_combined, feature_df, model)

    all_pred_indices: List[int] = []
    prediction_rows: List[Dict[str, Any]] = []
    batch_logs: List[str] = []

    num_batches = (len(x_seq) + batch_size - 1) // batch_size
    progress = st.progress(0, text="Running prediction batches...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(x_seq))

        probs = model.predict([x_seq[start_idx:end_idx], g_seq[start_idx:end_idx]], verbose=0)
        pred_idx = np.argmax(probs, axis=1)
        pred_labels = label_encoder.inverse_transform(pred_idx)
        non_benign_in_batch = int(np.sum([0 if _is_benign(x) else 1 for x in pred_labels]))
        batch_top = Counter([str(x) for x in pred_labels]).most_common(2)
        top_text = ", ".join([f"{label}:{count}" for label, count in batch_top]) if batch_top else "n/a"
        batch_logs.append(
            f"Batch {batch_idx + 1}/{num_batches} | Sequences: {end_idx - start_idx} | "
            f"Suspicious: {non_benign_in_batch} | Top: {top_text}"
        )

        all_pred_indices.extend(pred_idx.tolist())

        for i, (label, prob_row) in enumerate(zip(pred_labels, probs)):
            seq_idx = start_idx + i
            packet_idx = seq_idx + seq_length
            row_idx = min(packet_idx - 1, len(raw_df) - 1)
            feature_idx = min(seq_idx, len(feature_df) - 1)
            confidence = float(np.max(prob_row))
            label_text = str(label)

            src_ip = raw_df.iloc[row_idx]["src_ip"] if "src_ip" in raw_df.columns else "N/A"
            dst_ip = raw_df.iloc[row_idx]["dst_ip"] if "dst_ip" in raw_df.columns else "N/A"
            protocol = raw_df.iloc[row_idx]["protocol"] if "protocol" in raw_df.columns else "N/A"
            flags = raw_df.iloc[row_idx]["flags"] if "flags" in raw_df.columns else "NONE"
            event_time = (
                str(raw_df.iloc[row_idx]["timestamp"])
                if "timestamp" in raw_df.columns
                else str(raw_df.iloc[row_idx]["Time"])
                if "Time" in raw_df.columns
                else datetime.now().strftime("%H:%M:%S")
            )

            prediction_rows.append(
                {
                    "packet_num": int(packet_idx),
                    "timestamp": event_time,
                    "predicted_label": label_text,
                    "confidence": confidence,
                    "is_benign": _is_benign(label_text),
                    "src_ip": str(src_ip),
                    "dst_ip": str(dst_ip),
                    "protocol": str(protocol),
                    "flags": str(flags),
                    "packet_rate": float(feature_df.iloc[feature_idx]["packet_rate"]) if "packet_rate" in feature_df.columns else 0.0,
                }
            )

        progress.progress(int(((batch_idx + 1) / max(1, num_batches)) * 100), text=f"Processed batch {batch_idx + 1}/{num_batches}")

    progress.empty()

    pred_labels = [str(x) for x in label_encoder.inverse_transform(np.array(all_pred_indices))]
    distribution = pd.Series(pred_labels).value_counts().rename_axis("class").reset_index(name="count")

    non_benign_rows = [row for row in prediction_rows if not row["is_benign"]]
    avg_confidence = float(np.mean([row["confidence"] for row in prediction_rows])) if prediction_rows else 0.0

    agent_outputs: List[Dict[str, Any]] = []
    if run_agent_analysis and AGENTS_AVAILABLE and non_benign_rows:
        for row in non_benign_rows[:max_agent_items]:
            payload = {
                "prediction": row["predicted_label"],
                "packet_rate": int(len(x_seq)),
                "protocol": row["protocol"],
                "flags_pattern": [row["flags"]],
                "connection_count": int(feature_df["connection_count"].mean()) if "connection_count" in feature_df.columns else 0,
                "batch_summary": f"Anomaly around packet {row['packet_num']}",
                "avg_packet_size": float(feature_df["avg_packet_size"].mean()) if "avg_packet_size" in feature_df.columns else 0.0,
            }
            result = run_agents(payload, llm_model=llm_model)
            agent_outputs.append(
                {
                    "packet_num": row["packet_num"],
                    "predicted_label": row["predicted_label"],
                    "anomaly_type": result.get("analyzer_output", {}).get("anomaly_type", "unknown"),
                    "cause": result.get("analyzer_output", {}).get("cause", "unknown"),
                    "confidence": result.get("analyzer_output", {}).get("confidence", "low"),
                    "priority": result.get("remediation_output", {}).get("priority", "medium"),
                    "actions": "; ".join(result.get("remediation_output", {}).get("recommended_actions", [])),
                }
            )

    return {
        "total_packets": len(raw_df),
        "total_sequences": len(prediction_rows),
        "benign_count": sum(1 for row in prediction_rows if row["is_benign"]),
        "non_benign_count": len(non_benign_rows),
        "avg_confidence": avg_confidence,
        "prediction_rows": prediction_rows,
        "batch_logs": batch_logs,
        "distribution": distribution,
        "non_benign_table": pd.DataFrame(non_benign_rows),
        "agent_table": pd.DataFrame(agent_outputs),
    }


def main() -> None:
    st.set_page_config(page_title="Network IDS Dashboard", page_icon="N", layout="wide")
    _inject_theme()
    _render_pipeline_hero()

    if "detection_result" not in st.session_state:
        st.session_state["detection_result"] = None
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "detected_threats" not in st.session_state:
        st.session_state["detected_threats"] = []

    st.markdown(
        """
        <div class="stage-card">
            <div class="stage-title">Stage 1: Deep Learning Detection (No API Token Usage)</div>
            <p class="stage-note">Run only LSTM + GCN to show core detection pipeline and suspicious flows.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_csv = st.file_uploader("Upload CSV packet data", type=["csv"], key="detection_csv_uploader")

    c1, c2 = st.columns(2)
    with c1:
        csv_path_text = st.text_input("Or CSV path", value="data/small_dataset.csv")
        model_path_text = st.text_input("Model path", value="artifacts/lstm_gcn_model.keras")
    with c2:
        label_encoder_path_text = st.text_input("Label encoder path", value="artifacts/label_encoder.pkl")
        preprocessor_path_text = st.text_input("Preprocessor path", value="artifacts/preprocessor.pkl")

    batch_size = st.number_input("Batch size", min_value=1, max_value=5000, value=10, step=1)
    run_btn = st.button("Run Stage 1: Detection", type="primary", use_container_width=True, key="run_detection_btn")

    if run_btn:
        csv_path = _save_uploaded_csv(uploaded_csv) if uploaded_csv is not None else _to_abs_path(csv_path_text)
        model_path = _to_abs_path(model_path_text)
        label_encoder_path = _to_abs_path(label_encoder_path_text)
        preprocessor_path = _to_abs_path(preprocessor_path_text)

        missing = [
            str(path)
            for path in [csv_path, model_path, label_encoder_path, preprocessor_path]
            if not path.exists()
        ]

        if missing:
            st.error("Missing required file(s):")
            for item in missing:
                st.write(f"- {item}")
        else:
            with st.spinner("Running LSTM + GCN inference..."):
                try:
                    result = run_detection(
                        csv_path=csv_path,
                        model_path=model_path,
                        label_encoder_path=label_encoder_path,
                        preprocessor_path=preprocessor_path,
                        batch_size=int(batch_size),
                        run_agent_analysis=False,
                        llm_model="groq/llama3-8b-8192",
                        max_agent_items=3,
                    )
                    st.session_state["detection_result"] = result
                    st.session_state["detected_threats"] = result.get("non_benign_table", pd.DataFrame()).to_dict(orient="records")
                    st.session_state["analysis_result"] = None
                    st.success("Stage 1 complete. Stage 2 can now analyze suspicious traffic.")
                except Exception as exc:
                    st.exception(exc)

    result = st.session_state.get("detection_result")
    if not result:
        st.info("Run Stage 1 first. This will produce predictions without using LLM tokens.")
    else:
        benign_count = int(result.get("benign_count", 0))
        non_benign_count = int(result.get("non_benign_count", 0))
        total_seq = int(result.get("total_sequences", 0))
        avg_conf = float(result.get("avg_confidence", 0.0))

        benign_pct = (100.0 * benign_count / total_seq) if total_seq else 0.0
        malicious_pct = (100.0 * non_benign_count / total_seq) if total_seq else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total packets", result.get("total_packets", total_seq))
        m2.metric("Benign", f"{benign_count} ({benign_pct:.1f}%)")
        m3.metric("Suspicious", f"{non_benign_count} ({malicious_pct:.1f}%)")
        m4.metric("Average confidence", f"{avg_conf:.2%}")

        st.markdown('<div class="section-title">Detection Execution Log</div><div class="mini-caption">Stage-wise progress similar to terminal output</div>', unsafe_allow_html=True)
        detection_lines = result.get("batch_logs", [])[-40:]
        st.markdown(_format_log_box(detection_lines), unsafe_allow_html=True)

        st.markdown('<div class="section-title">Attack Distribution</div>', unsafe_allow_html=True)
        st.dataframe(result["distribution"], use_container_width=True)
        if not result["distribution"].empty:
            st.bar_chart(result["distribution"].set_index("class"))

        st.markdown('<div class="section-title">Suspicious Traffic Table</div>', unsafe_allow_html=True)
        non_benign_table = result["non_benign_table"].copy()
        if non_benign_table.empty:
            st.markdown('<div class="dark-card"><span class="metric-chip chip-green">No suspicious traffic detected</span></div>', unsafe_allow_html=True)
        else:
            non_benign_table["flow"] = non_benign_table["src_ip"].astype(str) + " -> " + non_benign_table["dst_ip"].astype(str)
            display_df = non_benign_table[["flow", "predicted_label", "confidence", "packet_rate"]].rename(
                columns={
                    "flow": "Source -> Destination",
                    "predicted_label": "Predicted Attack Type",
                    "confidence": "Confidence",
                    "packet_rate": "Packet Rate",
                }
            )
            st.dataframe(display_df, use_container_width=True)

        st.markdown('<div class="section-title">Top Threat Snapshot</div>', unsafe_allow_html=True)
        if non_benign_table.empty:
            st.markdown('<div class="dark-card"><span class="metric-chip chip-green">No active threat</span></div>', unsafe_allow_html=True)
        else:
            top = non_benign_table.sort_values("confidence", ascending=False).iloc[0]
            st.markdown(
                f"""
                <div class="dark-card threat-highlight">
                    <span class="metric-chip chip-red">Top Threat</span>
                    <p><b>Flow:</b> {html.escape(str(top['src_ip']))} -> {html.escape(str(top['dst_ip']))}</p>
                    <p><b>Attack:</b> {html.escape(str(top['predicted_label']))}</p>
                    <p><b>Confidence:</b> {float(top['confidence']):.2%}</p>
                    <p><b>Packet Rate:</b> {float(top.get('packet_rate', 0.0)):.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="stage-card">
            <div class="stage-title">Stage 2: Agentic AI (Token-Aware)</div>
            <p class="stage-note">Run analyzer and remediation only after reviewing Stage 1 detections.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    llm_col1, llm_col2, llm_col3 = st.columns([2, 1, 1])
    with llm_col1:
        llm_model = st.text_input("LLM model", value="groq/llama3-8b-8192", key="analysis_llm_model")
    with llm_col2:
        max_agent_items = st.number_input("Max anomaly analyses", min_value=1, max_value=50, value=5, step=1, key="analysis_max_items")
    with llm_col3:
        agent_call_delay = st.number_input(
            "Delay per call (s)",
            min_value=0.0,
            max_value=10.0,
            value=0.8,
            step=0.2,
            key="analysis_call_delay",
            help="Adds a pause between agent calls to reduce 429 rate-limit errors.",
        )

    run_analysis_btn = st.button(
        "Run Stage 2: Agent Analysis and Remediation",
        type="primary",
        use_container_width=True,
        key="run_analysis_btn",
    )

    threats = st.session_state.get("detected_threats", [])
    if threats:
        st.caption(f"{len(threats)} suspicious records available. Only first {int(max_agent_items)} will consume tokens.")
    else:
        st.warning("No suspicious traffic available yet. Complete Stage 1 first.")

    if threats and run_analysis_btn:
        if not AGENTS_AVAILABLE:
            st.error("agents.py dependencies are unavailable. Install crewai and provider keys first.")
        else:
            analysis_results: List[Dict[str, Any]] = []
            agent_log_lines: List[str] = []
            log_placeholder = st.empty()
            progress = st.progress(0, text="Running agent analysis...")
            total_agent_tasks = min(len(threats), int(max_agent_items))

            for idx, row in enumerate(threats[: int(max_agent_items)]):
                payload = {
                    "prediction": row.get("predicted_label", "unknown"),
                    "packet_rate": int(row.get("packet_rate", 0)),
                    "protocol": row.get("protocol", "UNKNOWN"),
                    "flags_pattern": [row.get("flags", "NONE")],
                    "connection_count": 0,
                    "batch_summary": f"Anomaly around packet {row.get('packet_num', 'N/A')}",
                    "avg_packet_size": 0.0,
                }
                agent_log_lines.append(
                    f"[{idx + 1}/{total_agent_tasks}] Analyzing packet {row.get('packet_num', 'N/A')} | "
                    f"{row.get('src_ip', 'N/A')} -> {row.get('dst_ip', 'N/A')} | class={row.get('predicted_label', 'unknown')}"
                )
                log_placeholder.markdown(_format_log_box(agent_log_lines[-20:]), unsafe_allow_html=True)

                agent_result = run_agents(payload, llm_model=llm_model)
                analyzer = agent_result.get("analyzer_output", {})
                remediation = agent_result.get("remediation_output", {})
                model_used = str(agent_result.get("model_used", "none"))
                fallback_used = bool(agent_result.get("fallback_used", False))
                attempted_models = agent_result.get("attempted_models", []) or []
                error_meta = agent_result.get("error", {}) or {}
                error_type = str(error_meta.get("type", "")).strip()
                error_hint = str(error_meta.get("hint", "")).strip()
                analysis_results.append(
                    {
                        "prediction": row.get("predicted_label", "unknown"),
                        "src_ip": row.get("src_ip", "N/A"),
                        "dst_ip": row.get("dst_ip", "N/A"),
                        "analyzer_output": analyzer,
                        "remediation_output": remediation,
                        "model_used": model_used,
                        "fallback_used": fallback_used,
                        "attempted_models": attempted_models,
                        "error_type": error_type,
                        "error_hint": error_hint,
                    }
                )

                agent_log_lines.append(
                    f"   -> model={model_used}{' (fallback)' if fallback_used else ''} | "
                    f"attempted={','.join(str(m) for m in attempted_models) if attempted_models else 'none'} | "
                    f"cause={analyzer.get('cause', 'unknown')} | priority={remediation.get('priority', 'medium')}"
                    f"{f' | error={error_type}' if error_type else ''}"
                )
                log_placeholder.markdown(_format_log_box(agent_log_lines[-20:]), unsafe_allow_html=True)
                progress.progress(
                    int(((idx + 1) / max(1, total_agent_tasks)) * 100),
                    text=f"Processed {idx + 1}/{total_agent_tasks} agent tasks",
                )

                if idx < total_agent_tasks - 1 and float(agent_call_delay) > 0:
                    time.sleep(float(agent_call_delay))

            progress.empty()
            st.session_state["analysis_result"] = analysis_results
            fallback_count = sum(1 for item in analysis_results if item.get("fallback_used"))
            if analysis_results and fallback_count == len(analysis_results):
                top_hints = [
                    str(item.get("error_hint", "")).strip()
                    for item in analysis_results
                    if str(item.get("error_hint", "")).strip()
                ]
                hint_text = top_hints[0] if top_hints else "Agent provider requests failed. Check provider key, model access, and quota."
                st.error(f"Stage 2 failed: all agent calls fell back. {hint_text}")
            elif fallback_count:
                st.warning(
                    f"Stage 2 completed with partial fallback ({fallback_count}/{len(analysis_results)} calls)."
                )
            else:
                st.success("Stage 2 complete.")

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result:
        st.markdown('<div class="section-title">Agent Findings and Recommended Actions</div>', unsafe_allow_html=True)

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        high_priority = sum(
            1
            for item in analysis_result
            if str(item.get("remediation_output", {}).get("priority", "medium")).lower() in {"high", "critical"}
        )
        summary_col1.metric("Cases analyzed", len(analysis_result))
        summary_col2.metric("High/Critical priority", high_priority)
        fallback_count = sum(1 for item in analysis_result if item.get("fallback_used"))
        summary_col3.metric("Fallback hits", fallback_count)
        if fallback_count:
            hints = [str(item.get("error_hint", "")).strip() for item in analysis_result if item.get("fallback_used")]
            hints = [hint for hint in hints if hint]
            if hints:
                st.warning(f"Some agent calls used fallback logic. Most common reason: {hints[0]}")

        left, right = st.columns(2)
        for idx, item in enumerate(analysis_result):
            with left if idx % 2 == 0 else right:
                _render_analysis_card(item)
    elif threats:
        st.info("Stage 2 is ready. Click the button above to generate root cause and remediation.")


if __name__ == "__main__":
    main()
