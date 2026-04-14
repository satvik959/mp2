from __future__ import annotations

import html
import pickle
import tempfile
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
            .stApp { background-color: #0b1220; color: #e5e7eb; }
            [data-testid="stAppViewContainer"] { background-color: #0b1220; }
            [data-testid="stHeader"] { background: #0b1220; }
            [data-testid="stSidebar"] { background-color: #111827; }
            [data-testid="stSidebar"] * { color: #e5e7eb; }
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stMarkdownContainer"] span,
            [data-testid="stText"] {
                color: #e5e7eb;
            }
            .dark-card {
                background: #111827;
                color: #f9fafb;
                border-radius: 12px;
                padding: 14px 16px;
                margin: 8px 0;
                border: 1px solid #1f2937;
            }
            .metric-chip {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.8rem;
                margin-right: 6px;
                font-weight: 600;
            }
            .chip-green { background: #14532d; color: #dcfce7; }
            .chip-red { background: #7f1d1d; color: #fee2e2; }
            .chip-gray { background: #374151; color: #e5e7eb; }
            .threat-highlight {
                border: 1px solid #ef4444;
                box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.35) inset;
            }
            .log-box {
                background: #0b1220;
                color: #d1d5db;
                padding: 10px;
                border-radius: 10px;
                height: 220px;
                overflow-y: auto;
                font-family: Consolas, monospace;
                font-size: 0.85rem;
                border: 1px solid #1f2937;
            }
            .section-title {
                color: #f9fafb;
                font-weight: 700;
                margin-top: 6px;
            }
            div[data-testid="stMetric"] {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 10px;
                padding: 10px 12px;
            }
            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] div {
                color: #f3f4f6;
            }
            .stDataFrame, .stTable {
                background: #111827;
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

    st.markdown(
        f"""
        <div class="dark-card">
            <div>
                <span class="metric-chip chip-red">🚨 {html.escape(str(item.get('prediction', 'unknown')).upper())}</span>
                <span class="metric-chip {priority_class}">Priority: {html.escape(priority.upper())}</span>
            </div>
            <p><b>From:</b> {html.escape(str(item.get('src_ip', 'N/A')))} &nbsp;&nbsp; <b>To:</b> {html.escape(str(item.get('dst_ip', 'N/A')))}</p>
            <p><b>Cause:</b> {html.escape(str(analyzer.get('cause', 'Unknown')))}</p>
            <p><b>Confidence:</b> {html.escape(str(analyzer.get('confidence', 'low')).upper())}</p>
            <p><b>Evidence:</b></p>
            <ul>{evidence_html}</ul>
            <p><b>Recommended Actions:</b></p>
            <ul>{actions_html}</ul>
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

    num_batches = (len(x_seq) + batch_size - 1) // batch_size
    progress = st.progress(0, text="Running prediction batches...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(x_seq))

        probs = model.predict([x_seq[start_idx:end_idx], g_seq[start_idx:end_idx]], verbose=0)
        pred_idx = np.argmax(probs, axis=1)
        pred_labels = label_encoder.inverse_transform(pred_idx)

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
        "distribution": distribution,
        "non_benign_table": pd.DataFrame(non_benign_rows),
        "agent_table": pd.DataFrame(agent_outputs),
    }


def main() -> None:
    st.set_page_config(page_title="Network IDS Dashboard", page_icon="N", layout="wide")
    _inject_theme()
    st.title("Network IDS Dashboard")
    st.caption("✅ Detection and 🚨 Analysis separated for clear academic presentation.")

    if "detection_result" not in st.session_state:
        st.session_state["detection_result"] = None
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "detected_threats" not in st.session_state:
        st.session_state["detected_threats"] = []

    tab_detection, tab_analysis = st.tabs(["Detection", "Analysis & Remediation"])

    with tab_detection:
        st.markdown('<div class="section-title">📊 Detection Controls</div>', unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Upload CSV packet data", type=["csv"], key="detection_csv_uploader")

        c1, c2 = st.columns(2)
        with c1:
            csv_path_text = st.text_input("Or CSV path", value="data/small_dataset.csv")
            model_path_text = st.text_input("Model path", value="artifacts/lstm_gcn_model.keras")
        with c2:
            label_encoder_path_text = st.text_input("Label encoder path", value="artifacts/label_encoder.pkl")
            preprocessor_path_text = st.text_input("Preprocessor path", value="artifacts/preprocessor.pkl")

        batch_size = st.number_input("Batch size", min_value=1, max_value=5000, value=10, step=1)
        run_btn = st.button("Run Detection", type="primary", use_container_width=True, key="run_detection_btn")

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
                with st.spinner("Running model inference..."):
                    try:
                        result = run_detection(
                            csv_path=csv_path,
                            model_path=model_path,
                            label_encoder_path=label_encoder_path,
                            preprocessor_path=preprocessor_path,
                            batch_size=int(batch_size),
                            run_agent_analysis=False,
                            llm_model="gemini/gemini-1.5-flash",
                            max_agent_items=3,
                        )
                        st.session_state["detection_result"] = result
                        st.session_state["detected_threats"] = result.get("non_benign_table", pd.DataFrame()).to_dict(orient="records")
                        st.session_state["analysis_result"] = None
                        st.success("✅ Detection completed. Open the Analysis tab to run remediation.")
                    except Exception as exc:
                        st.exception(exc)

        result = st.session_state.get("detection_result")
        if not result:
            st.info("Upload a CSV and click Run Detection.")
        else:
            benign_count = int(result.get("benign_count", 0))
            non_benign_count = int(result.get("non_benign_count", 0))
            total_seq = int(result.get("total_sequences", 0))
            avg_conf = float(result.get("avg_confidence", 0.0))

            benign_pct = (100.0 * benign_count / total_seq) if total_seq else 0.0
            malicious_pct = (100.0 * non_benign_count / total_seq) if total_seq else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total packets", result.get("total_packets", total_seq))
            m2.metric("✅ Benign", f"{benign_count} ({benign_pct:.1f}%)")
            m3.metric("🚨 Malicious", f"{non_benign_count} ({malicious_pct:.1f}%)")
            m4.metric("Average confidence", f"{avg_conf:.2%}")

            st.markdown('<div class="section-title">📊 Attack Distribution</div>', unsafe_allow_html=True)
            st.dataframe(result["distribution"], use_container_width=True)
            if not result["distribution"].empty:
                st.bar_chart(result["distribution"].set_index("class"))

            st.markdown('<div class="section-title">⚠️ Suspicious Traffic Table</div>', unsafe_allow_html=True)
            non_benign_table = result["non_benign_table"].copy()
            if non_benign_table.empty:
                st.markdown('<div class="dark-card"><span class="metric-chip chip-green">✅ No suspicious traffic detected</span></div>', unsafe_allow_html=True)
            else:
                non_benign_table["flow"] = non_benign_table["src_ip"].astype(str) + " → " + non_benign_table["dst_ip"].astype(str)
                display_df = non_benign_table[["flow", "predicted_label", "confidence", "packet_rate"]].rename(
                    columns={
                        "flow": "Source → Destination",
                        "predicted_label": "Predicted Attack Type",
                        "confidence": "Confidence",
                        "packet_rate": "Packet Rate",
                    }
                )
                st.dataframe(display_df, use_container_width=True)

            st.markdown('<div class="section-title">🧾 Live Detection Feed (Simulated)</div>', unsafe_allow_html=True)
            feed_rows = result.get("prediction_rows", [])[-30:]
            feed_lines = []
            for row in feed_rows:
                icon = "✅" if row.get("is_benign") else "⚠️"
                label = "NORMAL" if row.get("is_benign") else str(row.get("predicted_label", "ATTACK")).upper()
                feed_lines.append(
                    f"[{row.get('timestamp', datetime.now().strftime('%H:%M:%S'))}] "
                    f"{row.get('src_ip', 'N/A')} → {row.get('dst_ip', 'N/A')} "
                    f"{icon} {label} ({float(row.get('confidence', 0.0)):.1%})"
                )
            st.markdown(
                f'<div class="log-box">{"<br>".join(html.escape(line) for line in feed_lines) if feed_lines else "No events yet."}</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">🚨 Top Threat</div>', unsafe_allow_html=True)
            if non_benign_table.empty:
                st.markdown('<div class="dark-card"><span class="metric-chip chip-green">✅ No active threat</span></div>', unsafe_allow_html=True)
            else:
                top = non_benign_table.sort_values("confidence", ascending=False).iloc[0]
                st.markdown(
                    f"""
                    <div class="dark-card threat-highlight">
                        <span class="metric-chip chip-red">Top Threat</span>
                        <p><b>Flow:</b> {html.escape(str(top['src_ip']))} → {html.escape(str(top['dst_ip']))}</p>
                        <p><b>Attack:</b> {html.escape(str(top['predicted_label']))}</p>
                        <p><b>Confidence:</b> {float(top['confidence']):.2%}</p>
                        <p><b>Packet Rate:</b> {float(top.get('packet_rate', 0.0)):.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-title">📘 Explainability</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="dark-card">
                    <p><b>LSTM:</b> captures temporal behavior, such as bursty packet patterns over time.</p>
                    <p><b>GCN-style branch:</b> captures network relationships between communicating IP nodes.</p>
                    <p>Together, they detect attacks using both time dynamics and communication structure.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_analysis:
        st.markdown('<div class="section-title">⚙️ Analysis & Remediation</div>', unsafe_allow_html=True)
        llm_model = st.text_input("LLM model", value="gemini/gemini-1.5-flash", key="analysis_llm_model")
        max_agent_items = st.number_input("Max anomaly analyses", min_value=1, max_value=50, value=5, step=1, key="analysis_max_items")
        run_analysis_btn = st.button(
            "Run Analysis on Detected Threats",
            type="primary",
            use_container_width=True,
            key="run_analysis_btn",
        )

        threats = st.session_state.get("detected_threats", [])
        if not threats:
            st.warning("⚠️ No detected threats yet. Run Detection tab first.")
        elif run_analysis_btn:
            if not AGENTS_AVAILABLE:
                st.error("agents.py dependencies are unavailable. Install crewai and provider keys first.")
            else:
                analysis_results: List[Dict[str, Any]] = []
                with st.spinner("Running agent analysis..."):
                    for row in threats[: int(max_agent_items)]:
                        payload = {
                            "prediction": row.get("predicted_label", "unknown"),
                            "packet_rate": int(row.get("packet_rate", 0)),
                            "protocol": row.get("protocol", "UNKNOWN"),
                            "flags_pattern": [row.get("flags", "NONE")],
                            "connection_count": 0,
                            "batch_summary": f"Anomaly around packet {row.get('packet_num', 'N/A')}",
                            "avg_packet_size": 0.0,
                        }
                        agent_result = run_agents(payload, llm_model=llm_model)
                        analysis_results.append(
                            {
                                "prediction": row.get("predicted_label", "unknown"),
                                "src_ip": row.get("src_ip", "N/A"),
                                "dst_ip": row.get("dst_ip", "N/A"),
                                "analyzer_output": agent_result.get("analyzer_output", {}),
                                "remediation_output": agent_result.get("remediation_output", {}),
                            }
                        )
                st.session_state["analysis_result"] = analysis_results
                st.success("✅ Analysis complete.")

        analysis_result = st.session_state.get("analysis_result")
        if analysis_result:
            for item in analysis_result:
                _render_analysis_card(item)
        elif threats:
            st.info("Detected threats are ready. Click the analysis button to generate cause and remediation cards.")


if __name__ == "__main__":
    main()
