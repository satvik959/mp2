#!/usr/bin/env python3
"""
LSTM+GCN Streaming Detector with Agent Analysis (FINAL FLOW)
CSV → Build Features → LSTM+GCN Predict → Agent Analysis → Output
"""

import argparse
import sys
import pickle
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import os
from typing import Any, Dict, List, Optional

# Load environment variables from .env explicitly
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(env_path))

sys.path.insert(0, str(Path(__file__).parent))

from agents import run_agents
from dataset.dataset_builder import DatasetBuilder
from sklearn.feature_extraction.text import TfidfVectorizer


def _clean_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()
    return text if text else default


def _is_benign_label(label: Any) -> bool:
    return str(label).strip().lower() == "benign"


def _get_ip_column(batch_df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for column in candidates:
        if column in batch_df.columns:
            return column
    return None


def _detect_hub_ip(batch_df: pd.DataFrame, configured_hub: str = "") -> str:
    configured_hub = _clean_text(configured_hub, "")
    if configured_hub:
        return configured_hub

    for candidate in ("dst_ip", "Destination", "dst", "destination"):
        if candidate in batch_df.columns:
            values = batch_df[candidate].dropna().astype(str)
            if not values.empty:
                return str(values.value_counts().idxmax())

    for candidate in ("src_ip", "Source", "src", "source"):
        if candidate in batch_df.columns:
            values = batch_df[candidate].dropna().astype(str)
            if not values.empty:
                return str(values.value_counts().idxmax())

    return "UNKNOWN"


def _build_connectivity_rows(batch_df: pd.DataFrame, hub_ip: str) -> List[Dict[str, str]]:
    src_col = _get_ip_column(batch_df, ["src_ip", "Source", "src", "source"])
    dst_col = _get_ip_column(batch_df, ["dst_ip", "Destination", "dst", "destination"])
    if src_col is None or dst_col is None:
        return []

    subset = batch_df[[src_col, dst_col]].copy()
    subset[src_col] = subset[src_col].astype(str)
    subset[dst_col] = subset[dst_col].astype(str)

    if hub_ip != "UNKNOWN":
        subset = subset[(subset[src_col] == hub_ip) | (subset[dst_col] == hub_ip)]

    rows: List[Dict[str, str]] = []
    seen = set()
    for _, row in subset.dropna().iterrows():
        src_ip = _clean_text(row[src_col])
        dst_ip = _clean_text(row[dst_col])
        key = (src_ip, dst_ip)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"source": src_ip, "destination": dst_ip})
    return rows


def _print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    print(title)
    if not rows:
        print("   (no rows)")
        return

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    print(f"   {header_line}")
    print(f"   {separator}")
    for row in rows:
        print("   " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def _print_batch_connectivity(batch_idx: int, batch_df: pd.DataFrame, hub_ip: str) -> None:
    connectivity_rows = _build_connectivity_rows(batch_df, hub_ip)
    connected_systems = sorted({
        row["source"] if row["source"] != hub_ip else row["destination"]
        for row in connectivity_rows
        if row["source"] != hub_ip or row["destination"] != hub_ip
    })

    print(f"\n📡 BATCH {batch_idx} CONNECTIVITY SUMMARY")
    print(f"   Hub IP: {hub_ip}")
    print(f"   Connected systems ({len(connected_systems)}): {', '.join(connected_systems) if connected_systems else 'None'}")

    pair_rows = [[row["source"], row["destination"]] for row in connectivity_rows]
    _print_table("   Connection pairs:", ["source", "destination"], pair_rows)


def _print_batch_non_benign_report(batch_idx: int, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    non_benign_rows = [record for record in records if not record["is_benign"]]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in non_benign_rows:
        grouped[record["system_ip"]].append(record)

    print(f"\n🚨 BATCH {batch_idx} NON-BENIGN REPORT")
    if not grouped:
        print("   No non-benign packets in this batch.")
        return non_benign_rows

    for system_ip in sorted(grouped):
        system_rows = grouped[system_ip]
        table_rows = [
            [
                str(item["packet_num"]),
                item["predicted_label"],
                f"{item['confidence']:.1%}",
                item["src_ip"],
                item["dst_ip"],
            ]
            for item in system_rows
        ]
        print(f"   System: {system_ip} ({len(system_rows)} packet(s))")
        _print_table(
            "",
            ["packet", "class", "confidence", "src_ip", "dst_ip"],
            table_rows,
        )

    return non_benign_rows


def _print_final_summary(prediction_records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    total_packets = len(prediction_records)
    class_counts = Counter(record["predicted_label"] for record in prediction_records)
    benign_count = sum(1 for record in prediction_records if record["is_benign"])
    non_benign_count = total_packets - benign_count
    system_counts = Counter(record["system_ip"] for record in prediction_records if not record["is_benign"])

    print(f"Total packets     : {total_packets}")
    print(f"Total benign      : {benign_count}")
    print(f"Total non-benign  : {non_benign_count}")

    print("\nPer-class counts:")
    for label, count in class_counts.most_common():
        print(f"   {label:14s} {count}")

    print("\nTop 5 suspicious systems:")
    if not system_counts:
        print("   None")
        return

    rows = [[system_ip, str(count)] for system_ip, count in system_counts.most_common(5)]
    _print_table("", ["system_ip", "non_benign_count"], rows)

print("\n" + "="*80)
print("🔴 LSTM+GCN STREAMING DETECTOR → CREWAL ANALYSIS")
print("="*80)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--csv-path", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--label-encoder-path", required=True)
parser.add_argument("--preprocessor-path", required=True)
parser.add_argument("--batch-size", type=int, default=10)
parser.add_argument("--llm-model", default="groq/llama-3.1-8b-instant")
parser.add_argument("--hub-ip", default="", help="Optional fixed hub IP. If omitted, the hub is inferred per batch.")
args = parser.parse_args()

print(f"\n📋 CSV: {args.csv_path}")
print(f"   Model: {args.model_path}")
print(f"   Batch size: {args.batch_size}\n")
print(f"   LLM model: {args.llm_model}\n")
if args.hub_ip:
    print(f"   Hub IP: {args.hub_ip}\n")

# ============================================================================
# LOAD ARTIFACTS
# ============================================================================
print("─"*80)
print("STEP 1️⃣  LOADING MODEL & PREPROCESSORS")
print("─"*80)

model = tf.keras.models.load_model(args.model_path)
print(f"✅ Model loaded")

with open(args.label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
print(f"✅ Label encoder: {list(label_encoder.classes_)}")

with open(args.preprocessor_path, 'rb') as f:
    preprocessors = pickle.load(f)
print(f"✅ Preprocessors loaded\n")

# ============================================================================
# LOAD & BUILD FEATURES
# ============================================================================
print("─"*80)
print("STEP 2️⃣  LOADING & BUILDING FEATURES")
print("─"*80)

df = pd.read_csv(args.csv_path)
print(f"✅ Loaded {len(df)} packets")

timestamp_source = (
    pd.to_numeric(df['timestamp'], errors='coerce')
    if 'timestamp' in df.columns
    else pd.to_numeric(df['Time'], errors='coerce')
    if 'Time' in df.columns
    else pd.Series(np.arange(len(df), dtype=float))
)
src_source = (
    df['src_ip']
    if 'src_ip' in df.columns
    else df['Source']
    if 'Source' in df.columns
    else pd.Series(['0.0.0.0'] * len(df))
)
dst_source = (
    df['dst_ip']
    if 'dst_ip' in df.columns
    else df['Destination']
    if 'Destination' in df.columns
    else pd.Series(['0.0.0.0'] * len(df))
)
protocol_source = (
    df['protocol']
    if 'protocol' in df.columns
    else df['Protocol']
    if 'Protocol' in df.columns
    else pd.Series(['TCP'] * len(df))
)
packet_size_source = (
    df['packet_size']
    if 'packet_size' in df.columns
    else df['Length']
    if 'Length' in df.columns
    else pd.Series([0] * len(df))
)
src_port_source = df['src_port'] if 'src_port' in df.columns else pd.Series([None] * len(df))
dst_port_source = df['dst_port'] if 'dst_port' in df.columns else pd.Series([None] * len(df))
flags_source = df['flags'] if 'flags' in df.columns else pd.Series(['NONE'] * len(df))

# Build features using DatasetBuilder
builder = DatasetBuilder()
feature_df = builder.build_dataset(
    pd.DataFrame(
        {
            'timestamp': timestamp_source,
            'src_ip': src_source,
            'dst_ip': dst_source,
            'protocol': protocol_source,
            'packet_size': packet_size_source,
            'src_port': src_port_source,
            'dst_port': dst_port_source,
            'flags': flags_source,
        }
    ),
    window_size=1.0
)
feature_df['label'] = df['label'].values if 'label' in df.columns else 'benign'
feature_df['info'] = df['info'].values if 'info' in df.columns else df['Info'].values if 'Info' in df.columns else 'packet'
feature_df['flags'] = df['flags'].values if 'flags' in df.columns else 'NONE'
print(f"✅ Features built: {feature_df.shape}\n")

# ============================================================================
# FEATURE ENGINEERING (supports dict artifact or preprocessor object)
# ============================================================================
print("─"*80)
print("STEP 3️⃣  ENHANCED FEATURE ENGINEERING")
print("─"*80)

if isinstance(preprocessors, dict):
    struct_features = ['packet_size', 'packet_rate', 'connection_count', 'avg_packet_size', 'src_port', 'dst_port']

    # 1. Structured features
    X_struct = feature_df[struct_features].fillna(0).values
    X_struct_scaled = preprocessors['scaler_struct'].transform(X_struct)
    print(f"   ✅ Struct: {X_struct_scaled.shape}")

    # 2. Protocol
    X_protocol = preprocessors['protocol_encoder'].transform(feature_df['protocol']).reshape(-1, 1)
    X_protocol_scaled = preprocessors['scaler_protocol'].transform(X_protocol)
    print(f"   ✅ Protocol: {X_protocol_scaled.shape}")

    # 3. Flags
    flags_clean = feature_df['flags'].fillna('PA').astype(str)
    known_flags = set(preprocessors['flags_encoder'].classes_)
    flags_clean = flags_clean.apply(lambda x: x if x in known_flags else 'PA')
    X_flags = preprocessors['flags_encoder'].transform(flags_clean).reshape(-1, 1)
    X_flags_scaled = preprocessors['scaler_flags'].transform(X_flags)
    print(f"   ✅ Flags: {X_flags_scaled.shape}")

    # 4. TF-IDF
    info_text = feature_df['info'].fillna('').astype(str).values
    X_tfidf = preprocessors['vectorizer'].transform(info_text).toarray()
    print(f"   ✅ TF-IDF: {X_tfidf.shape}")

    # 5. Statistical
    time_delta = np.diff(feature_df['timestamp'].values, prepend=feature_df['timestamp'].values[0])
    X_stats = np.column_stack([
        np.log1p(feature_df['packet_size'].values),
        time_delta,
        (feature_df['src_port'].values % 256) / 256,
        (feature_df['dst_port'].values % 256) / 256,
    ])
    X_stats_scaled = preprocessors['scaler_stats'].transform(X_stats)
    print(f"   ✅ Statistical: {X_stats_scaled.shape}")

    X_combined = np.hstack([X_struct_scaled, X_protocol_scaled, X_flags_scaled, X_tfidf, X_stats_scaled])
else:
    # Current artifact format: serialized NetworkDataPreprocessor object.
    X_combined, _ = preprocessors.preprocess_data(feature_df, fit=False)
    print(f"   ✅ Preprocessor object output: {X_combined.shape}")

print(f"\n✅ TOTAL FEATURES: {X_combined.shape[1]}\n")

# ============================================================================
# PREPARE MODEL INPUTS
# ============================================================================
print("─"*80)
print("STEP 4️⃣  PREPARING SEQUENCES")
print("─"*80)

sequence_input_shape, graph_input_shape = model.input_shape
seq_length = int(sequence_input_shape[1] or 1)

def create_sequences(X, seq_length):
    seqs = []
    for i in range(len(X) - seq_length + 1):
        seqs.append(X[i:i+seq_length])
    return np.array(seqs)

if seq_length <= 1:
    X_seq = X_combined.reshape(len(X_combined), 1, X_combined.shape[1]).astype(np.float32)
else:
    X_seq = create_sequences(X_combined, seq_length).astype(np.float32)

# Lightweight graph features aligned with each sequence endpoint.
src_hash = feature_df['src_ip'].astype(str).apply(lambda x: hash(x) % 1000).to_numpy(dtype=np.float32)
dst_hash = feature_df['dst_ip'].astype(str).apply(lambda x: hash(x) % 1000).to_numpy(dtype=np.float32)
edge_activity = feature_df['connection_count'].to_numpy(dtype=np.float32)
packet_rate = feature_df['packet_rate'].to_numpy(dtype=np.float32)
graph_all = np.stack([src_hash, dst_hash, edge_activity, packet_rate], axis=1)
graph_all /= np.array([1000.0, 1000.0, 50.0, 50.0], dtype=np.float32)
G_seq = graph_all[seq_length - 1:] if seq_length > 1 else graph_all

if len(X_seq) != len(G_seq):
    min_len = min(len(X_seq), len(G_seq))
    X_seq = X_seq[:min_len]
    G_seq = G_seq[:min_len]

print(f"✅ Sequence shape: {X_seq.shape} (model expects {sequence_input_shape})")
print(f"✅ Graph shape   : {G_seq.shape} (model expects {graph_input_shape})\n")

# ============================================================================
# STREAMING PREDICTIONS
# ============================================================================
print("─"*80)
print("STEP 5️⃣  STREAMING PREDICTIONS")
print("─"*80)

all_predictions = []
anomalies = []
prediction_records = []

num_batches = (len(X_seq) + args.batch_size - 1) // args.batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(X_seq))
    X_batch = X_seq[start_idx:end_idx]
    G_batch = G_seq[start_idx:end_idx]
    batch_df = df.iloc[start_idx:end_idx].copy()
    
    # Predict
    probs = model.predict([X_batch, G_batch], verbose=0)
    preds = np.argmax(probs, axis=1)
    all_predictions.extend(preds)
    batch_records = []
    
    # Show batch
    pred_classes = label_encoder.inverse_transform(preds)
    print(f"\n📊 Batch {batch_idx+1}/{num_batches} ({len(X_batch)} sequences)")
    for i, (pred_class, prob) in enumerate(zip(pred_classes, probs)):
        pred_label = str(pred_class)
        conf = np.max(prob)
        seq_idx = start_idx + i
        pkt_idx = seq_idx + seq_length
        is_benign = _is_benign_label(pred_label)
        row_idx = min(pkt_idx - 1, len(df) - 1)
        src_ip = _clean_text(
            df.iloc[row_idx]['src_ip'] if 'src_ip' in df.columns else df.iloc[row_idx]['Source'] if 'Source' in df.columns else 'N/A'
        )
        dst_ip = _clean_text(
            df.iloc[row_idx]['dst_ip'] if 'dst_ip' in df.columns else df.iloc[row_idx]['Destination'] if 'Destination' in df.columns else 'N/A'
        )
        system_ip = src_ip if src_ip != 'N/A' else dst_ip
        emoji = "🟢" if is_benign else "🔴"
        print(f"   {emoji} Seq {seq_idx:3d} (Pkt {pkt_idx:3d}): {pred_label:12s} ({conf:5.1%})")

        record = {
            'batch_number': batch_idx + 1,
            'sequence_index': seq_idx,
            'packet_num': pkt_idx,
            'predicted_label': pred_label,
            'confidence': float(conf),
            'is_benign': is_benign,
            'system_ip': system_ip,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
        }
        batch_records.append(record)
        prediction_records.append(record)
        
        if not is_benign:
            anomalies.append({
                'packet_num': pkt_idx,
                'prediction': pred_label,
                'confidence': conf,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'info': df.iloc[min(pkt_idx-1, len(df)-1)]['info'] if 'info' in df.columns else '',
                'flags_pattern': [str(df.iloc[min(pkt_idx-1, len(df)-1)]['flags'])] if 'flags' in df.columns else [],
                'packet_rate': float(len(X_seq) / max(1, (feature_df['timestamp'].max() - feature_df['timestamp'].min()))),
                'protocol': str(df.iloc[min(pkt_idx-1, len(df)-1)]['protocol']) if 'protocol' in df.columns else 'TCP',
                'connection_count': len(df),
                'batch_summary': f"Anomaly in batch {batch_idx+1}",
                'avg_packet_size': float(feature_df['avg_packet_size'].mean()) if 'avg_packet_size' in feature_df.columns else 0.0,
            })

    hub_ip = _detect_hub_ip(batch_df, args.hub_ip)
    _print_batch_connectivity(batch_idx + 1, batch_df, hub_ip)
    _print_batch_non_benign_report(batch_idx + 1, batch_records)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "─"*80)
print("STEP 6️⃣  DETECTION SUMMARY")
print("─"*80 + "\n")

if len(all_predictions) == 0:
    pred_classes = np.array([])
else:
    pred_classes = label_encoder.inverse_transform(np.array(all_predictions))

print("📊 Distribution:")
if len(pred_classes) == 0:
    print("   No predictions available.")
    benign = 0
    malicious = 0
else:
    unique, counts = np.unique(pred_classes, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(pred_classes)
        print(f"   {str(cls):12s}: {count:4d} ({pct:5.1f}%)")

    pred_classes_str = np.array([str(x) for x in pred_classes])
    benign = np.sum(pred_classes_str == 'benign')
    malicious = len(pred_classes) - benign

print(f"\n📈 Total: {len(pred_classes)} | Benign: {benign} | Malicious: {malicious}\n")

# ============================================================================
# AGENT ANALYSIS
# ============================================================================
if len(anomalies) > 0:
    print("─"*80)
    print(f"STEP 7️⃣  CREWAL AGENT ANALYSIS ({len(anomalies)} ANOMALIES)")
    print("─"*80 + "\n")
    
    for idx, anom in enumerate(anomalies[:3], 1):
        print(f"🔍 Anomaly {idx}/{min(3, len(anomalies))}: {anom['prediction']} @ Packet {anom['packet_num']}")
        print(f"   {anom['src_ip']} → {anom['dst_ip']}\n")
        
        try:
            result = run_agents(anom, llm_model=args.llm_model)
            analyzer = result.get('analyzer_output', {})
            remediation = result.get('remediation_output', {})
            
            print(f"✅ Analysis: {analyzer.get('anomaly_type', 'unknown')}")
            print(f"   Cause: {analyzer.get('cause', 'unknown')}")
            print(f"   Confidence: {analyzer.get('confidence', 'low')}")
            
            print(f"✅ Remediation (Priority: {remediation.get('priority', 'medium').upper()}):")
            for action in remediation.get('recommended_actions', []):
                print(f"   • {action}")
            print()
        except Exception as e:
            print(f"   ⚠️  {str(e)[:80]}\n")
        
        import time
        if idx < 3:
            time.sleep(2)  # Rate limiting
    
    if len(anomalies) > 3:
        print(f"   ... and {len(anomalies)-3} more anomalies detected\n")
else:
    print("\n✅ NO ANOMALIES DETECTED - NETWORK SECURE\n")

_print_final_summary(prediction_records)

print("="*80)
print("✅ DETECTION & ANALYSIS COMPLETE")
print("="*80 + "\n")
