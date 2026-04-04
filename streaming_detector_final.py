#!/usr/bin/env python3
"""
LSTM+GCN Streaming Detector with Agent Analysis (FINAL FLOW)
CSV → Build Features → LSTM+GCN Predict → Agent Analysis → Output
"""

import argparse
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import os

# Load environment variables from .env explicitly
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(env_path))

sys.path.insert(0, str(Path(__file__).parent))

from agents import run_agents
from dataset.dataset_builder import DatasetBuilder
from sklearn.feature_extraction.text import TfidfVectorizer

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
args = parser.parse_args()

print(f"\n📋 CSV: {args.csv_path}")
print(f"   Model: {args.model_path}")
print(f"   Batch size: {args.batch_size}\n")

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

# Build features using DatasetBuilder
builder = DatasetBuilder()
feature_df = builder.build_dataset(
    df[['timestamp', 'src_ip', 'dst_ip', 'protocol', 'packet_size', 'src_port', 'dst_port', 'flags']],
    window_size=1.0
)
feature_df['label'] = df['label'].values if 'label' in df.columns else 'benign'
feature_df['info'] = df['info'].values if 'info' in df.columns else 'packet'
feature_df['flags'] = df['flags'].values
print(f"✅ Features built: {feature_df.shape}\n")

# ============================================================================
# ENHANCED FEATURE ENGINEERING (FULL 62 FEATURES)
# ============================================================================
print("─"*80)
print("STEP 3️⃣  ENHANCED FEATURE ENGINEERING")
print("─"*80)

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
# Handle unknown flags and NaN by replacing with most common flag 'PA'
flags_clean = feature_df['flags'].fillna('PA').astype(str)
# Replace any flags not seen during training with 'PA'
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

# Combine
X_combined = np.hstack([X_struct_scaled, X_protocol_scaled, X_flags_scaled, X_tfidf, X_stats_scaled])
print(f"\n✅ TOTAL FEATURES: {X_combined.shape[1]} (62 expected)\n")

# ============================================================================
# PREPARE SEQUENCES
# ============================================================================
print("─"*80)
print("STEP 4️⃣  PREPARING SEQUENCES")
print("─"*80)

seq_length = 10

def create_sequences(X, seq_length):
    seqs = []
    for i in range(len(X) - seq_length + 1):
        seqs.append(X[i:i+seq_length])
    return np.array(seqs)

X_seq = create_sequences(X_combined, seq_length)
print(f"✅ Created {len(X_seq)} sequences of length {seq_length}\n")

# ============================================================================
# STREAMING PREDICTIONS
# ============================================================================
print("─"*80)
print("STEP 5️⃣  STREAMING PREDICTIONS")
print("─"*80)

all_predictions = []
anomalies = []

num_batches = (len(X_seq) + args.batch_size - 1) // args.batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(X_seq))
    X_batch = X_seq[start_idx:end_idx]
    
    # Predict
    probs = model.predict([X_batch, X_batch], verbose=0)
    preds = np.argmax(probs, axis=1)
    all_predictions.extend(preds)
    
    # Show batch
    pred_classes = label_encoder.inverse_transform(preds)
    print(f"\n📊 Batch {batch_idx+1}/{num_batches} ({len(X_batch)} sequences)")
    for i, (pred_class, prob) in enumerate(zip(pred_classes, probs)):
        conf = np.max(prob)
        seq_idx = start_idx + i
        pkt_idx = seq_idx + seq_length
        emoji = "🔴" if pred_class != "benign" else "🟢"
        print(f"   {emoji} Seq {seq_idx:3d} (Pkt {pkt_idx:3d}): {pred_class:12s} ({conf:5.1%})")
        
        if pred_class != "benign":
            src_ip = df.iloc[min(pkt_idx-1, len(df)-1)]['src_ip'] if 'src_ip' in df.columns else 'N/A'
            dst_ip = df.iloc[min(pkt_idx-1, len(df)-1)]['dst_ip'] if 'dst_ip' in df.columns else 'N/A'
            anomalies.append({
                'packet_num': pkt_idx,
                'prediction': pred_class,
                'confidence': conf,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'info': df.iloc[min(pkt_idx-1, len(df)-1)]['info'] if 'info' in df.columns else '',
                'flags_pattern': [str(df.iloc[min(pkt_idx-1, len(df)-1)]['flags'])] if 'flags' in df.columns else [],
                'packet_rate': float(len(X_seq) / max(1, (feature_df['timestamp'].max() - feature_df['timestamp'].min()))),
                'protocol': str(df.iloc[min(pkt_idx-1, len(df)-1)]['protocol']) if 'protocol' in df.columns else 'TCP',
                'connection_count': len(df),
                'batch_summary': f"Anomaly in batch {batch_idx+1}"
            })

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "─"*80)
print("STEP 6️⃣  DETECTION SUMMARY")
print("─"*80 + "\n")

pred_classes = label_encoder.inverse_transform(np.array(all_predictions))
print("📊 Distribution:")
unique, counts = np.unique(pred_classes, return_counts=True)
for cls, count in zip(unique, counts):
    pct = 100 * count / len(pred_classes)
    print(f"   {cls:12s}: {count:4d} ({pct:5.1f}%)")

benign = np.sum(pred_classes == 'benign')
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
            result = run_agents(anom)
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

print("="*80)
print("✅ DETECTION & ANALYSIS COMPLETE")
print("="*80 + "\n")
