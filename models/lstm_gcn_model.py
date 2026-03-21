"""
Hybrid LSTM + GCN-style model training pipeline.

This module trains a two-branch model:
- LSTM branch for sequential traffic representation
- Graph branch (dense projection of graph-inspired features)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_loader import DatasetLoader
from dataset.dataset_builder import DatasetBuilder
from preprocessing.preprocessing import NetworkDataPreprocessor


CLASS_LABELS = ["benign", "probe", "brute_force", "flood", "exploit", "malware"]


def rule_based_multiclass_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-class labels so the training pipeline can run end-to-end."""
    labeled = df.copy()

    def assign_label(row: pd.Series) -> str:
        info = str(row.get("Info", "")).lower()
        proto = str(row.get("Protocol", "")).lower()
        length = float(row.get("Length", 0) or 0)

        if any(k in info for k in ["syn", "flood", "igmp", "mdns", "ssdp"]) and length > 140:
            return "flood"
        if any(k in info for k in ["password", "login", "ftp", "ssh", "rdp"]):
            return "brute_force"
        if any(k in info for k in ["scan", "query", "m-search", "who has"]):
            return "probe"
        if any(k in info for k in ["malware", "payload", "encrypted packet"]):
            return "malware"
        if proto in {"quic", "tlsv1.3", "tlsv1.2"} and length > 1200:
            return "exploit"
        return "benign"

    labeled["label"] = labeled.apply(assign_label, axis=1)
    return labeled


def build_graph_features(feature_df: pd.DataFrame) -> np.ndarray:
    """Build simple graph-inspired node/edge statistics as model input."""
    src_hash = feature_df["src_ip"].astype(str).apply(lambda x: hash(x) % 1000).to_numpy(dtype=np.float32)
    dst_hash = feature_df["dst_ip"].astype(str).apply(lambda x: hash(x) % 1000).to_numpy(dtype=np.float32)
    edge_activity = feature_df["connection_count"].to_numpy(dtype=np.float32)
    packet_rate = feature_df["packet_rate"].to_numpy(dtype=np.float32)

    graph_features = np.stack([src_hash, dst_hash, edge_activity, packet_rate], axis=1)
    graph_features /= np.array([1000.0, 1000.0, 50.0, 50.0], dtype=np.float32)
    return graph_features


def build_hybrid_model(seq_feature_dim: int, graph_feature_dim: int, num_classes: int):
    """Create the hybrid LSTM + GCN-style model."""
    try:
        tf = __import__("tensorflow")
        keras_layers = __import__("tensorflow.keras.layers", fromlist=["Input", "LSTM", "Dense", "Dropout", "Concatenate"])
        keras_model_module = __import__("tensorflow.keras", fromlist=["Model"])

        seq_input = keras_layers.Input(shape=(1, seq_feature_dim), name="sequence_input")
        x_seq = keras_layers.LSTM(64, return_sequences=False, name="lstm_branch")(seq_input)
        x_seq = keras_layers.Dropout(0.2)(x_seq)

        graph_input = keras_layers.Input(shape=(graph_feature_dim,), name="graph_input")
        x_graph = keras_layers.Dense(64, activation="relu", name="graph_dense_1")(graph_input)
        x_graph = keras_layers.Dense(32, activation="relu", name="graph_dense_2")(x_graph)

        merged = keras_layers.Concatenate(name="fusion")([x_seq, x_graph])
        merged = keras_layers.Dense(64, activation="relu")(merged)
        merged = keras_layers.Dropout(0.2)(merged)
        output = keras_layers.Dense(num_classes, activation="softmax", name="class_output")(merged)

        model = keras_model_module.Model(inputs=[seq_input, graph_input], outputs=output)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model, "tensorflow"
    except Exception:
        fallback_model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=300,
            random_state=42,
        )
        return fallback_model, "sklearn"


def prepare_training_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, NetworkDataPreprocessor]:
    """Load raw CSV, build features, and return tensors for hybrid training."""
    loader = DatasetLoader(str(csv_path))
    raw_df = loader.load_dataset()

    labeled_raw = rule_based_multiclass_label(raw_df)

    packet_df = pd.DataFrame(
        {
            "timestamp": labeled_raw["Time"],
            "src_ip": labeled_raw["Source"],
            "dst_ip": labeled_raw["Destination"],
            "protocol": labeled_raw["Protocol"],
            "packet_size": labeled_raw["Length"],
            "src_port": None,
            "dst_port": None,
        }
    )

    builder = DatasetBuilder()
    feature_df = builder.build_dataset(packet_df)
    feature_df["Info"] = labeled_raw["Info"].values
    feature_df["label"] = labeled_raw["label"].values

    preprocessor = NetworkDataPreprocessor()
    X_structured, y = preprocessor.preprocess_data(feature_df, fit=True)
    X_sequence = X_structured.reshape(X_structured.shape[0], 1, X_structured.shape[1])
    X_graph = build_graph_features(feature_df)

    return X_sequence, X_graph, y, preprocessor


def train_and_save(csv_path: Path, output_dir: Path, epochs: int = 8, batch_size: int = 32) -> None:
    """Train model and save artifacts for inference/streaming."""
    X_seq, X_graph, y, preprocessor = prepare_training_data(csv_path)

    X_seq_train, X_seq_test, X_graph_train, X_graph_test, y_train, y_test = train_test_split(
        X_seq, X_graph, y, test_size=0.2, random_state=42, stratify=y
    )

    model, backend = build_hybrid_model(
        seq_feature_dim=X_seq.shape[2],
        graph_feature_dim=X_graph.shape[1],
        num_classes=len(np.unique(y)),
    )

    if backend == "tensorflow":
        model.fit(
            [X_seq_train, X_graph_train],
            y_train,
            validation_data=([X_seq_test, X_graph_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        loss, acc = model.evaluate([X_seq_test, X_graph_test], y_test, verbose=0)
        print(f"Backend: tensorflow | Test Accuracy: {acc:.4f} | Loss: {loss:.4f}")
    else:
        X_train_flat = np.hstack([X_seq_train.reshape(X_seq_train.shape[0], -1), X_graph_train])
        X_test_flat = np.hstack([X_seq_test.reshape(X_seq_test.shape[0], -1), X_graph_test])
        model.fit(X_train_flat, y_train)
        preds = model.predict(X_test_flat)
        acc = accuracy_score(y_test, preds)
        print(f"Backend: sklearn-fallback | Test Accuracy: {acc:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "tensorflow":
        model_path = output_dir / "lstm_gcn_model.keras"
        model.save(model_path)
    else:
        model_path = output_dir / "lstm_gcn_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)

    artifact_paths = preprocessor.save_artifacts(str(output_dir))

    print("Saved model and preprocessing artifacts:")
    print(f"- model: {model_path}")
    for name, path in artifact_paths.items():
        print(f"- {name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid LSTM + GCN-style model")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/network_traffic_dataset.csv"),
        help="Path to network traffic CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saved model and preprocessors",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(42)

    train_and_save(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
