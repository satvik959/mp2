"""
Real-time CSV streaming detector.

This script monitors a growing CSV file, ingests only newly appended rows,
buffers them, and performs batch preprocessing + model inference.
"""

from __future__ import annotations

import argparse
import csv
import io
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dataset.dataset_builder import DatasetBuilder
from preprocessing.preprocessing import NetworkDataPreprocessor


class CSVRowStreamReader:
    """Reads only newly appended rows from a CSV file using a file offset."""

    def __init__(self, csv_path: Path, start_from_end: bool = True):
        self.csv_path = csv_path
        self.start_from_end = start_from_end
        self.columns: Optional[List[str]] = None
        self.file_offset: int = 0
        self.last_read_index: int = 0
        self._initialized = False

    def initialize(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            header_line = handle.readline()
            if not header_line:
                raise RuntimeError("CSV file is empty and has no header.")

            self.columns = next(csv.reader([header_line]))

            if self.start_from_end:
                remaining = handle.read()
                self.last_read_index = remaining.count("\n")
                self.file_offset = handle.tell()
            else:
                self.file_offset = handle.tell()
                self.last_read_index = 0

        self._initialized = True

    def read_new_rows(self) -> pd.DataFrame:
        if not self._initialized:
            self.initialize()

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            handle.seek(self.file_offset)
            new_text = handle.read()
            new_offset = handle.tell()

        if not new_text or not new_text.strip():
            return pd.DataFrame(columns=self.columns)

        rows_df = pd.read_csv(
            io.StringIO(new_text),
            header=None,
            names=self.columns,
            on_bad_lines="skip",
        )
        rows_df = rows_df.dropna(how="all")

        self.file_offset = new_offset
        self.last_read_index += len(rows_df)
        return rows_df


class ModelRuntime:
    """Loads model and related artifacts once, and performs batch inference."""

    def __init__(
        self,
        model_path: Optional[Path],
        label_encoder_path: Optional[Path],
        vectorizer_path: Optional[Path],
    ):
        self.model_path = model_path
        self.label_encoder = self._load_optional_pickle(label_encoder_path)
        self.vectorizer = self._load_optional_pickle(vectorizer_path)
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_optional_pickle(path: Optional[Path]):
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        with path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _load_model(path: Optional[Path]):
        if path is None:
            return None

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".pkl", ".joblib"}:
            with path.open("rb") as handle:
                return pickle.load(handle)

        if suffix in {".pth", ".pt"}:
            try:
                torch = __import__("torch")
            except ImportError as exc:
                raise RuntimeError(
                    "Torch is required for .pth/.pt models. Install torch or use a pickle model."
                ) from exc

            model = torch.load(path, map_location="cpu")
            if hasattr(model, "eval"):
                model.eval()
            return model

        if suffix in {".keras", ".h5"}:
            try:
                keras_models = __import__("tensorflow.keras.models", fromlist=["load_model"])
                return keras_models.load_model(path)
            except ImportError as exc:
                raise RuntimeError(
                    "TensorFlow is required for .keras/.h5 models. Install tensorflow to use this model."
                ) from exc

        raise RuntimeError(f"Unsupported model format: {path}")

    def predict(self, features: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        if self.model is None:
            if isinstance(features, tuple):
                return ["normal"] * len(features[0])
            return ["normal"] * len(features)

        if hasattr(self.model, "predict"):
            predict_input = features
            if isinstance(features, tuple):
                predict_input = [features[0], features[1]]
            raw = self.model.predict(predict_input)
            return self._decode(raw)

        try:
            torch = __import__("torch")
        except ImportError as exc:
            raise RuntimeError("Torch is required for this model type.") from exc

        if callable(self.model):
            with torch.no_grad():
                input_tensor = torch.tensor(features, dtype=torch.float32)
                output = self.model(input_tensor)
                if hasattr(output, "detach"):
                    output = output.detach().cpu().numpy()
                raw = np.argmax(output, axis=1)
                return self._decode(raw)

        raise RuntimeError("Loaded model does not support predict() or callable inference.")

    def _decode(self, raw_predictions) -> List[str]:
        raw_array = np.array(raw_predictions)

        if raw_array.ndim > 1:
            raw_array = np.argmax(raw_array, axis=1)

        if self.label_encoder is not None:
            try:
                decoded = self.label_encoder.inverse_transform(raw_array.astype(int))
                return [str(label) for label in decoded]
            except Exception:
                pass

        return [str(item) for item in raw_array.tolist()]


class StreamingPipeline:
    """Three-stage pipeline: ingestion, preprocessing, and inference."""

    def __init__(self, runtime: ModelRuntime, preprocessor_path: Optional[Path] = None):
        self.runtime = runtime
        self.dataset_builder = DatasetBuilder()
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        self._preprocessor_loaded_from_disk = preprocessor_path is not None

    @staticmethod
    def _load_preprocessor(preprocessor_path: Optional[Path]) -> NetworkDataPreprocessor:
        if preprocessor_path is None:
            return NetworkDataPreprocessor()

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        with preprocessor_path.open("rb") as handle:
            preprocessor = pickle.load(handle)
        return preprocessor

    def stage_ingestion(self, new_rows: pd.DataFrame, buffer: Deque[dict]) -> int:
        records = new_rows.to_dict(orient="records")
        for row in records:
            buffer.append(row)
        return len(records)

    def stage_preprocessing(self, batch_df: pd.DataFrame) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        packet_df = self._convert_batch_to_packet_frame(batch_df)
        feature_df = self.dataset_builder.build_dataset(packet_df, window_size=1.0)

        fit = not self.preprocessor.is_fitted and not self._preprocessor_loaded_from_disk
        features, _ = self.preprocessor.preprocess_data(feature_df, fit=fit)

        if self.runtime.vectorizer is not None and "Info" in batch_df.columns:
            text_features = self.runtime.vectorizer.transform(batch_df["Info"].fillna("").astype(str))
            if hasattr(text_features, "toarray"):
                text_features = text_features.toarray()
            features = np.hstack([features, text_features])

        if self.runtime.model is not None and hasattr(self.runtime.model, "inputs"):
            input_count = len(getattr(self.runtime.model, "inputs", []))
            if input_count >= 2:
                sequence_features = features.reshape(features.shape[0], 1, features.shape[1])
                graph_features = self._build_graph_features(feature_df)
                return sequence_features, graph_features

        return features

    def stage_inference(self, features: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        return self.runtime.predict(features)

    @staticmethod
    def _build_graph_features(feature_df: pd.DataFrame) -> np.ndarray:
        src_hash = feature_df["src_ip"].astype(str).apply(lambda x: hash(x) % 1000).to_numpy(dtype=np.float32)
        dst_hash = feature_df["dst_ip"].astype(str).apply(lambda x: hash(x) % 1000).to_numpy(dtype=np.float32)
        edge_activity = feature_df["connection_count"].to_numpy(dtype=np.float32)
        packet_rate = feature_df["packet_rate"].to_numpy(dtype=np.float32)

        graph_features = np.stack([src_hash, dst_hash, edge_activity, packet_rate], axis=1)
        graph_features /= np.array([1000.0, 1000.0, 50.0, 50.0], dtype=np.float32)
        return graph_features

    @staticmethod
    def _convert_batch_to_packet_frame(batch_df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in batch_df.columns:
            timestamp_series = pd.to_numeric(batch_df["timestamp"], errors="coerce")
        elif "Time" in batch_df.columns:
            timestamp_series = pd.to_numeric(batch_df["Time"], errors="coerce")
        else:
            timestamp_series = pd.Series(np.arange(len(batch_df), dtype=float))

        packet_df = pd.DataFrame(
            {
                "timestamp": timestamp_series.fillna(0.0),
                "src_ip": batch_df.get("src_ip", batch_df.get("Source", "0.0.0.0")),
                "dst_ip": batch_df.get("dst_ip", batch_df.get("Destination", "0.0.0.0")),
                "protocol": batch_df.get("protocol", batch_df.get("Protocol", "TCP")),
                "packet_size": pd.to_numeric(
                    batch_df.get("packet_size", batch_df.get("Length", 0)), errors="coerce"
                ).fillna(0.0),
                "src_port": batch_df.get("src_port", None),
                "dst_port": batch_df.get("dst_port", None),
            }
        )
        return packet_df


def pop_batch(buffer: Deque[dict], batch_size: int) -> pd.DataFrame:
    rows = [buffer.popleft() for _ in range(min(batch_size, len(buffer)))]
    return pd.DataFrame(rows)


def print_batch_predictions(batch_number: int, predictions: List[str]) -> None:
    print(f"\n[Batch {batch_number}]")
    for prediction in predictions:
        print(prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time CSV streaming anomaly detector")
    parser.add_argument("--csv-path", required=True, type=Path, help="Path to streaming CSV file")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to model (.pkl/.joblib/.pth/.pt)")
    parser.add_argument("--label-encoder-path", type=Path, default=None, help="Path to label encoder pickle")
    parser.add_argument("--vectorizer-path", type=Path, default=None, help="Path to TF-IDF vectorizer pickle")
    parser.add_argument("--preprocessor-path", type=Path, default=None, help="Path to fitted preprocessor pickle")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of rows per inference batch")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling delay between CSV checks")
    parser.add_argument(
        "--start-from-end",
        action="store_true",
        help="Ignore existing rows and start processing only newly appended rows",
    )
    return parser.parse_args()


def run_streaming_detector(args: argparse.Namespace) -> None:
    reader = CSVRowStreamReader(args.csv_path, start_from_end=args.start_from_end)
    runtime = ModelRuntime(args.model_path, args.label_encoder_path, args.vectorizer_path)
    pipeline = StreamingPipeline(runtime, preprocessor_path=args.preprocessor_path)

    buffer: Deque[dict] = deque()
    batch_counter = 0

    print("Streaming detector started")
    print(f"CSV: {args.csv_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Poll interval: {args.poll_seconds} sec")
    print(f"Start from end: {args.start_from_end}")

    if runtime.model is None:
        print("Warning: no model provided. Fallback prediction is 'normal'.")

    while True:
        if not reader._initialized:
            try:
                reader.initialize()
                print("CSV source detected. Streaming is active.")
            except FileNotFoundError:
                print(f"Waiting for CSV file to appear: {args.csv_path}")
                time.sleep(args.poll_seconds)
                continue

        new_rows = reader.read_new_rows()
        added_count = pipeline.stage_ingestion(new_rows, buffer)

        if added_count > 0:
            print(
                f"Ingestion: +{added_count} rows, buffer={len(buffer)}, last_read_index={reader.last_read_index}"
            )

        while len(buffer) >= args.batch_size:
            batch_counter += 1
            batch_df = pop_batch(buffer, args.batch_size)
            features = pipeline.stage_preprocessing(batch_df)
            predictions = pipeline.stage_inference(features)
            print_batch_predictions(batch_counter, predictions)

        time.sleep(args.poll_seconds)


def main() -> None:
    args = parse_args()
    try:
        run_streaming_detector(args)
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    main()
