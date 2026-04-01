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


# ── Agent integration (imported lazily to avoid crash if crewai not installed) ──
try:
    from agents import run_agents
    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False

# ── Tunable constants ────────────────────────────────────────────────────────
ANOMALY_THRESHOLD = 0.30   # trigger LLM only if ≥30% of batch is anomalous
LLM_MODEL         = "gemini/gemini-1.5-flash"
_last_agent_call  = 0.0    # epoch-seconds; rate-guard against back-to-back calls
LLM_COOLDOWN_SEC  = 30     # minimum seconds between LLM calls (token budget guard)
# ─────────────────────────────────────────────────────────────────────────────


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
        if self.label_encoder is not None:
            print("LABEL CLASSES:", self.label_encoder.classes_)

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
                    "Torch is required for .pth/.pt models."
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
                    "TensorFlow is required for .keras/.h5 models."
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
        feature_df["flags"] = batch_df["flags"].values

        if "Info" in batch_df.columns:
            feature_df["Info"] = batch_df["Info"].values
        if "flags" in batch_df.columns:
            feature_df["flags"] = batch_df["flags"].values
        else:
            feature_df["flags"] = "NONE"

        features, _ = self.preprocessor.preprocess_data(feature_df, fit=False)

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def pop_batch(buffer: Deque[dict], batch_size: int) -> pd.DataFrame:
    rows = [buffer.popleft() for _ in range(min(batch_size, len(buffer)))]
    return pd.DataFrame(rows)


def print_batch_predictions(batch_number: int, predictions: List[str]) -> None:
    print(f"\n[Batch {batch_number}]")

    total = len(predictions)
    attack_count = predictions.count('0')

    for prediction in predictions:
        if prediction == '1':
            print("🟢 NORMAL")
        else:
            print("🔴 ANOMALY")

    print("\n📊 Batch Summary:")
    if attack_count == 0:
        print("🟢 All Normal Traffic")
    elif attack_count < total * 0.3:
        print("🟡 Minor Anomalies (Probably Safe)")
    elif attack_count < total * 0.7:
        print("🟠 Suspicious Activity Detected")
    else:
        print("🔴 High Probability Attack 🚨")


def _extract_batch_stats(batch_df: pd.DataFrame, predictions: List[str]) -> dict:
    """
    Pull lightweight network stats from the batch.
    Uses the exact CSV columns: protocol, flags, src_ip, dst_ip, packet_size.
    """
    total = len(predictions)
    anomaly_count = predictions.count('0')
    anomaly_ratio = anomaly_count / total if total else 0.0

    # Protocol — column is 'protocol' (lowercase) in your CSV
    proto_col = "protocol" if "protocol" in batch_df.columns else "Protocol"
    if proto_col in batch_df.columns:
        proto_counts = batch_df[proto_col].value_counts()
        dominant_protocol = str(proto_counts.idxmax()) if len(proto_counts) else "UNKNOWN"
    else:
        dominant_protocol = "UNKNOWN"

    # Flags — single string per row like "SYN", "RST", "S"
    flags_seen: List[str] = []
    if "flags" in batch_df.columns:
        for f in batch_df["flags"].dropna().astype(str):
            # flags can be comma-separated or single values
            flags_seen.extend(x.strip().upper() for x in f.split(",") if x.strip())
        flags_seen = list(dict.fromkeys(flags_seen))[:6]   # unique, preserve order, cap at 6

    # Connections — unique (src_ip, dst_ip) pairs
    connection_count = 0
    src_col = "src_ip" if "src_ip" in batch_df.columns else "Source"
    dst_col = "dst_ip" if "dst_ip" in batch_df.columns else "Destination"
    if src_col in batch_df.columns and dst_col in batch_df.columns:
        connection_count = int(batch_df[[src_col, dst_col]].drop_duplicates().shape[0])

    # Packet rate: rough estimate — batch_size rows / ~1s window
    pkt_size_col = "packet_size" if "packet_size" in batch_df.columns else "Length"
    avg_pkt_size = float(
        pd.to_numeric(batch_df[pkt_size_col], errors="coerce").mean()
        if pkt_size_col in batch_df.columns else 0.0
    )
    packet_rate = total * 10  # batches polled ~every 2s but window ~0.1s; coarse estimate

    return {
        "anomaly_count": anomaly_count,
        "anomaly_ratio": anomaly_ratio,
        "protocol": dominant_protocol,
        "flags_pattern": flags_seen,
        "connection_count": connection_count,
        "packet_rate": packet_rate,
        "avg_pkt_size": avg_pkt_size,
    }


def _map_to_attack_type(stats: dict) -> str:
    """
    Lightweight heuristic — maps batch stats to an attack label for the LLM.
    No ML; just cheap signal.  The LLM refines this anyway.
    """
    flags = stats["flags_pattern"]
    rate  = stats["packet_rate"]
    conns = stats["connection_count"]

    if "SYN" in flags and rate > 500:
        return "flood"
    if conns > 20 and rate < 200:
        return "probe"
    if "RST" in flags or "FIN" in flags:
        return "exploit"
    if stats["avg_pkt_size"] > 1000:
        return "malware"
    return "exploit"   # safe default


def _call_agents(batch_df: pd.DataFrame, predictions: List[str], batch_number: int) -> None:
    """
    Extract stats → build payload → call run_agents → pretty-print result.
    Guarded by: cooldown timer, agents import check, exception handler.
    """
    global _last_agent_call

    if not _AGENTS_AVAILABLE:
        print("⚠️  agents.py not importable — skipping LLM analysis.")
        return

    now = time.time()
    if now - _last_agent_call < LLM_COOLDOWN_SEC:
        remaining = int(LLM_COOLDOWN_SEC - (now - _last_agent_call))
        print(f"⏳ LLM cooldown active — {remaining}s remaining. Skipping this batch.")
        return

    stats = _extract_batch_stats(batch_df, predictions)
    attack_type = _map_to_attack_type(stats)

    payload = {
        "prediction":       attack_type,
        "packet_rate":      stats["packet_rate"],
        "protocol":         stats["protocol"],
        "flags_pattern":    stats["flags_pattern"],
        "connection_count": stats["connection_count"],
        "batch_summary":    (
            f"{stats['anomaly_count']} anomalies out of {len(predictions)} "
            f"packets in batch #{batch_number} "
            f"({stats['anomaly_ratio']*100:.1f}% attack rate)"
        ),
    }

    print(f"\n⚠️  Anomaly threshold exceeded ({stats['anomaly_ratio']*100:.1f}%). "
          f"Triggering LLM analysis (attack hint: {attack_type})...")

    try:
        result = run_agents(payload, llm_model=LLM_MODEL)
        _last_agent_call = time.time()

        analyzer    = result.get("analyzer_output", {})
        remediation = result.get("remediation_output", {})

        print("\n" + "=" * 58)
        print("🔍  ANOMALY ANALYSIS")
        print("=" * 58)
        print(f"  Type       : {analyzer.get('anomaly_type', 'unknown')}")
        print(f"  Cause      : {analyzer.get('cause', 'N/A')}")
        print(f"  Confidence : {analyzer.get('confidence', 'N/A')}")

        evidence = analyzer.get("evidence", [])
        if evidence:
            print(f"  Evidence   : {', '.join(evidence)}")

        print(f"\n💡  REMEDIATION  [priority: {remediation.get('priority', 'N/A')}]")
        for action in remediation.get("recommended_actions", []):
            print(f"    • {action}")

        notes = remediation.get("notes", "")
        if notes:
            print(f"  Notes      : {notes}")

        print("=" * 58 + "\n")

    except Exception as exc:
        print(f"⚠️  Agent call failed: {exc}")
        print("   Continuing stream...\n")


# ── Arg parsing ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time CSV streaming anomaly detector")
    parser.add_argument("--csv-path",           required=True, type=Path)
    parser.add_argument("--model-path",          type=Path, default=None)
    parser.add_argument("--label-encoder-path",  type=Path, default=None)
    parser.add_argument("--vectorizer-path",     type=Path, default=None)
    parser.add_argument("--preprocessor-path",   type=Path, default=None)
    parser.add_argument("--batch-size",          type=int,   default=50)
    parser.add_argument("--poll-seconds",        type=float, default=2.0)
    parser.add_argument("--anomaly-threshold",   type=float, default=ANOMALY_THRESHOLD,
                        help="Fraction of anomalies in a batch to trigger LLM (default 0.30)")
    parser.add_argument("--llm-cooldown",        type=int,   default=LLM_COOLDOWN_SEC,
                        help="Minimum seconds between LLM calls (default 30)")
    parser.add_argument("--start-from-end",      action="store_true")
    return parser.parse_args()


# ── Core pipeline functions ──────────────────────────────────────────────────

def detect_anomaly(batch_df: pd.DataFrame, pipeline: StreamingPipeline) -> List[str]:
    try:
        processed   = pipeline.stage_preprocessing(batch_df)
        predictions = pipeline.stage_inference(processed)
        return predictions
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        return ["error"] * len(batch_df)


def run_streaming_detector(args: argparse.Namespace) -> None:
    global ANOMALY_THRESHOLD, LLM_COOLDOWN_SEC

    # Allow CLI overrides
    ANOMALY_THRESHOLD = args.anomaly_threshold
    LLM_COOLDOWN_SEC  = args.llm_cooldown

    reader   = CSVRowStreamReader(args.csv_path, start_from_end=args.start_from_end)
    runtime  = ModelRuntime(args.model_path, args.label_encoder_path, args.vectorizer_path)
    pipeline = StreamingPipeline(runtime, preprocessor_path=args.preprocessor_path)

    buffer: Deque[dict] = deque()
    batch_counter = 0

    print("Streaming detector started")
    print(f"CSV            : {args.csv_path}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Poll interval  : {args.poll_seconds}s")
    print(f"Anomaly thresh : {ANOMALY_THRESHOLD*100:.0f}%")
    print(f"LLM cooldown   : {LLM_COOLDOWN_SEC}s")
    print(f"Agents ready   : {_AGENTS_AVAILABLE}")
    print(f"Start from end : {args.start_from_end}")

    if runtime.model is None:
        print("Warning: no model provided — fallback prediction is 'normal'.")

    while True:
        if not reader._initialized:
            try:
                reader.initialize()
                print("CSV source detected. Streaming is active.")
            except FileNotFoundError:
                print(f"Waiting for CSV: {args.csv_path}")
                time.sleep(args.poll_seconds)
                continue

        new_rows  = reader.read_new_rows()
        added     = pipeline.stage_ingestion(new_rows, buffer)

        if added > 0:
            print(f"Ingestion: +{added} rows  buffer={len(buffer)}  idx={reader.last_read_index}")

        while len(buffer) >= args.batch_size:
            batch_counter += 1
            batch_df    = pop_batch(buffer, args.batch_size)
            predictions = detect_anomaly(batch_df, pipeline)

            print_batch_predictions(batch_counter, predictions)

            # ── Agent integration ────────────────────────────────────────────
            anomaly_ratio = predictions.count('0') / len(predictions) if predictions else 0.0
            if anomaly_ratio >= ANOMALY_THRESHOLD:
                _call_agents(batch_df, predictions, batch_counter)
            # ────────────────────────────────────────────────────────────────

        time.sleep(args.poll_seconds)


def main() -> None:
    args = parse_args()
    try:
        run_streaming_detector(args)
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    main()