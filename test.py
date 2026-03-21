"""
Basic real-time detection loop entrypoint.

Runs continuous CSV monitoring with buffering and batch inference.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from streaming_detector import run_streaming_detector


def main() -> None:
    args = Namespace(
        csv_path=Path("data/captured_packets.csv"),
        model_path=Path("artifacts/lstm_gcn_model.pkl"),
        label_encoder_path=Path("artifacts/label_encoder.pkl"),
        vectorizer_path=Path("artifacts/tfidf_vectorizer.pkl"),
        preprocessor_path=Path("artifacts/preprocessor.pkl"),
        batch_size=50,
        poll_seconds=2.0,
        start_from_end=True,
    )
    run_streaming_detector(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
