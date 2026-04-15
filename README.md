# Network Anomaly Detection with LSTM + Graph Features

An end-to-end network intrusion detection project that supports:

- packet capture and CSV ingestion
- feature engineering over traffic windows
- hybrid LSTM + graph-style model inference
- optional LLM-based incident analysis and remediation
- terminal and Streamlit dashboard workflows

## 1) What This Project Does

This repository detects suspicious network traffic from packet logs and presents:

- class predictions (benign / attack family)
- confidence scores
- suspicious source -> destination flows
- optional AI-generated root-cause and remediation guidance

## 2) High-Level Pipeline

```text
Network Traffic
    -> Packet Capture / CSV Input
    -> Dataset Builder (windowed features)
    -> Preprocessing (encoding + scaling)
    -> LSTM + graph-input model
    -> Prediction labels + confidence
    -> Optional CrewAI analysis
    -> CLI output or Streamlit dashboard
```

## 3) Input and Output at Each Step

### Step A: Ingestion

Input:
- live packets from Scapy capture
- or CSV files like `data/network_traffic_dataset.csv`, `data/captured_packets.csv`, `data/small_dataset.csv`

Output:
- packet-level rows with fields such as timestamp, src/dst IP, protocol, packet size, ports, flags, info

Primary files:
- `capture/packet_capture.py`
- `streaming_detector.py`
- `streaming_detector_final.py`

### Step B: Dataset Builder

Input:
- packet-level DataFrame

Processing:
- per-time-window aggregation and network statistics

Output features include:
- `packet_rate`
- `connection_count`
- `avg_packet_size`
- `dominant_protocol`

Primary file:
- `dataset/dataset_builder.py`

### Step C: Preprocessing

Input:
- engineered feature DataFrame

Processing:
- protocol and flags encoding
- numeric scaling
- inference-time safe handling of unknown categories

Output:
- model-ready numeric arrays

Primary file:
- `preprocessing/preprocessing.py`

### Step D: Model Inference

Input:
- sequence tensor for LSTM branch
- graph-style tensor from src/dst hash + traffic activity
- saved model and label encoder artifacts

Output:
- predicted class per sequence/packet
- confidence per prediction

Primary files:
- `models/lstm_gcn_model.py`
- `streaming_detector.py`
- `streaming_detector_final.py`
- `webapp.py`

### Step E: Optional Agentic Analysis

Input:
- detected non-benign events (attack type, protocol, flags, packet rate, summary)

Output:
- anomaly analysis: type, cause, confidence, evidence
- remediation: actions, priority, notes

Primary file:
- `agents.py`

## 4) Repository Layout

```text
mp2/
    agents.py
    create_small_dataset.py
    streaming_detector.py
    streaming_detector_final.py
    webapp.py
    requirements.txt
    README.md
    artifacts/
        lstm_gcn_model.keras
        label_encoder.pkl
        preprocessor.pkl
        protocol_encoder.pkl
        scaler.pkl
        tfidf_vectorizer.pkl
    capture/
        packet_capture.py
    data/
        network_traffic_dataset.csv
        captured_packets.csv
        small_dataset.csv
    dataset/
        dataset_builder.py
        dataset_loader.py
    models/
        lstm_gcn_model.py
    preprocessing/
        preprocessing.py
```

## 5) Environment Setup

### Recommended

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Python Version

- Python 3.9+ is recommended by the codebase dependencies.

## 6) Required Artifacts

For detection/inference you need:

- `artifacts/lstm_gcn_model.keras`
- `artifacts/label_encoder.pkl`
- `artifacts/preprocessor.pkl`

These are already present in this repository.

## 7) Run Modes

### A) Streamlit Dashboard (recommended for demos)

```bash
streamlit run webapp.py --server.port 8501
```

Open in browser:
- `http://localhost:8501`

UI structure:
- Detection tab: upload/select CSV, run detection, metrics/charts/feed/threat table
- Analysis & Remediation tab: run CrewAI analysis on detected threats

### B) Terminal detector (full pipeline)

```bash
python -u streaming_detector_final.py \
    --csv-path data/small_dataset.csv \
    --model-path artifacts/lstm_gcn_model.keras \
    --label-encoder-path artifacts/label_encoder.pkl \
    --preprocessor-path artifacts/preprocessor.pkl \
    --batch-size 10 \
    --llm-model gemini/gemini-1.5-flash \
    --max-batch-details 3
```

### C) Streaming CSV monitor loop

```bash
python streaming_detector.py \
    --csv-path data/captured_packets.csv \
    --model-path artifacts/lstm_gcn_model.keras \
    --label-encoder-path artifacts/label_encoder.pkl \
    --preprocessor-path artifacts/preprocessor.pkl \
    --batch-size 50 \
    --poll-seconds 2.0
```

## 8) Training / Re-Training

To train and save artifacts again:

```bash
python models/lstm_gcn_model.py \
    --csv-path data/network_traffic_dataset.csv \
    --output-dir artifacts \
    --epochs 8 \
    --batch-size 32
```

Expected outputs in `artifacts/`:
- model file (`.keras` or fallback `.pkl`)
- preprocessing artifacts
- label encoder

## 9) Agent (LLM) Configuration

Set API key in `.env` (project root), based on provider:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Gemini models
- `OPENAI_API_KEY` for OpenAI models
- `GROQ_API_KEY` for Groq models

If keys are missing, agent calls return safe fallback responses.

## 10) CSV Schema Compatibility

Supported column aliases in current code include:

- timestamp: `timestamp` or `Time`
- source IP: `src_ip` or `Source`
- destination IP: `dst_ip` or `Destination`
- protocol: `protocol` or `Protocol`
- packet size: `packet_size` or `Length`

Optional columns:
- `src_port`, `dst_port`, `flags`, `Info`/`info`, `label`

## 11) Troubleshooting

### Streamlit starts but dashboard looks empty

- In Detection tab, click `Run Detection` after selecting/uploading CSV.

### Model file not found

- Verify sidebar paths in dashboard or CLI arguments.
- Confirm files exist under `artifacts/`.

### Agent analysis does not run

- Ensure dependencies are installed from `requirements.txt`.
- Ensure provider API key is set in `.env`.

### Capture issues on Windows

- Packet capture may require admin privileges and correct network interface.
- Capture module is Scapy-based, not direct Wireshark runtime integration.

## 12) Current Status

Implemented and usable now:

- dataset loading and building
- preprocessing artifacts
- model training/inference scripts
- terminal detector flow
- Streamlit tabbed dashboard
- optional CrewAI analysis/remediation

## 13) License

Educational / research usage.
