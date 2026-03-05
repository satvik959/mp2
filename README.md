# AI-Powered Network Monitoring & Intrusion Detection System

**LSTM + GCN based anomaly detection for network traffic**

---

## 🎯 Project Goal

Build an intelligent network monitoring system that:
- Captures real-time network packets
- Detects anomalies using LSTM + GCN deep learning
- Performs root-cause analysis
- Suggests automated remediation
- Visualizes network security in real-time

---

## 🏗️ System Architecture

```
Network Traffic → Packet Capture → Dataset Generation → Preprocessing 
    ↓
LSTM + GCN Model → Anomaly Detection → Analysis Agent
    ↓
Root Cause Identification → Remediation Engine → Dashboard Visualization
```

---

## 📁 Project Structure

```
network-anomaly-detection/
├── data/
│   ├── network_traffic_dataset.csv     # Sample Wireshark packet data
│   ├── captured_packets.csv            # Live captured packets
│   └── processed_dataset.csv           # Preprocessed data
├── capture/
│   └── packet_capture.py               # Live packet capture (Scapy)
├── dataset/
│   ├── dataset_loader.py               # Dataset loading & validation
│   └── dataset_builder.py              # Convert packets to ML dataset
├── preprocessing/
│   └── preprocessing.py                # Data cleaning & feature engineering
├── models/
│   └── lstm_gcn_model.py               # LSTM + GCN hybrid model (TODO)
├── detection/
│   └── anomaly_detector.py             # Anomaly detection logic (TODO)
├── analysis/
│   └── root_cause_analysis.py          # Root cause identification (TODO)
├── remediation/
│   └── remediation_engine.py           # Mitigation suggestions (TODO)
├── visualization/
│   └── dashboard.py                    # Streamlit/Dash dashboard (TODO)
├── main.py                             # Main execution pipeline (TODO)
├── requirements.txt
└── README.md
```

---

## 🚀 Phase 1: Completed ✅

### **Dataset & Loading**
- ✅ `network_traffic_dataset.csv` - 100 sample Wireshark packets
- ✅ `dataset_loader.py` - Loads, validates, cleans dataset
- ✅ `packet_capture.py` - Live packet capture using Scapy
- ✅ `dataset_builder.py` - Converts packets to ML-ready format
- ✅ `preprocessing.py` - Feature engineering & scaling

---

## 📊 Dataset Format

**Input Columns (from Wireshark):**
```csv
No,Time,Source,Destination,Protocol,Length,Info
10001,375.057,192.168.1.100,192.168.1.1,TCP,54,"SYN"
```

**Processed Features (for ML):**
```
- packet_size
- packet_rate
- connection_count
- avg_packet_size
- protocol_encoded
- label (normal/attack)
```

---

## 🔧 Installation

```bash
cd network-anomaly-detection
pip install -r requirements.txt
```

---

## 🧪 Testing Phase 1

### Test Dataset Loader
```bash
cd dataset
python dataset_loader.py
```

### Test Dataset Builder
```bash
cd dataset
python dataset_builder.py
```

### Test Preprocessing
```bash
cd preprocessing
python preprocessing.py
```

### Capture Live Packets (requires root)
```bash
cd capture
sudo python packet_capture.py
```

---

## 🎯 Next Steps (Phase 2)

- [ ] Implement LSTM + GCN hybrid model
- [ ] Build anomaly detector with confidence scoring
- [ ] Create analysis agent for root cause identification
- [ ] Develop remediation engine
- [ ] Build real-time visualization dashboard
- [ ] Integrate full pipeline in main.py

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **ML/DL:** TensorFlow/PyTorch, scikit-learn
- **Networking:** Scapy, PyShark
- **Graph:** NetworkX
- **Visualization:** Plotly, Dash, Streamlit
- **Data:** Pandas, NumPy

---

## 📖 Usage Example

```python
from dataset.dataset_loader import load_network_traffic
from preprocessing.preprocessing import NetworkDataPreprocessor

# Load dataset
df = load_network_traffic('../data/network_traffic_dataset.csv')

# Preprocess
preprocessor = NetworkDataPreprocessor()
X, y = preprocessor.preprocess_data(df)

# Train model (coming in Phase 2)
# model.fit(X, y)
```

---

## 🔬 Research Concept

**LSTM** learns temporal patterns in packet sequences  
**GCN** analyzes communication graph (IP nodes, traffic edges)  
**Hybrid approach** combines time-series + graph features for superior anomaly detection

---

## ⚠️ Requirements

- Python 3.9+
- Root/sudo access for live packet capture
- Network interface for monitoring
- Sufficient RAM for ML model training

---

## 📝 License

Educational/Research Project

---

**Status:** Phase 1 Complete - Dataset & Preprocessing Ready ✅
