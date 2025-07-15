# Network Anomaly Detection Dashboard

This project is a **Network Anomaly Detection Dashboard** built with [Streamlit](https://streamlit.io/), designed to visualize network traffic, generate graph embeddings, and detect anomalies using rule-based techniques. It supports datasets like UNSW-NB15, CIC-IDS2017, or synthetic network traffic data, and includes features like graph visualization, performance metrics, and attack simulation.

## Features
- **Data Upload**: Upload custom CSV datasets (e.g., UNSW-NB15, CIC-IDS2017) or use a default synthetic dataset.
- **Graph Embeddings**: Generate first-order and second-order graph embeddings based on network traffic features.
- **Visualization**: Interactive graphs (using NetworkX and Matplotlib) and anomaly frequency charts (using Plotly).
- **Rule-Based Anomaly Detection**: Detect anomalies with customizable rules for protocols like HTTP, DNS, FTP, etc.
- **Performance Metrics**: Evaluate detection performance with accuracy, precision, recall, F1-score, and confusion matrix.
- **Attack Simulation**: Simulate the attack scenarios (e.g., DoS, Port Errors, Suspicious Ports) and download generated traffic data.

## Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

## Installation

To get a local copy running, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/sarveshsce/Network-Anomaly-Detection-Using-Graph-Embedding.git
cd Network-Anomaly-Detection
```

2. **Install dependencies:**

```bash
pip install requirements.txt
```


3. **Run the application:**

```bash
streamlit run app.py
```


