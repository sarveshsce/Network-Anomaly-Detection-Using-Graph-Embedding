import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os

# GAT and PyTorch imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# Set page configuration
st.set_page_config(page_title="Network Anomaly Detection with GAT", layout="wide")

# Title and description
st.title("Network Anomaly Detection Dashboard (GAT-Based)")
st.markdown("""
This dashboard visualizes network traffic using Graph Attention Networks (GAT) and detects anomalies using rule-based methods.
""")

# GAT model class
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

# Data loader
@st.cache_data

def load_and_process_data(file):
    df = pd.read_csv(file) if file else pd.read_csv("synthetic_network_traffic.csv")

    column_map = {
        'source_ip': 'src_ip', 'destination_ip': 'dst_ip',
        'source_bytes': 'src_bytes', 'destination_bytes': 'dst_bytes',
        'label': 'attack_type'
    }
    df.rename(columns=column_map, inplace=True)

    needed_cols = ['src_ip', 'src_bytes', 'dst_bytes', 'attack_type']
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0

    df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.date_range('2025-03-27', periods=len(df), freq='S')))
    return df

# Build PyG Data

def build_pyg_data(df):
    vocab = df['src_ip'].unique().tolist()
    df_grouped = df.groupby('src_ip')[['src_bytes', 'dst_bytes']].mean().reindex(vocab).fillna(0)
    features = df_grouped[['src_bytes', 'dst_bytes']].values

    similarity_matrix = cosine_similarity(features)
    edge_index = []
    threshold = 0.7
    for i in range(len(vocab)):
        for j in range(i+1, len(vocab)):
            if similarity_matrix[i, j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index), vocab

# GAT Embedding Generator

def generate_gat_embeddings(data):
    model = GAT(input_dim=data.x.shape[1], hidden_dim=8, output_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = torch.mean(torch.var(out, dim=0))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings = model(data).numpy()
    return embeddings

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file or st.button("Use Default Dataset"):
    df = load_and_process_data(uploaded_file)

    st.subheader("Dataset Statistics")
    st.write(df[['src_ip', 'src_bytes', 'dst_bytes', 'attack_type']].describe())

    st.subheader("GAT Embedding Generation")
    with st.spinner("Generating GAT-based graph embeddings..."):
        pyg_data, vocab = build_pyg_data(df)
        embeddings = generate_gat_embeddings(pyg_data)

    st.write("Sample GAT Embeddings:")
    emb_df = pd.DataFrame(embeddings, columns=[f'embed_{i}' for i in range(embeddings.shape[1])])
    emb_df['Node'] = vocab
    st.dataframe(emb_df)

else:
    st.info("Please upload a CSV file or click 'Use Default Dataset' to proceed.")