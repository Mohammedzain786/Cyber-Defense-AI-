import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data

from model import NodeEncoder, GraphClassifier, NodeRecommender

# 🔹 Load models
@st.cache_resource
def load_models():
    encoder = NodeEncoder(in_channels=10)

    model_g = GraphClassifier(encoder)
    model_n = NodeRecommender(encoder)

    model_g.eval()
    model_n.eval()

    return model_g, model_n


model_g, model_n = load_models()

# 🔹 UI
st.set_page_config(page_title="Cyber Defense AI", layout="wide")
st.title("🚀 Cyber Defense System (Auto Graph Mode)")

uploaded_file = st.file_uploader("📂 Upload ANY CSV File")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean columns
    df.columns = df.columns.str.strip().str.lower()

    st.write("### 📊 Uploaded Data")
    st.dataframe(df)

    st.write("### 🧾 Columns")
    st.write(df.columns)

    # 🔥 AUTO GRAPH CREATION (works for ANY dataset)

    # Convert all values to string nodes
    df_str = df.astype(str)

    nodes = set()
    edges_src = []
    edges_dst = []

    # Connect row-wise values
    for i in range(len(df_str) - 1):
        src = "_".join(df_str.iloc[i].values[:2])   # take first 2 columns
        dst = "_".join(df_str.iloc[i+1].values[:2])

        nodes.add(src)
        nodes.add(dst)

        edges_src.append(src)
        edges_dst.append(dst)

    nodes = list(nodes)
    node_map = {n: i for i, n in enumerate(nodes)}

    edge_index = [
        [node_map[s] for s in edges_src],
        [node_map[d] for d in edges_dst]
    ]

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Features
    x = torch.rand((len(nodes), 10))

    data = Data(x=x, edge_index=edge_index)

    # Ensure batch
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    # 🔹 Model predictions
    with torch.no_grad():
        severity = model_g(data)
        actions = model_n(data)

    # 🔹 Results
    st.subheader("🔍 Analysis Results")

    threat_score = float(severity.mean())
    st.metric("Threat Score", threat_score)

    if threat_score > 0.5:
        st.error("🚨 High Threat Detected")
    else:
        st.success("✅ System Looks Safe")

    # Actions
    st.write("### 🛠️ Recommended Actions")
    actions_np = actions.detach().numpy()
    st.write(actions_np)

    # Chart
    st.write("### 📈 Action Distribution")
    fig, ax = plt.subplots()

    if actions_np.ndim == 1:
        values = actions_np
    else:
        values = actions_np[0]

    ax.bar(range(len(values)), values)
    st.pyplot(fig)

    # Graph visualization
    st.write("### 🌐 Network Graph")
    G = nx.Graph()

    for s, d in zip(edges_src, edges_dst):
        G.add_edge(s, d)

    st.write(G)
