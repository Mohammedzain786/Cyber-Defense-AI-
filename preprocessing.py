import pandas as pd
import numpy as np
import torch
import networkx as nx
from sklearn.preprocessing import RobustScaler
from torch_geometric.utils import from_networkx


def aggregate_host_features(df, window='5min'):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    host_features = df.groupby(['src_ip', pd.Grouper(freq=window)]).agg({
        'dur': ['mean', 'std', 'max'],
        'sbytes': ['sum', 'mean'],
        'dbytes': ['sum', 'mean'],
        'ct_srv_dst': 'max',
        'proto': lambda x: x.nunique()
    }).reset_index()

    return host_features


def preprocess_data(df, sessions):
    # Step 1: Feature aggregation
    features = aggregate_host_features(df)

    # Flatten column names
    features.columns = ['_'.join(col).strip('_') for col in features.columns]

    # Step 2: Scaling
    X_raw = features.select_dtypes(include=[np.number]).values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Step 3: Graph construction
    G = nx.DiGraph()
    for _, row in sessions.iterrows():
        G.add_edge(row['src_ip'], row['dst_ip'],
                   weight=np.log1p(row['session_count']))

    # Convert to PyG
    data = from_networkx(G)
    data.x = torch.tensor(X_scaled, dtype=torch.float)

    return data