import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# --- 1. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    ee_df = pd.read_csv('data_pp_a_ee.csv')        # electrons
    jet_photon_df = pd.read_csv('data_pp_jja_di_jet_photon_multiplicity.csv')  # jets/photons
    axion_df = pd.read_csv('pp_z_z_ax_a_ax_bb_di_bottomjets_photon_axion_data.csv')

    return ee_df, jet_photon_df, axion_df

def build_event_graph(event_electrons, event_jets_photons):

    nodes_features = []

    # Electrons
    for _, row in event_electrons.iterrows():
        feat = [row['px'], row['py'], row.get('ptl', 0), row.get('etal', 0)]
        nodes_features.append(feat)

    # Jets (dummy features for demo)
    jet_mult = int(event_jets_photons['Jet Multiplicity'])
    for _ in range(jet_mult):
        nodes_features.append([1, 0, 0, 0])

    # Photons (dummy features for demo)
    photon_mult = int(event_jets_photons['Photon Multiplicity'])
    for _ in range(photon_mult):
        nodes_features.append([0, 1, 0, 0])

    x = torch.tensor(nodes_features, dtype=torch.float)  

    N = x.shape[0]
    if N == 0:
        # Empty event
        adj = torch.zeros((0,0), dtype=torch.float)
    else:
        # Fully connected adjacency except self loops
        adj = torch.ones((N, N), dtype=torch.float) - torch.eye(N)

    # Dummy label (replace with real label)
    y = torch.tensor([1])  # binary label

    return x, adj, y

# --- 2. MANUAL GRAPH CONV LAYER ---

class SimpleGraphConv(nn.Module):

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, x, adj):
        # Add self-loops
        A_hat = adj + torch.eye(adj.size(0), device=adj.device)

        # Degree matrix
        D_hat_diag = torch.sum(A_hat, dim=1)
        D_hat_inv_sqrt = torch.diag(torch.pow(D_hat_diag, -0.5))

        # Normalize adjacency
        A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

        # Apply GCN layer
        support = self.linear(x)
        out = A_norm @ support

        return F.relu(out)

# --- 3. GNN MODEL (Stacking SimpleGraphConv layers + global pooling) ---

class ParticleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SimpleGraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SimpleGraphConv(hidden_channels, hidden_channels))
        self.convs.append(SimpleGraphConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = conv(x, adj)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        # Global mean pooling (average over nodes)
        x = torch.mean(x, dim=0)
        return x.unsqueeze(0)  

# --- 4. TRAINING & UTILS ---

def plot_graph(x, adj):
    # Convert adjacency & features to NetworkX graph for visualization
    G = nx.from_numpy_array(adj.cpu().numpy())
    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G, seed=42)
    # Node colors scaled by first feature dimension
    node_colors = x[:,0].cpu().numpy() if x.size(0) > 0 else []
    nx.draw(G, pos, node_color=node_colors, cmap='coolwarm', with_labels=True)
    st.pyplot(plt.gcf())
    plt.clf()

def collate_batch(graphs):
    # graphs: list of (x, adj, y)
    # Since adj size varies, we process batch as a list, no tensor batching
    return graphs

def train_epoch(model, graphs, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for x, adj, y in graphs:
        x = x.to(device)
        adj = adj.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x, adj)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(graphs), acc

def eval_epoch(model, graphs, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, adj, y in graphs:
            x = x.to(device)
            adj = adj.to(device)
            y = y.to(device)

            out = model(x, adj)
            loss = criterion(out, y)

            total_loss += loss.item()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(graphs), acc

# --- 5. STREAMLIT INTERFACE ---

def main():
    st.title("Particle Identification with GNN WITHOUT PyTorch Geometric")

    ee_df, jet_photon_df, axion_df = load_data()

    st.sidebar.header("Data Exploration")
    option = st.sidebar.selectbox("Choose Dataset", ["Electron pairs", "Jet/Photon multiplicity", "Axion candidates"])
    if option == "Electron pairs":
        st.dataframe(ee_df.head())
    elif option == "Jet/Photon multiplicity":
        st.dataframe(jet_photon_df.head())
    else:
        st.dataframe(axion_df.head())

    st.sidebar.header("Model and Training Config")

    hidden_dim = st.sidebar.slider("Hidden Layer Size", 16, 128, 64)
    num_layers = st.sidebar.slider("Number of GNN layers", 2, 4, 2)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1)
    epochs = st.sidebar.slider("Epochs", 1, 20, 10)
    learning_rate = st.sidebar.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")

    # Prepare simplified dataset of first N events (for demo)
    n_samples = min(50, len(ee_df))
    graphs = []
    labels = []
    for i in range(n_samples):
        ev_e = ee_df.iloc[[i]]
        ev_jp = jet_photon_df.iloc[[i]] if i < len(jet_photon_df) else jet_photon_df.iloc[[0]]

        x, adj, y = build_event_graph(ev_e, ev_jp)

        graphs.append((x, adj, y))

    # Dummy labels for example, randomly generated (replace by actual labels)
    for i in range(n_samples):
        graphs[i] = (graphs[i][0], graphs[i][1], torch.tensor([np.random.randint(0, 2)]))

    train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

    st.sidebar.header("Train Control")
    if st.sidebar.button("Train Model"):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.write("Training on device:", device)

        model = ParticleGNN(in_channels=4, hidden_channels=hidden_dim, out_channels=2,
                            num_layers=num_layers, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_graphs, optimizer, criterion, device)
            val_loss, val_acc = eval_epoch(model, val_graphs, criterion, device)
            st.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        st.success("Training completed!")

    # Graph Visualization Example on first event
    st.header("Example Event Graph Visualization")
    ex_x, ex_adj, _ = graphs[0]
    plot_graph(ex_x, ex_adj)

if __name__ == "__main__":
    main()
