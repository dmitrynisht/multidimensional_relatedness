import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dgl.nn import SAGEConv
from dgl.dataloading.negative_sampler import GlobalUniform
import random
import numpy as np
from typing import Optional
from p10_Chroma_dbm import ChromaEmbeddingStore


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_exports_bipartite_graph_v2(exports_df=None, node_mapping=None):
    if exports_df is None:
        raise ValueError("exports_df must be provided")

    if node_mapping is None:
        raise ValueError("node_mapping must be provided")

    # Use the global node mapping to encode edges
    country_encoded = exports_df["country_id"].map(node_mapping).values
    product_encoded = exports_df["product_id"].map(node_mapping).values

    # Convert the list of NumPy arrays to a single NumPy array, for efficiency and warning suppression
    edges = np.array([country_encoded, product_encoded])

    # Convert the NumPy array to a PyTorch tensor
    edges = torch.tensor(edges, dtype=torch.int64)
    weights = torch.tensor(
        np.log1p(exports_df["average_export_value"].values),
        dtype=torch.float32,
    )

    # graph = dgl.graph((edges[0], edges[1]))
    graph = dgl.graph((edges[0], edges[1]), num_nodes=len(node_mapping))
    graph.edata["weight"] = weights  # Assign export values as edge weights

    return graph


# GraphSAGE Model for a Single Year
class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGEModel, self).__init__()
        self.sage1 = SAGEConv(in_feats, hidden_feats, "mean")
        self.sage2 = SAGEConv(hidden_feats, hidden_feats, "pool")
        self.sage3 = SAGEConv(hidden_feats, hidden_feats, "gcn")
        self.sage4 = SAGEConv(hidden_feats, hidden_feats, "mean")
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, graph, features):
        h = self.sage1(graph, features)
        h = self.sage2(graph, h)
        h = self.sage3(graph, h)
        h = self.sage4(graph, h)
        return self.fc(h)  # Output embeddings


# GRU-based Temporal Graph Model
class TemporalGraphModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_years):
        super().__init__()
        self.graphsage = GraphSAGEModel(in_feats, hidden_feats, out_feats)
        self.gru = nn.GRU(
            input_size=out_feats,
            hidden_size=hidden_feats,
            num_layers=1,  # 2,
            batch_first=True,
        )

    def forward(self, graphs_per_year, features_per_year):
        yearly_embeddings = []

        for graph, features in zip(graphs_per_year, features_per_year):
            with graph.local_scope():
                # Get yearly embeddings
                h = self.graphsage(graph, features)  # shape: [num_nodes, out_feats]
                yearly_embeddings.append(h.unsqueeze(1))  # add time dimension

        # Shape: [num_nodes, num_years, out_feats]
        embeddings_sequence = torch.cat(yearly_embeddings, dim=1)

        # GRU processes embeddings over years
        gru_out, _ = self.gru(embeddings_sequence)  # same shape as input
        final_embeddings = gru_out[
            :, -1, :
        ]  # embeddings from last time step (last year's prediction)

        return final_embeddings, yearly_embeddings  # for loss + final output


# (Updated, v2) Example Data Preparation with global node indexing
def prepare_data_v2(df, f_debug=False):
    graphs_per_year = []
    features_per_year = []

    # Create global node ID mapping
    country_nodes = df["country_id"].sort_values().unique()
    product_nodes = df["product_id"].sort_values().unique()
    all_nodes = pd.Index(np.concatenate([country_nodes, product_nodes]))
    node_ids, node_labels = pd.factorize(all_nodes)
    node_lookup = dict(zip(node_labels, node_ids))

    for year in df["year"].unique():
        df_year = df[df["year"] == year]

        # Use the global node mapping to encode edges
        graph = generate_exports_bipartite_graph_v2(
            exports_df=df_year, node_mapping=node_lookup
        )
        degrees = graph.in_degrees()

        # Sanity check
        if f_debug:
            print(
                f"Year {year} → degrees.shape[0]: {degrees.shape[0]}, nodes: {graph.num_nodes()}"
            )

            if degrees.shape[0] < graph.num_nodes():
                print(
                    f"Year {year} → degrees.shape[0]: {degrees.shape[0]}, nodes: {graph.num_nodes()}"
                )
                padding = torch.zeros(graph.num_nodes() - degrees.shape[0])
                degrees = torch.cat([degrees, padding])
        # features = torch.log1p(degrees.float().unsqueeze(1)).repeat(1, 16)
        features = torch.cat(
            [
                torch.log1p(degrees.float().unsqueeze(1)),  # Original log-degree
                degrees.float().unsqueeze(1) / degrees.max(),  # Normalized degree
                torch.randn(
                    degrees.shape[0], 14
                ),  # Random noise to break symmetry: to prevent identical nodes from having the same features
            ],
            dim=1,
        )  # Produces 16D feature vectors

        # Sanity check
        if f_debug:
            print(
                f"Year {year} → nodes: {graph.num_nodes()}, features: {features.shape[0]}"
            )

        graphs_per_year.append(graph)
        features_per_year.append(features)

    return graphs_per_year, features_per_year


def compute_temporal_consistency_loss_v2(
    yearly_embeddings, 
    lambda_temporal=1.0, 
    detach_past=False,
    **kwargs
):
    """
    Encourage embeddings to change smoothly over time by minimizing
    the MSE between adjacent year embeddings.
    Goal: Keep node embeddings consistent across adjacent years.
    Focus: Temporal stability of node embeddings.
    Assumption: Trade patterns evolve smoothly over time. Nodes' embeddings should not change drastically from one year to the next.
    Good for:Modeling gradual economic shifts.
    Loss: MSE(embedding_t, embedding_{t-1})
    """
    if len(yearly_embeddings) <= 1:
        loss = torch.tensor(0.0, device=yearly_embeddings[0].device, requires_grad=True)
        
        if "output" in kwargs:
            embeddings = kwargs["output"]
        else:
            print("No output provided")
            return loss

        if "graphs_per_year" in kwargs:
            graphs_per_year = kwargs["graphs_per_year"]
            graph = graphs_per_year[0]
        else:
            print("No graphs_per_year provided")
            return loss
        
        # Assume embeddings is shape [num_nodes, dim]
        print(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

        # Positive edges and weights
        print("Sampling positives...")
        src_pos, dst_pos = graph.edges()
        pos_weights = graph.edata["weight"]
        print(f"Positive edge shape: src={src_pos.shape}, dst={dst_pos.shape}")

        # Weighed edges solution 
        # Negative edges (sampled)
        print("Sampling negatives...")
        # neg_src, neg_dst = sample_negative_edges(graph, num_neg_samples=len(src_pos))
        # Create a DGL Negative Sampler
        neg_sampler = GlobalUniform(1)  # 1 negative per positiveedge
        # Extract negative edges
        # neg_src, neg_dst = neg_graph.edges()
        src_neg, dst_neg = neg_sampler(graph, torch.arange(graph.num_edges()))
        neg_weights = torch.zeros(len(src_neg), device=embeddings.device)
        print(f"Negative edge shape: src={src_neg.shape}, dst={dst_neg.shape}")

        # Compute scores
        print("Computing positive scores...")
        pos_scores = compute_scores(embeddings, src_pos, dst_pos)  # dot product
        print("Computing negative scores...")
        neg_scores = compute_scores(embeddings, src_neg, dst_neg)

        # Concatenate
        all_scores = torch.cat([pos_scores, neg_scores])
        all_targets = torch.cat([pos_weights, neg_weights])

        # Loss: mean squared error between predicted and true weights
        print("Computing loss...")
        loss = F.mse_loss(all_scores, all_targets)
        print(f"========================>>>>> loss: {loss}")
        return loss

    temporal_loss = 0.0
    for t in range(1, len(yearly_embeddings)):
        target = yearly_embeddings[t - 1].detach() if detach_past else yearly_embeddings[t - 1]
        temporal_loss += F.mse_loss(yearly_embeddings[t], target)

    return lambda_temporal * temporal_loss / (len(yearly_embeddings) - 1)


# Dot product scores
def compute_scores(embeddings, src, dst):
    return (embeddings[src] * embeddings[dst]).sum(dim=1)


def sample_negative_edges(graph, num_neg_samples):
    num_nodes = graph.num_nodes()

    neg_src = torch.randint(0, num_nodes, (num_neg_samples,))
    neg_dst = torch.randint(0, num_nodes, (num_neg_samples,))

    # Remove negatives that accidentally exist in the graph
    edge_set = set(zip(graph.edges()[0].tolist(), graph.edges()[1].tolist()))
    neg_edges = []
    for s, d in zip(neg_src, neg_dst):
        if (s.item(), d.item()) not in edge_set:
            neg_edges.append((s.item(), d.item()))
        if len(neg_edges) >= num_neg_samples:
            break

    if len(neg_edges) < num_neg_samples:
        # Recursively sample more if needed
        return sample_negative_edges(graph, num_neg_samples)

    neg_src, neg_dst = zip(*neg_edges)
    return torch.tensor(neg_src), torch.tensor(neg_dst)


# Sample training loop
def train_TemporalSAGE_pure_vdb(
    df,
    num_epochs=10,
    version=2,
    lambda_temporal=0.05,
    f_debug=False,
    embedding_store: Optional[ChromaEmbeddingStore] = None,
    return_embeddings: bool = True,
):
    """VDB: Vector Database (ChromaDB) for storing embeddings.
    Args:
        df: DataFrame containing the export data.
        num_epochs: Number of training epochs.
        version: Version of the model (default is 2).
        lambda_temporal: Weight for the temporal consistency loss.
        f_debug: Flag for debugging information.
        embedding_store: Optional ChromaEmbeddingStore to store embeddings.
        return_embeddings: Flag to return embeddings or not.
    Returns:
        model: Trained TemporalGraphModel.
        embeddings: None or a list of the final output embeddings, if return_embeddings is True.
    """
    torch.manual_seed(42)
    graphs_per_year, features_per_year = prepare_data_v2(df, f_debug=f_debug)
    in_feats = features_per_year[0].shape[1]
    model = TemporalGraphModel(
        in_feats=in_feats, hidden_feats=32, out_feats=16, num_years=len(graphs_per_year)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    embeddings = None  # Store last output for visualization

    # Debug: Check if the model is initialized correctly
    if f_debug:
        print(f"==> Debug start")
        # Debug: Run a forward pass before training to inspect embeddings
        output, all_yearly_embeddings = model(graphs_per_year, features_per_year)

        # Debug: Check initial embedding structure
        print(
            f"First embedding shape: {all_yearly_embeddings[0].shape}"
        )  # Expected: (num_nodes, out_feats)
        print(
            f"Unique vectors in year 0: {torch.unique(all_yearly_embeddings[0], dim=0).shape[0]}"
        )

        # Debug: Print unique embeddings count for each year
        for i, emb in enumerate(all_yearly_embeddings):
            print(f"Year {i} unique embeddings: {torch.unique(emb, dim=0).shape[0]}")
        print(f"<== Debug end")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, all_yearly_embeddings = model(graphs_per_year, features_per_year)
        if f_debug:
            print(f"==> Debug: First embedding shape: {all_yearly_embeddings[0].shape}")
            print(
                f"==> Debug: Unique vectors in year 0: {torch.unique(all_yearly_embeddings[0], dim=0).shape[0]}"
            )
        loss = compute_temporal_consistency_loss_v2(
            all_yearly_embeddings, lambda_temporal=lambda_temporal
        )
        loss.backward()
        optimizer.step()
        if f_debug:
            print(f"Epoch {epoch+1} of {num_epochs}, Loss: {loss.item()}")
        embeddings = output.detach().cpu()  # Store output for visualization

    print(f"Output type: {type(embeddings)}")

    # Store embeddings if requested
    if embedding_store is not None:
        embedding_store.clear()
        embedding_store.store(embeddings)

    if return_embeddings:
        return model, embeddings
    else:
        return model


def testing_SAGE():
    import os
    import sys
    from pathlib import Path

    # # Add the project root to sys.path to import config
    # project_root = Path(__file__).resolve().parent.parent
    # if str(project_root) not in sys.path:
    #     sys.path.append(str(project_root))

    # # Import and run the configuration script
    # import config
    # from p01_raw_data_imports import read_hs92_parquet_data, save_hs92_parquet_data
    # from p14_GraphSAGE_v01 import train_TemporalSAGE_model

    # print(f"os.getcwd():\n{os.getcwd()}", "=" * 60, sep="\n")
    # # Define the relative path to the input data folder
    # input_file_location = os.path.join("data", "03_preprocessed")
    # input_file_name = "X_cp.parquet"
    # x_cp = read_hs92_parquet_data(
    #     input_file_location, input_file_name, f_convert_dtype=False
    # )
    # model, output = train_TemporalSAGE_model(x_cp)

    pass


def main(argv=None):
    if argv is None:
        argv = sys.argv

    s = "stop"

    # test comes here
    testing_SAGE()

    s = "stop"


if __name__ == "__main__":
    import sys

    sys.exit(main())
