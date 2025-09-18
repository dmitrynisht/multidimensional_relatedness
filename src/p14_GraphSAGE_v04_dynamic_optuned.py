from importlib import reload
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
import networkx as nx
import pandas as pd
import numpy as np
import optuna
import random
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import gc
from p12_Utils import (
    build_ordered_node_index, 
    get_oredered_node_ids,
    encode_product_hierarchy,
    get_node_type_tensor,
)
from p12_Utils import set_seed
import p14_GraphSAGE_v03_dynamic_chrdb

reload(p14_GraphSAGE_v03_dynamic_chrdb)
# TemporalGraphModel = p14_GraphSAGE_v03_dynamic_chrdb.TemporalGraphModel
generate_exports_bipartite_graph_v2 = (
    p14_GraphSAGE_v03_dynamic_chrdb.generate_exports_bipartite_graph_v2
)
compute_temporal_consistency_loss_v2 = (
    p14_GraphSAGE_v03_dynamic_chrdb.compute_temporal_consistency_loss_v2
)


class OptunaDynamicGraphSAGEModel(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_feats,
        out_feats,
        num_layers,
        agg_list,
        dropout=0.0,
        use_relu=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.use_relu = use_relu

        # Input layer
        self.layers.append(
            SAGEConv(in_feats, hidden_feats, aggregator_type=agg_list[0])
        )

        # Hidden layers
        for i in range(1, num_layers - 1):
            self.layers.append(
                SAGEConv(hidden_feats, hidden_feats, aggregator_type=agg_list[i])
            )

        # Output GraphSAGE layer
        self.layers.append(
            SAGEConv(
                hidden_feats, hidden_feats, aggregator_type=agg_list[num_layers - 1]
            )
        )

        # Final linear projection
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, graph, features):
        # Step 1: take input node features
        h = features
        for i in range(self.num_layers):
            # Step 2: next layer
            h = self.layers[i](graph, h)
            if self.use_relu:
                h = F.relu(h)
            h = self.dropout(h)

        return self.fc(h)  # Final projection


# GRU-based Temporal Graph Model
class OptunaTemporalGraphModel(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_feats,
        out_feats,
        num_years,
        num_layers,
        agg_list,
        dropout=0.0,
        gru_num_layers=1,
        use_relu=False,
    ):
        super().__init__()
        self.graphsage = OptunaDynamicGraphSAGEModel(
            in_feats, hidden_feats, out_feats, num_layers, agg_list, dropout, use_relu
        )
        # Adjusting to single-year inputs
        if num_years > 1:
            self.use_gru = True
            self.gru = nn.GRU(
                input_size=out_feats,
                hidden_size=hidden_feats,
                num_layers=gru_num_layers,
                batch_first=True,
            )
        else:
            self.use_gru = False

    def forward(self, graphs_per_year, features_per_year):
        yearly_embeddings = []

        for graph, features in zip(graphs_per_year, features_per_year):
            with graph.local_scope():
                # Get yearly embeddings
                h = self.graphsage(graph, features)  # shape: [num_nodes, out_feats]
                yearly_embeddings.append(h.unsqueeze(1))  # add time dimension

        # # Adjusting to single-year inputs
        if self.use_gru:
            # Shape: [num_nodes, num_years, out_feats]
            embeddings_sequence = torch.cat(yearly_embeddings, dim=1)

            # GRU processes embeddings over years
            gru_out, _ = self.gru(embeddings_sequence)  # same shape as input
            final_embeddings = gru_out[
                :, -1, :
            ]  # embeddings from last time step (last year's prediction)
        else:
            final_embeddings = yearly_embeddings[0].squeeze(1)

        return final_embeddings, yearly_embeddings  # for loss + final output


# Replace original GraphSAGEModel with Optuna-dynamic one
# GraphSAGEModel = OptunaDynamicGraphSAGE  # <- you can uncomment and integrate into TemporalGraphModel if modular


# Refactored, to accept trial-based hyperparams
def prepare_data_v2(
    df, f_debug=False, use_random_noise=True, node_type_encoding="none", node_type_dim=0
):
    """
    Topological features implemented:
        - normalized degrees (in-degree)
    """
    graphs_per_year = []
    features_per_year = []

    # Create global node ID mapping
    node_ids, node_labels = build_ordered_node_index(df)
    node_lookup = dict(zip(node_labels, node_ids))

    for year in df["year"].unique():
        df_year = df[df["year"] == year]
        graph = generate_exports_bipartite_graph_v2(
            exports_df=df_year, node_mapping=node_lookup
        )
        degrees = graph.in_degrees()

        if f_debug:
            print(
                f"Year {year} → degrees.shape[0]: {degrees.shape[0]}, nodes: {graph.num_nodes()}"
            )

        if degrees.shape[0] < graph.num_nodes():
            padding = torch.zeros(graph.num_nodes() - degrees.shape[0])
            degrees = torch.cat([degrees, padding])

        base_feats = [
            torch.log1p(degrees.float().unsqueeze(1)),
            degrees.float().unsqueeze(1) / degrees.max(),
        ]

        if use_random_noise:
            rand_dim = 14  # fixed here; could be tunable too
            base_feats.append(torch.randn(graph.num_nodes(), rand_dim))

        if node_type_encoding != "none":
            node_types = get_node_type_tensor(
                graph
            )  # must return 0 for country, 1 for product
            if node_type_encoding == "onehot":
                base_feats.append(F.one_hot(node_types, num_classes=2).float())
            elif node_type_encoding == "embedding":
                embed = nn.Embedding(2, node_type_dim).to(graph.device)
                base_feats.append(embed(node_types))

        features = torch.cat(base_feats, dim=1)
        graphs_per_year.append(graph)
        features_per_year.append(features)

    return graphs_per_year, features_per_year


# This version is for single-year inputs, in MoR style
def prepare_data_v3(
    df,
    f_debug=False,
    use_random_noise=True,
    node_type_encoding="none",
    node_type_dim=0,
    top_features=None,
    product_features=None,
):
    """
    Prepare data for Temporal/non-temporal GraphSAGE model.
    Args:
        df (pd.DataFrame): Input dataframe with columns "country_id", "product_id", "year", "exports", "imports"
        f_debug (bool, optional): Whether to print debug information. Defaults to False.
        use_random_noise (bool, optional): Whether to use random noise in feature generation. Defaults to True.
        node_type_encoding (str, optional): Encoding for node types. Options are "none", "onehot", "embedding". Defaults to "none".
        node_type_dim (int, optional): Dimension of node type embeddings. Defaults to 0.
        top_features (list, optional): List of topological features to include. Defaults to None.
            Topological features implemented:
            "nd" - normalized degrees (in-degree, out-degree)
            "cd" - centrality degree ((in + out degree) / num_nodes)
            "pr" - PageRank
            "ht" - HITS scores
            "cn" - Core number
        product_features (list, optional): List of product nodes features to include. Defaults to None.
            "ph" - Product hierarchy features
    Returns:
        graphs_per_year (list): List of graphs per year.
        features_per_year (list): List of features per year.
    """
    # If top_features is None, default to ["nd"]
    top_features = top_features if bool(top_features) else ["nd"]

    # If product_features is None, default to []
    product_features = product_features if bool(product_features) else []

    graphs_per_year = []
    features_per_year = []

    # Create global node ID mapping
    node_ids, node_labels = build_ordered_node_index(df)
    node_lookup = dict(zip(node_labels, node_ids))
    if f_debug:
        print("node_ids", node_ids)
        print("node_labels", node_labels)
        print("node_lookup", node_lookup)
        assert False, "Debugging forced stop"

    if "ht" in top_features:
        for year in df["year"].unique():
            df_year = df[df["year"] == year]
            graph = generate_exports_bipartite_graph_v2(
                exports_df=df_year, node_mapping=node_lookup
            )

            # Convert to NetworkX for graph-level algorithms
            nx_graph = graph.to_networkx().to_directed()

            # Map NetworkX node order to DGL node index
            nx_to_dgl = {node: i for i, node in enumerate(graph.nodes().numpy())}

            # Test run of HITS, if it fails the whole process will fail, to avoid confusion
            hits_hubs, hits_auth = nx.hits(
                nx_graph, max_iter=1000, tol=1e-08, normalized=True
            )

    if "cn" in top_features:
        for year in df["year"].unique():
            df_year = df[df["year"] == year]
            graph = generate_exports_bipartite_graph_v2(
                exports_df=df_year, node_mapping=node_lookup
            )

            # Convert to NetworkX for graph-level algorithms
            nx_graph = graph.to_networkx().to_directed()

            # Map NetworkX node order to DGL node index
            nx_to_dgl = {node: i for i, node in enumerate(graph.nodes().numpy())}

            # Test run of Core Number, if it fails the whole process will fail, to avoid confusion
            core_nums = nx.core_number(nx_graph)

    # In this version, we expect only one year of data at a time
    for year in df["year"].unique():
        df_year = df[df["year"] == year]
        graph = generate_exports_bipartite_graph_v2(
            exports_df=df_year, node_mapping=node_lookup
        )

        in_deg = graph.in_degrees().float().unsqueeze(1)
        out_deg = graph.out_degrees().float().unsqueeze(1)
        num_nodes = graph.num_nodes()
        if f_debug:
            print(
                f"Year {year} → in_degrees.shape: {in_deg.shape}, in_degrees.shape[0]: {in_deg.shape[0]}, nodes: {num_nodes}",
                f"Year {year} → out_degrees.shape: {out_deg.shape}, out_degrees.shape[0]: {out_deg.shape[0]}, nodes: {num_nodes}",
                sep="\n",
            )

        # Safety check: pad if needed
        if in_deg.shape[0] < num_nodes:
            pad_len = num_nodes - in_deg.shape[0]
            in_deg = torch.cat([in_deg, torch.zeros(pad_len, dtype=in_deg.dtype)])

        if out_deg.shape[0] < num_nodes:
            pad_len = num_nodes - out_deg.shape[0]
            out_deg = torch.cat([out_deg, torch.zeros(pad_len, dtype=out_deg.dtype)])

        max_in = in_deg.max().item() if in_deg.max().item() > 0 else 1.0
        max_out = out_deg.max().item() if out_deg.max().item() > 0 else 1.0

        # Base degree features
        base_feats = []

        # Normalized degree features
        if "nd" in top_features:
            nd_feats = [
                torch.log1p(in_deg),
                torch.log1p(out_deg),
                in_deg / max_in,
                out_deg / max_out,
            ]
            base_feats.extend(nd_feats)

        # Optional: combined centrality degree feature
        if "cd" in top_features:
            # Normalized by number of nodes in the graph
            degree_centrality = (in_deg + out_deg) / num_nodes
            base_feats.append(degree_centrality)

        # Convert to NetworkX for graph-level algorithms
        nx_graph = graph.to_networkx().to_directed()

        # Map NetworkX node order to DGL node index
        nx_to_dgl = {node: i for i, node in enumerate(graph.nodes().numpy())}

        # Optional: PageRank (PR)
        if "pr" in top_features:
            pr_scores = nx.pagerank(nx_graph)
            pr_tensor = torch.zeros(num_nodes)
            for nx_id, score in pr_scores.items():
                pr_tensor[nx_to_dgl[nx_id]] = score
            base_feats.append(pr_tensor.unsqueeze(1))  # (N, 1)

        # Optional: HITS (hub & authority scores)
        if "ht" in top_features:
            try:
                hits_hubs, hits_auth = nx.hits(
                    nx_graph, max_iter=1000, tol=1e-08, normalized=True
                )
                hub_tensor = torch.zeros(num_nodes)
                auth_tensor = torch.zeros(num_nodes)
                for nx_id in hits_hubs:
                    hub_tensor[nx_to_dgl[nx_id]] = hits_hubs[nx_id]
                    auth_tensor[nx_to_dgl[nx_id]] = hits_auth[nx_id]
                base_feats.append(hub_tensor.unsqueeze(1))
                base_feats.append(auth_tensor.unsqueeze(1))
            except nx.PowerIterationFailedConvergence:
                print(
                    f"HITS failed to converge in year {year}, defaulting to skipping HITS(hubs & authorities)."
                )

        # Optional: Core Number (K-core)
        if "cn" in top_features:
            try:
                core_nums = nx.core_number(nx_graph)
                core_tensor = torch.zeros(num_nodes)
                for nx_id, core in core_nums.items():
                    core_tensor[nx_to_dgl[nx_id]] = core
                base_feats.append(core_tensor.unsqueeze(1))
            except nx.NetworkXError as e:
                print(f"Core number failed for year {year}: {e}")

        # Optional: Product hierarchy features
        if "ph" in top_features:
            # Note: While the full product hierarchy format can go up to 6 levels (e.g., X.XXX.YYYY.ZZZZZ),
            # all products in the x_cp dataset follow the X.XXX.YYYY structure, corresponding to level 4.
            # Since x_cp["product_level"] is always 4, we only observe two meaningful hierarchy levels,
            # and will not include "product_level" as a feature in the graph.
            hierarchy_tensor = encode_product_hierarchy(
                df_year=df_year,
                node_labels=node_labels,
                hierarchy_column="product_id_hierarchy",
                hierarchy_levels=2,
                normalize=True,
                device=graph.device,
            )
            base_feats.append(hierarchy_tensor)

        # Optional: Random noise (helps with symmetry breaking)
        if use_random_noise:
            rand_dim = 14
            base_feats.append(torch.randn(num_nodes, rand_dim))

        # Optional: Node type encoding
        if node_type_encoding != "none":
            node_types = get_node_type_tensor(
                graph
            )  # must return 0 for country, 1 for product
            if node_type_encoding == "onehot":
                base_feats.append(F.one_hot(node_types, num_classes=2).float())
            elif node_type_encoding == "embedding":
                embed = nn.Embedding(2, node_type_dim).to(graph.device)
                base_feats.append(embed(node_types))

        features = torch.cat(base_feats, dim=1)
        graphs_per_year.append(graph)
        features_per_year.append(features)

        if f_debug:
            print(f"Year {year}: features shape = {features.shape}")

    return graphs_per_year, features_per_year


# def encode_product_hierarchy(
#     df_year: pd.DataFrame,
#     node_labels: pd.Index,
#     hierarchy_column: str = "product_id_hierarchy",
#     hierarchy_levels: int = 2,
#     normalize: bool = True,
#     device: str = "cpu",
# ) -> torch.Tensor:
#     """
#     Extract and encode product hierarchy levels as node features for product nodes.

#     Args:
#         df_year (pd.DataFrame): One-year slice of the main dataframe.
#         node_labels (pd.Index): Full list of node labels (countries + products) to align features with.
#         hierarchy_column (str): Column containing hierarchical product codes in format like "8.194.1875.10008".
#         hierarchy_levels (int): Number of hierarchy levels to extract from the hierarchy code.
#         normalize (bool): Whether to apply MinMax normalization to hierarchy levels.
#         device (str): Target device for the resulting tensor ("cpu" or "cuda").

#     Returns:
#         torch.Tensor: (num_nodes x num_features) tensor, with product features aligned by node,
#                       and zero-padded for non-product nodes.
#     """
#     assert (
#         hierarchy_column in df_year.columns
#     ), f"Missing required column: {hierarchy_column}"

#     # Data preparation. Base product metadata columns
#     # prod_cols = ["product_code", "product_level", "top_parent_id"]
#     # prod_cols = ["product_code", "top_parent_id"]
#     # "product_code" is a string, requires preprocessing
#     # "product_level" didn't have meaningful values
#     prod_cols = ["top_parent_id"]
#     required_cols = ["product_id", hierarchy_column] + prod_cols
#     hierarchy_df = df_year[required_cols].drop_duplicates()

#     # Clean hierarchy path: strip final segment (actual product_id)
#     clean_hierarchy_col = f"{hierarchy_column}_clean"
#     hierarchy_df[clean_hierarchy_col] = (
#         hierarchy_df[hierarchy_column].str.rsplit(".", n=1).str[0]
#     )

#     # Parse hierarchy levels into separate columns
#     levels_df = (
#         hierarchy_df[clean_hierarchy_col]
#         .str.split(".", expand=True)
#         .iloc[:, :hierarchy_levels]
#         .fillna("0")
#         .astype(int)
#     )
#     level_cols = [f"level_{i}" for i in range(hierarchy_levels)]

#     if normalize:
#         # Here we expect that each level of X.XXX.YYYY represents increasing specificity
#         scaler = MinMaxScaler()
#         levels_df = pd.DataFrame(
#             scaler.fit_transform(levels_df),
#             columns=level_cols,
#             index=hierarchy_df.index,
#         )

#     # Combine hierarchy + external product attributes
#     feature_df = pd.concat(
#         [hierarchy_df[["product_id"]], levels_df, hierarchy_df[prod_cols]],
#         axis=1,
#     )
#     feature_df.columns = ["product_id"] + level_cols + prod_cols
#     feature_df = feature_df.drop_duplicates("product_id").set_index("product_id")

#     # Build aligned feature matrix
#     feat_num = len(feature_df.columns)
#     external_attrs = []
#     for node in node_labels:
#         if node in feature_df.index:
#             external_attrs.append(feature_df.loc[node].values.tolist())
#         else:
#             external_attrs.append([0.0] * feat_num)

#     hierarchy_tensor = torch.tensor(external_attrs, dtype=torch.float32, device=device)

#     return hierarchy_tensor


def optuna_train_TemporalSAGE_pure_vdb(
    df,
    trial=None,
    num_epochs=10,
    version=2,
    lr=0.01,
    dropout=0,
    lambda_temporal=0.05,
    use_retain_graph=False,
    f_debug=False,
    embedding_store=None,
    return_embeddings=True,
    # in_feats=16,
    hidden_feats=32,
    out_feats=16,
    use_random_noise=True,
    node_type_encoding="none",
    node_type_dim=0,
    num_layers=4,
    agg_list=None,
    gru_num_layers=1,
    use_relu=False,
    top_features=None,
    trial_name=None,
):

    if len(df["year"].unique()) > 1:
        # Temporal multi-year model
        # adjust here to new arg name "top_features"; current implementation corresponds to the "nd" value
        # prepare_data = prepare_data_v2
        prepare_data = prepare_data_v3
        top_features = top_features if bool(top_features) else ["nd"]
    else:
        # Single-year model (MoR style)
        # adjust here to new arg name "top_features"; current implementation corresponds to the "nd-cd" value
        prepare_data = prepare_data_v3
        top_features = top_features if bool(top_features) else ["nd", "cd"]

    graphs_per_year, features_per_year = prepare_data(
        df,
        f_debug=f_debug,
        use_random_noise=use_random_noise,
        node_type_encoding=node_type_encoding,
        node_type_dim=node_type_dim,
        top_features=top_features,
    )

    in_feats = features_per_year[0].shape[1]
    if f_debug:
        print(
            f"module 'optuna_train_TemporalSAGE_pure_vdb': Input features shape (in_feats): {in_feats}"
        )
    trial_name = trial_name if bool(trial_name) else "Trial"

    model = OptunaTemporalGraphModel(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_years=len(graphs_per_year),
        num_layers=num_layers,
        agg_list=agg_list,
        dropout=dropout,
        gru_num_layers=gru_num_layers,
        use_relu=use_relu,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    detach_past = not use_retain_graph
    embeddings = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, yearly_embeddings = model(graphs_per_year, features_per_year)
        loss = compute_temporal_consistency_loss_v2(
            yearly_embeddings, lambda_temporal, detach_past=detach_past
        )
        # Dynamically choose how to backprop
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        optimizer.step()
        print(
            f"[{trial_name} {trial.number}] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}"
        )
        embeddings = output.detach().cpu()
        del loss
        yearly_embeddings = [y.detach() for y in yearly_embeddings]
        del yearly_embeddings
        torch.cuda.empty_cache()
        gc.collect()

    if embedding_store is not None:
        embedding_store.clear()
        embedding_store.store(embeddings)

    return model, embeddings


def optuna_train_TemporalSAGE_pure_vdb_sye(
    df,
    trial=None,
    num_epochs=10,
    version=2,
    lr=0.01,
    dropout=0,
    lambda_temporal=0.05,
    use_retain_graph=False,
    f_debug=False,
    embedding_store=None,
    return_embeddings=True,
    # in_feats=16,
    hidden_feats=32,
    out_feats=16,
    use_random_noise=True,
    node_type_encoding="none",
    node_type_dim=0,
    num_layers=4,
    agg_list=None,
    gru_num_layers=1,
    use_relu=False,
    top_features=None,
    trial_name=None,
    **kwargs,
):
    """sye - single year embeddings
    """
    f_deterministic = kwargs.get("f_deterministic", False)
    
    if f_deterministic:
        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if len(df["year"].unique()) > 1:
        # Temporal multi-year model
        # adjust here to new arg name "top_features"; current implementation corresponds to the "nd" value
        # prepare_data = prepare_data_v2
        prepare_data = prepare_data_v3
        top_features = top_features if bool(top_features) else ["nd"]
    else:
        # Single-year model (MoR style)
        # adjust here to new arg name "top_features"; current implementation corresponds to the "nd-cd" value
        prepare_data = prepare_data_v3
        top_features = top_features if bool(top_features) else ["nd", "cd"]

    graphs_per_year, features_per_year = prepare_data(
        df,
        f_debug=f_debug,
        use_random_noise=use_random_noise,
        node_type_encoding=node_type_encoding,
        node_type_dim=node_type_dim,
        top_features=top_features,
    )

    in_feats = features_per_year[0].shape[1]
    if f_debug:
        print(
            f"module 'optuna_train_TemporalSAGE_pure_vdb': Input features shape (in_feats): {in_feats}"
        )
    trial_name = trial_name if bool(trial_name) else "Trial"

    model = OptunaTemporalGraphModel(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_years=len(graphs_per_year),
        num_layers=num_layers,
        agg_list=agg_list,
        dropout=dropout,
        gru_num_layers=gru_num_layers,
        use_relu=use_relu,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    detach_past = not use_retain_graph
    embeddings = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, yearly_embeddings = model(graphs_per_year, features_per_year)
        loss = compute_temporal_consistency_loss_v2(
            yearly_embeddings, 
            lambda_temporal, 
            detach_past=detach_past,
            output=output.detach().cpu(),
            graphs_per_year=graphs_per_year
        )
        # Dynamically choose how to backprop
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        optimizer.step()
        print(
            f"[{trial_name} {trial.number}] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}"
        )
        embeddings = output.detach().cpu()
        print(f"Embeddings type: {type(embeddings)}")
        del loss
        yearly_embeddings = [y.detach() for y in yearly_embeddings]
        del yearly_embeddings
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Embeddings store: {embedding_store}")
    if embedding_store is not None:
        embedding_store.clear()
        embedding_store.store(embeddings)
        print(f"Embeddings {embedding_store.collection.name} stored in {embedding_store}")

    return model, embeddings


def optuna_train_TemporalSAGE_pure_vdb_sye_m_loss(
    df,
    trial=None,
    num_epochs=10,
    version=2,
    lr=0.01,
    dropout=0,
    lambda_temporal=0.05,
    use_retain_graph=False,
    f_debug=False,
    embedding_store=None,
    return_embeddings=True,
    # in_feats=16,
    hidden_feats=32,
    out_feats=16,
    use_random_noise=True,
    node_type_encoding="none",
    node_type_dim=0,
    num_layers=4,
    agg_list=None,
    gru_num_layers=1,
    use_relu=False,
    top_features=None,
    trial_name=None,
    **kwargs,
):
    """sye - single year embeddings
    """
    f_deterministic = kwargs.get("f_deterministic", False)
    
    if f_deterministic:
        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if len(df["year"].unique()) > 1:
        # Temporal multi-year model
        # adjust here to new arg name "top_features"; current implementation corresponds to the "nd" value
        # prepare_data = prepare_data_v2
        prepare_data = prepare_data_v3
        top_features = top_features if bool(top_features) else ["nd"]
    else:
        # Single-year model (MoR style)
        # adjust here to new arg name "top_features"; current implementation corresponds to the "nd-cd" value
        prepare_data = prepare_data_v3
        top_features = top_features if bool(top_features) else ["nd", "cd"]

    graphs_per_year, features_per_year = prepare_data(
        df,
        f_debug=f_debug,
        use_random_noise=use_random_noise,
        node_type_encoding=node_type_encoding,
        node_type_dim=node_type_dim,
        top_features=top_features,
    )

    in_feats = features_per_year[0].shape[1]
    if f_debug:
        print(
            f"module 'optuna_train_TemporalSAGE_pure_vdb': Input features shape (in_feats): {in_feats}"
        )
    trial_name = trial_name if bool(trial_name) else "Trial"

    model = OptunaTemporalGraphModel(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_years=len(graphs_per_year),
        num_layers=num_layers,
        agg_list=agg_list,
        dropout=dropout,
        gru_num_layers=gru_num_layers,
        use_relu=use_relu,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    detach_past = not use_retain_graph
    embeddings = None
    best_loss = float("inf")
    best_embeddings = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, yearly_embeddings = model(graphs_per_year, features_per_year)
        loss = compute_temporal_consistency_loss_v2(
            yearly_embeddings, 
            lambda_temporal, 
            detach_past=detach_past,
            output=output.detach().cpu(),
            graphs_per_year=graphs_per_year
        )
        # Dynamically choose how to backprop
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_embeddings = output.detach().cpu()
            print(f"Embeddings type: {type(best_embeddings)}")

        print(
            f"[{trial_name} {trial.number}] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, best loss: {best_loss}"
        )

        del loss
        yearly_embeddings = [y.detach() for y in yearly_embeddings]
        del yearly_embeddings
        torch.cuda.empty_cache()
        gc.collect()

    embeddings = best_embeddings
    print(f"Embeddings store: {embedding_store}")
    if embedding_store is not None:
        embedding_store.clear()
        embedding_store.store(embeddings)
        print(f"Embeddings {embedding_store.collection.name} stored in {embedding_store}")

    return model, embeddings


def eval_TemporalSAGE_pure_vdb(
    f_debug=False, df_eval=None, best_params=None, model=None, num_countries=144, y=None
):
    """
    Evaluate the model using linear regression on the country embeddings.

    Args:
        f_debug: Debug flag.
        df_eval: DataFrame for evaluation.
        best_params: Dictionary of best hyperparameters.
        model: Trained model.
        num_countries: Number of countries in the dataset.
        y: Target variable for regression.
    Returns:
        r2_score: R-squared score of the linear regression model.
    """
    if best_params["use_random_noise"]:
        set_seed(42)  # for reproducibility

    graphs_eval, features_eval = prepare_data_v2(
        f_debug=f_debug,
        df=df_eval,
        use_random_noise=best_params["use_random_noise"],
        node_type_encoding=best_params["node_type_encoding"],
        node_type_dim=best_params.get("node_type_dim", 0),
    )

    model.eval()
    with torch.no_grad():
        output, _ = model(graphs_eval, features_eval)  # output = final embeddings

    country_embeddings = output[:num_countries]
    country_np = country_embeddings.numpy()

    model = LinearRegression()
    model.fit(country_np, y)
    r2_score = model.score(country_np, y)
    print(f"Linear Regression R² Score:")
    print(f"   R² = {r2_score:.9f}")

    return r2_score


# # Implement get_node_type_tensor utility function
# def get_node_type_tensor(graph):
#     """
#     Assign node types for bipartite graph:
#     Assumes countries are first N nodes and products are the rest.
#     Returns tensor of 0s (country) and 1s (product) of length = num_nodes.
#     """
#     # num_nodes = graph.num_nodes()
#     # num_edges = graph.num_edges()
#     in_degrees = graph.in_degrees()
#     out_degrees = graph.out_degrees()

#     country_nodes = torch.where(
#         out_degrees > in_degrees, torch.tensor(1.0), torch.tensor(0.0)
#     )
#     # Heuristic: country nodes tend to have more outgoing edges (exporting)

#     node_types = country_nodes.to(torch.int64)
#     return node_types


# Version 01: Using this set of hyperparameters for Optuna I could find the best r-squared score of 0.9185
def objective_v01(trial, df, y, num_countries=144, f_debug=False, **kwargs):
    # Hyperparameters
    if f_debug:
        num_epochs = trial.suggest_int("num_epochs", 2, 3)
    else:
        num_epochs = trial.suggest_int("num_epochs", 2, 30)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lambda_temporal = trial.suggest_float("lambda_temporal", 0.0, 1.0)
    use_retain_graph = trial.suggest_categorical("retain_graph", [True, False])
    # in_feats = 16  # fixed from feature construction logic
    hidden_feats = trial.suggest_int("hidden_feats", 16, 128)
    out_feats = trial.suggest_int("out_feats", 8, 64)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    num_layers = trial.suggest_int("num_layers", 2, 5)
    possible_aggs = ["mean", "pool", "gcn", "lstm"]
    agg_list = [
        trial.suggest_categorical(f"agg_layer_{i}", possible_aggs)
        for i in range(num_layers)
    ]
    gru_num_layers = trial.suggest_int("gru_num_layers", 1, 3)
    use_relu = trial.suggest_categorical("use_relu", [False, True])

    use_random_noise = trial.suggest_categorical("use_random_noise", [True, False])
    node_type_encoding = trial.suggest_categorical(
        "node_type_encoding", ["none", "onehot", "embedding"]
    )
    node_type_dim = 0
    if node_type_encoding == "embedding":
        node_type_dim = trial.suggest_int("node_type_dim", 2, 8)

    print("=" * 60, sep="\n")
    print(f"Trial {trial.number} Hyperparameters:")
    print(f"num_epochs: {num_epochs}")
    print(f"lr: {lr}")
    print(f"lambda_temporal: {lambda_temporal}")
    print(f"use_retain_graph: {use_retain_graph}")
    print(f"hidden_feats: {hidden_feats}")
    print(f"out_feats: {out_feats}")
    print(f"dropout: {dropout}")
    print(f"num_layers: {num_layers}")
    print(f"agg_list: {agg_list}")
    print(f"gru_num_layers: {gru_num_layers}")
    print(f"use_relu: {use_relu}")
    print(f"use_random_noise: {use_random_noise}")
    print(f"node_type_encoding: {node_type_encoding}")
    print(f"node_type_dim: {node_type_dim}")
    print("")

    embedding_store = None

    model, embeddings = optuna_train_TemporalSAGE_pure_vdb(
        df=df,
        trial=trial,
        num_epochs=num_epochs,
        lambda_temporal=lambda_temporal,
        use_retain_graph=use_retain_graph,
        f_debug=f_debug,
        embedding_store=embedding_store,
        return_embeddings=True,
        # in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        use_random_noise=use_random_noise,
        node_type_encoding=node_type_encoding,
        node_type_dim=node_type_dim,
        lr=lr,
        dropout=dropout,
        num_layers=num_layers,
        agg_list=agg_list,
        gru_num_layers=gru_num_layers,
        use_relu=use_relu,
    )

    try:
        country_embeddings = embeddings[:num_countries]
        country_np = country_embeddings.numpy()

        reg = LinearRegression()
        reg.fit(country_np, y)
        r2 = reg.score(country_np, y)

        trial.report(r2, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        torch.cuda.empty_cache()  # safe on CPU too
        gc.collect()

        return r2
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise optuna.exceptions.TrialPruned()


# Version 02: Reducing the search space for hyperparameters
def objective_r2(trial, df, y, num_countries=144, f_debug=False, **kwargs):
    # Hyperparameters
    if f_debug:
        num_epochs = trial.suggest_int("num_epochs", 2, 3)
    else:
        num_epochs = trial.suggest_int("num_epochs", 2, 15)  # reduced from 30 to 15
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lambda_temporal = trial.suggest_float("lambda_temporal", 0.0, 1.0)
    use_retain_graph = trial.suggest_categorical("retain_graph", [True, False])
    # in_feats = 16  # fixed from feature construction logic
    hidden_feats = trial.suggest_int(
        "hidden_feats", 16, 176
    )  # increased from 128 to 160
    out_feats = trial.suggest_int("out_feats", 8, 40)  # reduced from 64 to 32)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    num_layers = trial.suggest_int("num_layers", 2, 4)  # reduced from 5 to 4
    possible_aggs = ["mean", "pool", "gcn"]  # , "lstm"]
    agg_list = [
        trial.suggest_categorical(f"agg_layer_{i}", possible_aggs)
        for i in range(num_layers)
    ]
    gru_num_layers = trial.suggest_int("gru_num_layers", 1, 3)
    use_relu = trial.suggest_categorical("use_relu", [False, True])

    use_random_noise = trial.suggest_categorical("use_random_noise", [True, False])
    node_type_encoding = trial.suggest_categorical(
        "node_type_encoding", ["none", "onehot"]  # , "embedding"]
    )
    node_type_dim = 0
    if node_type_encoding == "embedding":
        node_type_dim = trial.suggest_int("node_type_dim", 2, 8)

    print("=" * 60, sep="\n")
    print(f"Trial {trial.number} Hyperparameters:")
    print(f"num_epochs: {num_epochs}")
    print(f"lr: {lr}")
    print(f"lambda_temporal: {lambda_temporal}")
    print(f"use_retain_graph: {use_retain_graph}")
    print(f"hidden_feats: {hidden_feats}")
    print(f"out_feats: {out_feats}")
    print(f"dropout: {dropout}")
    print(f"num_layers: {num_layers}")
    print(f"agg_list: {agg_list}")
    print(f"gru_num_layers: {gru_num_layers}")
    print(f"use_relu: {use_relu}")
    print(f"use_random_noise: {use_random_noise}")
    print(f"node_type_encoding: {node_type_encoding}")
    print(f"node_type_dim: {node_type_dim}")
    print("")

    embedding_store = None

    model, embeddings = optuna_train_TemporalSAGE_pure_vdb(
        df=df,
        trial=trial,
        num_epochs=num_epochs,
        lambda_temporal=lambda_temporal,
        use_retain_graph=use_retain_graph,
        f_debug=f_debug,
        embedding_store=embedding_store,
        return_embeddings=True,
        # in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        use_random_noise=use_random_noise,
        node_type_encoding=node_type_encoding,
        node_type_dim=node_type_dim,
        lr=lr,
        dropout=dropout,
        num_layers=num_layers,
        agg_list=agg_list,
        gru_num_layers=gru_num_layers,
        use_relu=use_relu,
    )

    try:
        country_embeddings = embeddings[:num_countries]
        country_np = country_embeddings.numpy()

        reg = LinearRegression()
        reg.fit(country_np, y)
        # Here the score is r-squared, the coefficient of determination of prediction
        score = reg.score(country_np, y)

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        torch.cuda.empty_cache()  # safe on CPU too
        gc.collect()

        return score
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise optuna.exceptions.TrialPruned()


# Version 03:
def objective_corr(
    trial,
    df,
    y,
    num_countries=144,
    f_debug=False,
    f_iter_top_features=False,
    top_features=None,
    trial_name=None,
    **kwargs,
):
    # Hyperparameters
    if f_debug:
        num_epochs = trial.suggest_int("num_epochs", 2, 3)
    else:
        max_epochs = 15  # reduced from 30 to 15
        if "fixed_epochs" in kwargs:
            fe = kwargs["fixed_epochs"]
            if fe:
                max_epochs = fe
        num_epochs = trial.suggest_int("num_epochs", 2, max_epochs)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lambda_temporal = trial.suggest_float("lambda_temporal", 0.0, 1.0)
    use_retain_graph = trial.suggest_categorical("retain_graph", [True, False])
    # in_feats = 16  # fixed from feature construction logic
    hidden_feats = trial.suggest_int(
        "hidden_feats", 16, 176
    )  # increased from 128 to 160
    out_feats = trial.suggest_int("out_feats", 8, 40)  # reduced from 64 to 32)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    num_layers = trial.suggest_int("num_layers", 2, 4)  # reduced from 5 to 4
    possible_aggs = ["mean", "pool", "gcn"]  # , "lstm"]
    agg_list = [
        trial.suggest_categorical(f"agg_layer_{i}", possible_aggs)
        for i in range(num_layers)
    ]
    gru_num_layers = trial.suggest_int("gru_num_layers", 1, 3)
    use_relu = trial.suggest_categorical("use_relu", [False, True])

    use_random_noise = trial.suggest_categorical("use_random_noise", [True, False])
    node_type_encoding = trial.suggest_categorical(
        "node_type_encoding", ["none", "onehot"]  # , "embedding"]
    )
    node_type_dim = 0
    if node_type_encoding == "embedding":
        node_type_dim = trial.suggest_int("node_type_dim", 2, 8)

    trial_name = trial_name if bool(trial_name) else "Trial"
    top_features = top_features if bool(top_features) else ["nd", "cd"]
    # # Later we can add a trial.suggest_categorical to select between [["nd"], ["nd", "cd"], ["cd"]]
    # if f_iter_top_features:

    print("=" * 60, sep="\n")
    print(f"Trial {trial.number} Hyperparameters:")
    print(f"num_epochs: {num_epochs}")
    print(f"lr: {lr}")
    print(f"lambda_temporal: {lambda_temporal}")
    print(f"use_retain_graph: {use_retain_graph}")
    print(f"hidden_feats: {hidden_feats}")
    print(f"out_feats: {out_feats}")
    print(f"dropout: {dropout}")
    print(f"num_layers: {num_layers}")
    print(f"agg_list: {agg_list}")
    print(f"gru_num_layers: {gru_num_layers}")
    print(f"use_relu: {use_relu}")
    print(f"use_random_noise: {use_random_noise}")
    print(f"node_type_encoding: {node_type_encoding}")
    print(f"node_type_dim: {node_type_dim}")
    print("")

    embedding_store = None

    model, embeddings = optuna_train_TemporalSAGE_pure_vdb(
        df=df,
        trial=trial,
        num_epochs=num_epochs,
        lambda_temporal=lambda_temporal,
        use_retain_graph=use_retain_graph,
        f_debug=f_debug,
        embedding_store=embedding_store,
        return_embeddings=True,
        # in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        use_random_noise=use_random_noise,
        node_type_encoding=node_type_encoding,
        node_type_dim=node_type_dim,
        lr=lr,
        dropout=dropout,
        num_layers=num_layers,
        agg_list=agg_list,
        gru_num_layers=gru_num_layers,
        use_relu=use_relu,
        top_features=top_features,
        trial_name=trial_name,
    )

    try:
        country_emb = embeddings[:num_countries].cpu()
        country_emb_np = country_emb.numpy()

        n_components = 2
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(country_emb_np)
        if n_components == 1:
            pc1 = reduced.squeeze()
        else:
            pc1 = reduced[:, 0]
        # Here the score is correlation coefficient
        corr, p_value = pearsonr(pc1, y)
        score = abs(corr)  # we want the absolute value of the correlation

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        torch.cuda.empty_cache()  # safe on CPU too
        gc.collect()

        return score
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise optuna.exceptions.TrialPruned()


# Version 04:
def objective_corr_sye(
    trial,
    df,
    y,
    num_countries=144,
    f_debug=False,
    f_iter_top_features=False,
    top_features=None,
    trial_name=None,
    **kwargs,
):
    # Hyperparameters
    if f_debug:
        num_epochs = trial.suggest_int("num_epochs", 2, 3)
    else:
        # max_epochs = 15  # reduced from 30 to 15
        # if "fixed_epochs" in kwargs:
        #     fe = kwargs["fixed_epochs"]
        #     if fe:
        #         max_epochs = fe
        # num_epochs = trial.suggest_int("num_epochs", 2, max_epochs)
        max_epochs = 50  # reduced from 30 to 15
        if "fixed_epochs" in kwargs:
            fe = kwargs["fixed_epochs"]
            if fe:
                max_epochs = fe
        num_epochs = trial.suggest_int("num_epochs", 15, max_epochs)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lambda_temporal = trial.suggest_float("lambda_temporal", 0.0, 1.0)
    use_retain_graph = trial.suggest_categorical("retain_graph", [True, False])
    # in_feats = 16  # fixed from feature construction logic
    hidden_feats = trial.suggest_int(
        "hidden_feats", 16, 176
    )  # increased from 128 to 160
    out_feats = trial.suggest_int("out_feats", 8, 40)  # reduced from 64 to 32)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    num_layers = trial.suggest_int("num_layers", 2, 4)  # reduced from 5 to 4
    possible_aggs = ["mean", "pool", "gcn"]  # , "lstm"]
    agg_list = [
        trial.suggest_categorical(f"agg_layer_{i}", possible_aggs)
        for i in range(num_layers)
    ]
    gru_num_layers = trial.suggest_int("gru_num_layers", 1, 3)
    use_relu = trial.suggest_categorical("use_relu", [False, True])

    use_random_noise = trial.suggest_categorical("use_random_noise", [True, False])
    node_type_encoding = trial.suggest_categorical(
        "node_type_encoding", ["none", "onehot"]  # , "embedding"]
    )
    node_type_dim = 0
    if node_type_encoding == "embedding":
        node_type_dim = trial.suggest_int("node_type_dim", 2, 8)

    trial_name = trial_name if bool(trial_name) else "Trial"
    top_features = top_features if bool(top_features) else ["nd", "cd"]
    # # Later we can add a trial.suggest_categorical to select between [["nd"], ["nd", "cd"], ["cd"]]
    # if f_iter_top_features:

    print("=" * 60, sep="\n")
    print(f"Trial {trial.number} Hyperparameters:")
    print(f"num_epochs: {num_epochs}")
    print(f"lr: {lr}")
    print(f"lambda_temporal: {lambda_temporal}")
    print(f"use_retain_graph: {use_retain_graph}")
    print(f"hidden_feats: {hidden_feats}")
    print(f"out_feats: {out_feats}")
    print(f"dropout: {dropout}")
    print(f"num_layers: {num_layers}")
    print(f"agg_list: {agg_list}")
    print(f"gru_num_layers: {gru_num_layers}")
    print(f"use_relu: {use_relu}")
    print(f"use_random_noise: {use_random_noise}")
    print(f"node_type_encoding: {node_type_encoding}")
    print(f"node_type_dim: {node_type_dim}")
    print("")

    embedding_store = None

    model, embeddings = optuna_train_TemporalSAGE_pure_vdb_sye(
        df=df,
        trial=trial,
        num_epochs=num_epochs,
        lambda_temporal=lambda_temporal,
        use_retain_graph=use_retain_graph,
        f_debug=f_debug,
        embedding_store=embedding_store,
        return_embeddings=True,
        # in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        use_random_noise=use_random_noise,
        node_type_encoding=node_type_encoding,
        node_type_dim=node_type_dim,
        lr=lr,
        dropout=dropout,
        num_layers=num_layers,
        agg_list=agg_list,
        gru_num_layers=gru_num_layers,
        use_relu=use_relu,
        top_features=top_features,
        trial_name=trial_name,
    )

    try:
        country_emb = embeddings[:num_countries].cpu()
        country_emb_np = country_emb.numpy()

        n_components = 2
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(country_emb_np)
        if n_components == 1:
            pc1 = reduced.squeeze()
        else:
            pc1 = reduced[:, 0]
        # Here the score is correlation coefficient
        corr, p_value = pearsonr(pc1, y)
        score = abs(corr)  # we want the absolute value of the correlation

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        torch.cuda.empty_cache()  # safe on CPU too
        gc.collect()

        return score
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise optuna.exceptions.TrialPruned()
    

# Version 05:
def objective_corr_sye_m_loss(
    trial,
    df,
    y,
    num_countries=144,
    f_debug=False,
    f_iter_top_features=False,
    top_features=None,
    trial_name=None,
    **kwargs,
):
    # Hyperparameters
    if f_debug:
        num_epochs = trial.suggest_int("num_epochs", 2, 3)
    else:
        # max_epochs = 15  # reduced from 30 to 15
        # if "fixed_epochs" in kwargs:
        #     fe = kwargs["fixed_epochs"]
        #     if fe:
        #         max_epochs = fe
        # num_epochs = trial.suggest_int("num_epochs", 2, max_epochs)
        max_epochs = 50  # reduced from 30 to 15
        if "fixed_epochs" in kwargs:
            fe = kwargs["fixed_epochs"]
            if fe:
                max_epochs = fe
        num_epochs = trial.suggest_int("num_epochs", 15, max_epochs)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lambda_temporal = trial.suggest_float("lambda_temporal", 0.0, 1.0)
    use_retain_graph = trial.suggest_categorical("retain_graph", [True, False])
    # in_feats = 16  # fixed from feature construction logic
    hidden_feats = trial.suggest_int(
        "hidden_feats", 16, 176
    )  # increased from 128 to 160
    out_feats = trial.suggest_int("out_feats", 8, 40)  # reduced from 64 to 32)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    num_layers = trial.suggest_int("num_layers", 2, 4)  # reduced from 5 to 4
    possible_aggs = ["mean", "pool", "gcn"]  # , "lstm"]
    agg_list = [
        trial.suggest_categorical(f"agg_layer_{i}", possible_aggs)
        for i in range(num_layers)
    ]
    gru_num_layers = trial.suggest_int("gru_num_layers", 1, 3)
    use_relu = trial.suggest_categorical("use_relu", [False, True])

    use_random_noise = trial.suggest_categorical("use_random_noise", [True, False])
    node_type_encoding = trial.suggest_categorical(
        "node_type_encoding", ["none", "onehot"]  # , "embedding"]
    )
    node_type_dim = 0
    if node_type_encoding == "embedding":
        node_type_dim = trial.suggest_int("node_type_dim", 2, 8)

    trial_name = trial_name if bool(trial_name) else "Trial"
    top_features = top_features if bool(top_features) else ["nd", "cd"]
    # # Later we can add a trial.suggest_categorical to select between [["nd"], ["nd", "cd"], ["cd"]]
    # if f_iter_top_features:

    print("=" * 60, sep="\n")
    print(f"Trial {trial.number} Hyperparameters:")
    print(f"num_epochs: {num_epochs}")
    print(f"lr: {lr}")
    print(f"lambda_temporal: {lambda_temporal}")
    print(f"use_retain_graph: {use_retain_graph}")
    print(f"hidden_feats: {hidden_feats}")
    print(f"out_feats: {out_feats}")
    print(f"dropout: {dropout}")
    print(f"num_layers: {num_layers}")
    print(f"agg_list: {agg_list}")
    print(f"gru_num_layers: {gru_num_layers}")
    print(f"use_relu: {use_relu}")
    print(f"use_random_noise: {use_random_noise}")
    print(f"node_type_encoding: {node_type_encoding}")
    print(f"node_type_dim: {node_type_dim}")
    print("")

    embedding_store = None

    model, embeddings = optuna_train_TemporalSAGE_pure_vdb_sye_m_loss(
        df=df,
        trial=trial,
        num_epochs=num_epochs,
        lambda_temporal=lambda_temporal,
        use_retain_graph=use_retain_graph,
        f_debug=f_debug,
        embedding_store=embedding_store,
        return_embeddings=True,
        # in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        use_random_noise=use_random_noise,
        node_type_encoding=node_type_encoding,
        node_type_dim=node_type_dim,
        lr=lr,
        dropout=dropout,
        num_layers=num_layers,
        agg_list=agg_list,
        gru_num_layers=gru_num_layers,
        use_relu=use_relu,
        top_features=top_features,
        trial_name=trial_name,
    )

    try:
        country_emb = embeddings[:num_countries].cpu()
        country_emb_np = country_emb.numpy()

        n_components = 2
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(country_emb_np)
        if n_components == 1:
            pc1 = reduced.squeeze()
        else:
            pc1 = reduced[:, 0]
        # Here the score is correlation coefficient
        corr, p_value = pearsonr(pc1, y)
        score = abs(corr)  # we want the absolute value of the correlation

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        torch.cuda.empty_cache()  # safe on CPU too
        gc.collect()

        return score
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise optuna.exceptions.TrialPruned()