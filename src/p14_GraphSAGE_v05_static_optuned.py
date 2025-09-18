from importlib import reload
from collections import defaultdict
import networkx as nx
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import optuna
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import p12_Utils
reload(p12_Utils)
from p12_Utils import (
    build_ordered_node_index_v2, 
    get_oredered_node_ids,
    encode_product_hierarchy,
    get_node_type_tensor,
    weighted_hits,
    weighted_one_step_hits,
    closeness_per_node_type,
    bipartite_constraint,
)

import p14_GraphSAGE_v03_dynamic_chrdb
reload(p14_GraphSAGE_v03_dynamic_chrdb)
# TemporalGraphModel = p14_GraphSAGE_v03_dynamic_chrdb.TemporalGraphModel
generate_exports_bipartite_graph_v2 = (
    p14_GraphSAGE_v03_dynamic_chrdb.generate_exports_bipartite_graph_v2
)
generate_exports_bipartite_graph_v3 = (
    p12_Utils.generate_exports_bipartite_graph_v3
)
generate_exports_bipartite_graph_v4 = (
    p12_Utils.generate_exports_bipartite_graph_v4
)
from p14_GraphSAGE_v04_dynamic_optuned import prepare_data_v3, OptunaTemporalGraphModel
import p14_GraphSAGE_u_loss
reload(p14_GraphSAGE_u_loss)
GraphSAGEUnsupervisedLoss = (
    p14_GraphSAGE_u_loss.GraphSAGEUnsupervisedLoss
)

def optuna_train_sye_sage_pure_conv_u_loss(
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
    unsup_loss = GraphSAGEUnsupervisedLoss(num_negative=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    detach_past = not use_retain_graph
    embeddings = None
    best_loss = float("inf")
    best_embeddings = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, yearly_embeddings = model(graphs_per_year, features_per_year)
        loss = compute_static_consistency_loss_v1(
            yearly_embeddings=yearly_embeddings,
            unsup_loss=unsup_loss,
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


def optuna_train_sye_sage_pure_conv_u_loss_bi_ms(
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
        prepare_data = prepare_data_v4
        top_features = top_features if bool(top_features) else ["wosh"]
        assert False, "This version is not yet implemented for multi-year data"
    else:
        # Single-year model (MoR style)
        # adjust here to new arg name "top_features"; current implementation corresponds to the "nd-cd" value
        prepare_data = prepare_data_v4
        top_features = top_features if bool(top_features) else ["wosh"]

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
    unsup_loss = GraphSAGEUnsupervisedLoss(num_negative=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    detach_past = not use_retain_graph
    embeddings = None
    best_loss = float("inf")
    best_embeddings = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, yearly_embeddings = model(graphs_per_year, features_per_year)
        loss = compute_static_consistency_loss_v1(
            yearly_embeddings=yearly_embeddings,
            unsup_loss=unsup_loss,
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


def compute_static_consistency_loss_v1(
    yearly_embeddings,
    **kwargs,
):
    loss = torch.tensor(0.0, device=yearly_embeddings[0].device, requires_grad=True)
    
    if "output" in kwargs:
        embeddings = kwargs["output"]  # Shape: [num_nodes, dim]
    else:
        print("No output provided")
        return loss

    if "graphs_per_year" in kwargs:
        graphs_per_year = kwargs["graphs_per_year"]
        # Use the first (and only) graph
        graph = graphs_per_year[0]
    else:
        print("No graphs_per_year provided")
        return loss
    
    if "unsup_loss" in kwargs:
        unsup_loss = kwargs["unsup_loss"]
    else:
        print("No loss function provided")
        return loss

    # Positive edges
    src_pos, dst_pos = graph.edges()

    # Sample negatives: shape [B, num_negative]
    num_neg = unsup_loss.num_negative
    neg_dst = torch.randint(
        low=0,
        high=embeddings.shape[0],
        size=(src_pos.shape[0], num_neg),
        device=embeddings.device,
    )

    # Gather embeddings
    z_u = embeddings[src_pos]               # [B, D]
    z_v = embeddings[dst_pos]               # [B, D]
    z_neg = embeddings[neg_dst]             # [B, Q, D]

    # Compute loss
    loss = unsup_loss(z_u, z_v, z_neg)
    
    return loss


# Version 05:
def objective_corr_sye_u_loss(
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
    node_type_encoding = "onehot"  # fixed for now
    node_type_encoding = trial.suggest_categorical(
        "node_type_encoding", ["onehot"]  # "none", "embedding"]
    )
    node_type_dim = 0
    # if node_type_encoding == "embedding":
    #     node_type_dim = trial.suggest_int("node_type_dim", 2, 8)

    trial_name = trial_name if bool(trial_name) else "Trial"
    top_features = top_features if bool(top_features) else ["nd", "cd"]
    # # Later we can add a trial.suggest_categorical to select between [["nd"], ["nd", "cd"], ["cd"]]
    # if f_iter_top_features:
    if "learn_embeddings_func" in kwargs:
        learn_embeddings_func = kwargs["learn_embeddings_func"]
    else:
        learn_embeddings_func = optuna_train_sye_sage_pure_conv_u_loss

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
                        
    model, embeddings = learn_embeddings_func(
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
    

def get_graph_statistics(
    rolling_exports_years=None,
    n_years=None,
    x_cp=None,
    num_countries=0,
    f_debug=False,
    use_random_noise=False,
    node_type_encoding="none",
    top_features=None,
):
    # If top_features is None, default to ["cd"]
    top_features = top_features if bool(top_features) else ["cd"]
    prepare_data = prepare_data_v4

    columns=[]
    if "nd" in top_features:
        columns.extend(["nd_in", "nd_out", "nd_in_norm", "nd_out_norm"])
    if "cd" in top_features:
        columns.append("cd")
    if "pr" in top_features:
        columns.append("pr")
    if "ht" in top_features:
        columns.extend(["hub", "auth"])
    if "wosh" in top_features:
        columns.append("wosh")

    all_years_feature_rows = []

    for year in rolling_exports_years[:n_years]:
        # Filter export data for single year
        x_cp_single_year = x_cp[x_cp["year"] == year]

        # Generate graph and features for this year
        graphs_per_year, features_per_year = prepare_data(
            df=x_cp_single_year,
            f_debug=f_debug,
            use_random_noise=use_random_noise,
            node_type_encoding=node_type_encoding,
            top_features=top_features,
        )

        # Get country IDs in correct order
        ordered_node_ids, _ = build_ordered_node_index_v2(x_cp_single_year)
        if f_debug:
            print(f"Number of countries: {len(ordered_node_ids[:num_countries])}")
            print(f"Ordered country IDs: {ordered_node_ids[:num_countries]}")
            # assert False, "Debugging forced stop"
        
        # Convert tensor to NumPy
        features = features_per_year[0].cpu().numpy()
        country_features = features[:num_countries, :]  # country rows

        # Building per-year feature DataFrame
        year_df = pd.DataFrame(
            country_features,
            columns=columns,
        )
        year_df["country_id"] = ordered_node_ids[:num_countries]
        year_df["year"] = year

        all_years_feature_rows.append(year_df)

    # Combine all years into a single DataFrame
    graph_statistics_all_years = pd.concat(all_years_feature_rows, ignore_index=True)

    return graph_statistics_all_years


# This version is for single-year inputs, in MoR style
def prepare_data_v4_archived(
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
    Specifically, in this version, we don't support temporal approach.
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
            "wosh" - weighted one-step hub scores
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
    node_ids, node_labels = build_ordered_node_index_v2(df)
    node_lookup = dict(zip(node_labels, node_ids))
    if f_debug:
        print("node_ids", node_ids)
        print("node_labels", node_labels)
        print("node_lookup", node_lookup)
        # assert False, "Debugging forced stop"

    # if "ht" in top_features:
    #     for year in df["year"].unique():
    #         df_year = df[df["year"] == year]
    #         graph = generate_exports_bipartite_graph_v3(
    #             exports_df=df_year, node_mapping=node_lookup
    #         )
            
    #         # Convert to NetworkX for graph-level algorithms
    #         nx_graph = graph.to_networkx().to_directed()

    #         # Map NetworkX node order to DGL node index
    #         nx_to_dgl = {node: i for i, node in enumerate(graph.nodes().numpy())}

    #         # Test run of HITS, if it fails the whole process will fail, to avoid confusion
    #         hits_hubs, hits_auth = nx.hits(
    #             nx_graph, max_iter=1000, tol=1e-08, normalized=True
    #         )

    # if "cn" in top_features:
    #     for year in df["year"].unique():
    #         df_year = df[df["year"] == year]
    #         graph = generate_exports_bipartite_graph_v2(
    #             exports_df=df_year, node_mapping=node_lookup
    #         )

    #         # Convert to NetworkX for graph-level algorithms
    #         nx_graph = graph.to_networkx().to_directed()

    #         # Map NetworkX node order to DGL node index
    #         nx_to_dgl = {node: i for i, node in enumerate(graph.nodes().numpy())}

    #         # Test run of Core Number, if it fails the whole process will fail, to avoid confusion
    #         core_nums = nx.core_number(nx_graph)

    # In this version, we expect only one year of data at a time
    for year in df["year"].unique():
        df_year = df[df["year"] == year]
        graph = generate_exports_bipartite_graph_v4(
            exports_df=df_year, node_mapping=node_lookup
        )

        # in_deg = graph.in_degrees().float().unsqueeze(1)
        # out_deg = graph.out_degrees().float().unsqueeze(1)
        num_nodes = graph.num_nodes()
        # if f_debug:
        #     print(
        #         f"Year {year} → in_degrees.shape: {in_deg.shape}, in_degrees.shape[0]: {in_deg.shape[0]}, nodes: {num_nodes}",
        #         f"Year {year} → out_degrees.shape: {out_deg.shape}, out_degrees.shape[0]: {out_deg.shape[0]}, nodes: {num_nodes}",
        #         sep="\n",
        #     )

        # # Safety check: pad if needed
        # if in_deg.shape[0] < num_nodes:
        #     pad_len = num_nodes - in_deg.shape[0]
        #     in_deg = torch.cat([in_deg, torch.zeros(pad_len, dtype=in_deg.dtype)])

        # if out_deg.shape[0] < num_nodes:
        #     pad_len = num_nodes - out_deg.shape[0]
        #     out_deg = torch.cat([out_deg, torch.zeros(pad_len, dtype=out_deg.dtype)])

        # max_in = in_deg.max().item() if in_deg.max().item() > 0 else 1.0
        # max_out = out_deg.max().item() if out_deg.max().item() > 0 else 1.0

        # Base degree features
        base_feats = []

        # # Normalized degree features
        # if "nd" in top_features:
        #     nd_feats = [
        #         torch.log1p(in_deg),
        #         torch.log1p(out_deg),
        #         in_deg / max_in,
        #         out_deg / max_out,
        #     ]
        #     base_feats.extend(nd_feats)

        # # Optional: combined centrality degree feature
        # if "cd" in top_features:
        #     # Normalized by number of nodes in the graph
        #     degree_centrality = (in_deg + out_deg) / num_nodes
        #     base_feats.append(degree_centrality)

        # # Convert to NetworkX for graph-level algorithms
        # nx_graph = graph.to_networkx().to_directed()

        # # Map NetworkX node order to DGL node index
        # nx_to_dgl = {node: i for i, node in enumerate(graph.nodes().numpy())}

        # # Optional: PageRank (PR)
        # if "pr" in top_features:
        #     pr_scores = nx.pagerank(nx_graph)
        #     pr_tensor = torch.zeros(num_nodes)
        #     for nx_id, score in pr_scores.items():
        #         pr_tensor[nx_to_dgl[nx_id]] = score
        #     base_feats.append(pr_tensor.unsqueeze(1))  # (N, 1)

        # # Optional: HITS (hub & authority scores)
        # if "ht" in top_features:
        #     try:
        #         hits_hubs, hits_auth = weighted_hits(
        #             g=nx_graph, 
        #             max_iter=1000, 
        #             tol=1e-08, 
        #             norm="l1",
        #         )
        #         hub_tensor = torch.zeros(num_nodes)
        #         auth_tensor = torch.zeros(num_nodes)
        #         for nx_id in hits_hubs:
        #             hub_tensor[nx_to_dgl[nx_id]] = hits_hubs[nx_id]
        #             auth_tensor[nx_to_dgl[nx_id]] = hits_auth[nx_id]
        #         base_feats.append(hub_tensor.unsqueeze(1))
        #         base_feats.append(auth_tensor.unsqueeze(1))
        #     except nx.PowerIterationFailedConvergence:
        #         print(
        #             f"HITS failed to converge in year {year}, defaulting to skipping HITS(hubs & authorities)."
        #         )

        # Optional: weighted one-step hubs scores)
        # implementation directly on the DGL graph, skipping converting to NetworkX
        if "wosh" in top_features:
            try:
                wosh_scores = weighted_one_step_hits(
                    g=graph, 
                    norm="l1",
                    f_debug=f_debug,
                )
                wosh_tensor = torch.zeros(num_nodes)
                for dgl_id in range(num_nodes):
                    wosh_tensor[dgl_id] = wosh_scores[dgl_id]
                base_feats.append(wosh_tensor.unsqueeze(1))
            except Exception as e:
                print(f"Weighted one-step hubs scores failed for year {year}: {e}")            

        # # Optional: Core Number (K-core)
        # if "cn" in top_features:
        #     try:
        #         core_nums = nx.core_number(nx_graph)
        #         core_tensor = torch.zeros(num_nodes)
        #         for nx_id, core in core_nums.items():
        #             core_tensor[nx_to_dgl[nx_id]] = core
        #         base_feats.append(core_tensor.unsqueeze(1))
        #     except nx.NetworkXError as e:
        #         print(f"Core number failed for year {year}: {e}")

        # # Optional: Product hierarchy features
        # if "ph" in top_features:
        #     # Note: While the full product hierarchy format can go up to 6 levels (e.g., X.XXX.YYYY.ZZZZZ),
        #     # all products in the x_cp dataset follow the X.XXX.YYYY structure, corresponding to level 4.
        #     # Since x_cp["product_level"] is always 4, we only observe two meaningful hierarchy levels,
        #     # and will not include "product_level" as a feature in the graph.
        #     hierarchy_tensor = encode_product_hierarchy(
        #         df_year=df_year,
        #         node_labels=node_labels,
        #         hierarchy_column="product_id_hierarchy",
        #         hierarchy_levels=2,
        #         normalize=True,
        #         device=graph.device,
        #     )
        #     base_feats.append(hierarchy_tensor)

        # Optional: Random noise (helps with symmetry breaking)
        if use_random_noise:
            rand_dim = 14
            base_feats.append(torch.randn(num_nodes, rand_dim))

        # # Optional: Node type encoding
        # if node_type_encoding != "none":
        #     node_types = get_node_type_tensor(
        #         graph
        #     )  # must return 0 for country, 1 for product
        #     if node_type_encoding == "onehot":
        #         base_feats.append(F.one_hot(node_types, num_classes=2).float())
        #     elif node_type_encoding == "embedding":
        #         embed = nn.Embedding(2, node_type_dim).to(graph.device)
        #         base_feats.append(embed(node_types))

        features = torch.cat(base_feats, dim=1)
        graphs_per_year.append(graph)
        features_per_year.append(features)

        if f_debug:
            print(f"Year {year}: features shape = {features.shape}")

    return graphs_per_year, features_per_year


# This version is for single-year inputs, in MoR style
def prepare_data_v4(
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
    Specifically, in this version, we don't support temporal approach.
    Args:
        df (pd.DataFrame): Input dataframe with columns "country_id", "product_id", "year", "exports", "imports"
        f_debug (bool, optional): Whether to print debug information. Defaults to False.
        use_random_noise (bool, optional): Whether to use random noise in feature generation. Defaults to True.
        node_type_encoding (str, optional): Encoding for node types. Options are "none", "onehot", "embedding". Defaults to "none".
        node_type_dim (int, optional): Dimension of node type embeddings. Defaults to 0.
        top_features (list, optional): List of topological features to include. Defaults to None.
            Topological features implemented:
            "wosh" - weighted one-step hub scores
            "cpt" - closeness per type
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
    node_ids, node_labels = build_ordered_node_index_v2(df)
    node_lookup = dict(zip(node_labels, node_ids))
    if f_debug:
        print("node_ids", node_ids)
        print("node_labels", node_labels)
        print("node_lookup", node_lookup)
        # assert False, "Debugging forced stop"

    # In this version, we expect only one year of data at a time
    for year in df["year"].unique():
        df_year = df[df["year"] == year]
        graph = generate_exports_bipartite_graph_v4(
            exports_df=df_year, 
            node_mapping=node_lookup
        )

        # # Compute node types once (if not stored)
        # if "node_type" not in graph.ndata:
        #     graph.ndata["node_type"] = get_node_type_tensor(
        #         graph
        #     )

        # Extract node types (0 = country, 1 = product) from DGL
        # Used to ensure we only compute closeness to *opposite* type
        node_types = graph.ndata["node_type"].cpu().numpy()

        num_nodes = graph.num_nodes()

        # Base features
        base_feats = []

        # Weighted one-step hubs scores
        # implementation directly on the DGL graph, skipping converting to NetworkX
        if "wosh" in top_features:
            try:
                wosh_scores = weighted_one_step_hits(
                    g=graph, 
                    norm="l1",
                    f_debug=f_debug,
                )
                wosh_tensor = torch.zeros(num_nodes)
                for dgl_id in range(num_nodes):
                    wosh_tensor[dgl_id] = wosh_scores[dgl_id]
                base_feats.append(wosh_tensor.unsqueeze(1))
            except Exception as e:
                print(f"Weighted one-step hubs scores failed for year {year}: {e}")            
        
        # Closeness per node type (cpt)
        if "cpt" in top_features:
            try:
                # Add the closeness tensor to the feature list
                cpt_tensor = closeness_per_node_type(
                    g=graph,
                    node_types=node_types, 
                    f_debug=f_debug
                )
                base_feats.append(cpt_tensor.unsqueeze(1))  # shape: (N, 1)

            except Exception as e:
                print(f"Closeness per node type feature failed for year {year}: {e}")

        # Bipartite constraint (bcnt)
        if "bcnt" in top_features:
            try:
                bcnt_tensor = bipartite_constraint(
                    graph, 
                    f_debug=f_debug
                )
                # Append final tensor to base_feats list
                # Shape is (N, 1), consistent with other topological features
                base_feats.append(bcnt_tensor.unsqueeze(1))

            except Exception as e:
                print(f"Bipartite constraint feature failed for year {year}: {e}")

        # Optional: Node type encoding
        if node_type_encoding != "none":
            if node_type_encoding == "onehot":
                base_feats.append(F.one_hot(node_types, num_classes=2).float())
            elif node_type_encoding == "embedding":
                embed = nn.Embedding(2, node_type_dim).to(graph.device)
                base_feats.append(embed(node_types))

        # # Convert to NetworkX for graph-level algorithms
        # nx_graph = graph.to_networkx().to_directed()

        # # Map NetworkX node order to DGL node index
        # nx_to_dgl = {node: i for i, node in enumerate(graph.nodes().numpy())}

        # # Optional: BiRank (bir)
        # if "bir" in top_features:
        #   # placeholder for biRank implementation
        #     pr_scores = nx.pagerank(nx_graph)
        #     pr_tensor = torch.zeros(num_nodes)
        #     for nx_id, score in pr_scores.items():
        #         pr_tensor[nx_to_dgl[nx_id]] = score
        #     base_feats.append(pr_tensor.unsqueeze(1))  # (N, 1)

        # # Optional: Core Number (K-core)
        # if "cn" in top_features:
        #     try:
        #         core_nums = nx.core_number(nx_graph)
        #         core_tensor = torch.zeros(num_nodes)
        #         for nx_id, core in core_nums.items():
        #             core_tensor[nx_to_dgl[nx_id]] = core
        #         base_feats.append(core_tensor.unsqueeze(1))
        #     except nx.NetworkXError as e:
        #         print(f"Core number failed for year {year}: {e}")

        # Optional: Random noise (helps with symmetry breaking)
        if use_random_noise:
            rand_dim = 14
            base_feats.append(torch.randn(num_nodes, rand_dim))

        features = torch.cat(base_feats, dim=1)
        graphs_per_year.append(graph)
        features_per_year.append(features)

        if f_debug:
            print(f"Year {year}: features shape = {features.shape}")

    return graphs_per_year, features_per_year