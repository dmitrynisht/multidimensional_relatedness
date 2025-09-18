import pandas as pd
import numpy as np
import torch
import dgl
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import random


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


def get_optuna_storage_uri(
    study_name=None,
    os_path=None,
    optuna_study_location=None,
):
    """
    Get the SQLite storage URI for Optuna.
    """
    db_name = ".".join([study_name, "db"])
    print(f"Optuna db name: {db_name}")

    # Full path to the SQLite file (e.g., study name = "graphsage_v1")
    db_path = os_path.join(optuna_study_location, db_name)
    # Normalize the path for SQLite URI format
    db_path = os_path.abspath(db_path).replace("\\", "/")

    # Convert to SQLite URI format
    storage_uri = f"sqlite:///{db_path}"
    print(f"Optuna storage location:\n\t{storage_uri}")

    return storage_uri


def get_optuna_study_name(
    prefix="sage-mor",
    top_features=None,
    target_score=None,
    year=None,
    f_init=None,
    version=None,
    **kwargs,
):
    """
    Generate a study name based on the provided parameters.
    Args:
        prefix (str): The prefix for the study name.
        top_features (list): List of top features.
        target_score (str): The target score.
        year (int): The year for the study.
        f_init (bool): Flag indicating if it's an initial study.
        version (str): The version of the study.
    Returns:
        str: The generated study name.
    """
    if f_init:
        postfix = "init"
    else:
        postfix = f"train-v{version}"

    if "fixed_epochs" in kwargs:
        fe = kwargs["fixed_epochs"]
        if fe:
            prefix = f"{prefix}-fe_{fe}"

    study_name = f"{prefix}-{'-'.join(top_features)}-t_{target_score}-{year}-{postfix}"

    return study_name


def build_ordered_node_index(df):
    """
    Build an ordered node index for countries and products.
    """
    # Get ordered node IDs for countries and products
    country_nodes = df["country_id"].sort_values().unique()
    product_nodes = df["product_id"].sort_values().unique()
    all_nodes = pd.Index(np.concatenate([country_nodes, product_nodes]))
    node_ids, node_labels = pd.factorize(all_nodes)

    return node_ids, node_labels


def get_oredered_node_ids(df, num_countries=144):
    """
    Get ordered node IDs for countries and products.
    """
    node_ids, node_labels = build_ordered_node_index(df)
    ordered_country_ids = node_labels[:num_countries].to_numpy()
    ordered_product_ids = node_labels[num_countries:].to_numpy()

    return ordered_country_ids, ordered_product_ids


def build_ordered_node_index_v2(df):
    """
    Build an ordered node index for countries and products.
    Disambiguing country vs. product IDs
    """
    assert (
        "node_country_id" in df.columns
    ), "DataFrame must contain 'node_country_id' column"
    assert (
        "node_product_id" in df.columns
    ), "DataFrame must contain 'node_product_id' column"

    # Get ordered node IDs for countries and products, maintaining the order of integer IDs
    country_nodes = df.loc[
        df["country_id"].sort_values().index, "node_country_id"
    ].unique()
    product_nodes = df.loc[
        df["product_id"].sort_values().index, "node_product_id"
    ].unique()

    all_nodes = pd.Index(np.concatenate([country_nodes, product_nodes]))
    node_ids, node_labels = pd.factorize(all_nodes)

    return node_ids, node_labels


def encode_product_hierarchy(
    df_year: pd.DataFrame,
    node_labels: pd.Index,
    hierarchy_column: str = "product_id_hierarchy",
    hierarchy_levels: int = 2,
    normalize: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Extract and encode product hierarchy levels as node features for product nodes.

    Args:
        df_year (pd.DataFrame): One-year slice of the main dataframe.
        node_labels (pd.Index): Full list of node labels (countries + products) to align features with.
        hierarchy_column (str): Column containing hierarchical product codes in format like "8.194.1875.10008".
        hierarchy_levels (int): Number of hierarchy levels to extract from the hierarchy code.
        normalize (bool): Whether to apply MinMax normalization to hierarchy levels.
        device (str): Target device for the resulting tensor ("cpu" or "cuda").

    Returns:
        torch.Tensor: (num_nodes x num_features) tensor, with product features aligned by node,
                      and zero-padded for non-product nodes.
    """
    assert (
        hierarchy_column in df_year.columns
    ), f"Missing required column: {hierarchy_column}"

    # Data preparation. Base product metadata columns
    # prod_cols = ["product_code", "product_level", "top_parent_id"]
    # prod_cols = ["product_code", "top_parent_id"]
    # "product_code" is a string, requires preprocessing
    # "product_level" didn't have meaningful values
    prod_cols = ["top_parent_id"]
    required_cols = ["product_id", hierarchy_column] + prod_cols
    hierarchy_df = df_year[required_cols].drop_duplicates()

    # Clean hierarchy path: strip final segment (actual product_id)
    clean_hierarchy_col = f"{hierarchy_column}_clean"
    hierarchy_df[clean_hierarchy_col] = (
        hierarchy_df[hierarchy_column].str.rsplit(".", n=1).str[0]
    )

    # Parse hierarchy levels into separate columns
    levels_df = (
        hierarchy_df[clean_hierarchy_col]
        .str.split(".", expand=True)
        .iloc[:, :hierarchy_levels]
        .fillna("0")
        .astype(int)
    )
    level_cols = [f"level_{i}" for i in range(hierarchy_levels)]

    if normalize:
        # Here we expect that each level of X.XXX.YYYY represents increasing specificity
        scaler = MinMaxScaler()
        levels_df = pd.DataFrame(
            scaler.fit_transform(levels_df),
            columns=level_cols,
            index=hierarchy_df.index,
        )

    # Combine hierarchy + external product attributes
    feature_df = pd.concat(
        [hierarchy_df[["product_id"]], levels_df, hierarchy_df[prod_cols]],
        axis=1,
    )
    feature_df.columns = ["product_id"] + level_cols + prod_cols
    feature_df = feature_df.drop_duplicates("product_id").set_index("product_id")

    # Build aligned feature matrix
    feat_num = len(feature_df.columns)
    external_attrs = []
    for node in node_labels:
        if node in feature_df.index:
            external_attrs.append(feature_df.loc[node].values.tolist())
        else:
            external_attrs.append([0.0] * feat_num)

    hierarchy_tensor = torch.tensor(external_attrs, dtype=torch.float32, device=device)

    return hierarchy_tensor


# Implement get_node_type_tensor utility function
def get_node_type_tensor(graph):
    """
    Assign node types for bipartite graph:
    Assumes countries are first N nodes and products are the rest.
    Returns tensor of 0s (country) and 1s (product) of length = num_nodes.
    """
    # num_nodes = graph.num_nodes()
    # num_edges = graph.num_edges()
    in_degrees = graph.in_degrees()
    out_degrees = graph.out_degrees()

    country_nodes = torch.where(
        out_degrees > in_degrees, torch.tensor(1.0), torch.tensor(0.0)
    )
    # Heuristic: country nodes tend to have more outgoing edges (exporting)

    node_types = country_nodes.to(torch.int64)
    return node_types


def generate_exports_bipartite_graph_v3(exports_df=None, node_mapping=None):
    """
    Generate a (mono) directed bipartite graph from the exports data frame.
    """
    # Check for required inputs
    if exports_df is None:
        raise ValueError("exports_df must be provided")

    if node_mapping is None:
        raise ValueError("node_mapping must be provided")

    # Use the global node mapping to encode edges
    country_encoded = exports_df["node_country_id"].map(node_mapping).values
    product_encoded = exports_df["node_product_id"].map(node_mapping).values

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


def generate_exports_bipartite_graph_v4(exports_df=None, node_mapping=None):
    """
    Generate a bidirected bipartite graph from the exports data frame.
    To allow countries to receive topological signals from their exported products, we construct a bidirected version of the export graph. This allows graph-based methods to assign meaningful embeddings to both node types.
    """
    # Check for required inputs
    if exports_df is None:
        raise ValueError("exports_df must be provided")

    if node_mapping is None:
        raise ValueError("node_mapping must be provided")

    # Use the global node mapping to encode edges
    country_encoded = exports_df["node_country_id"].map(node_mapping).values
    product_encoded = exports_df["node_product_id"].map(node_mapping).values

    # Forward: country → product
    src = torch.tensor(country_encoded, dtype=torch.int64)
    dst = torch.tensor(product_encoded, dtype=torch.int64)
    weight = torch.tensor(
        np.log1p(exports_df["average_export_value"].values),
        dtype=torch.float32,
    )

    # Reverse: product → country, with same weights (or optionally uniform)
    rev_src = dst
    rev_dst = src
    rev_weight = weight  # or: torch.ones_like(weight)

    # Combine edges
    all_src = torch.cat([src, rev_src])
    all_dst = torch.cat([dst, rev_dst])
    all_weight = torch.cat([weight, rev_weight])

    # Build graph
    graph = dgl.graph((all_src, all_dst), num_nodes=len(node_mapping))
    graph.edata["weight"] = all_weight

    # Assign node types
    node_type = torch.full((len(node_mapping),), fill_value=-1, dtype=torch.int64)
    node_type[src.unique()] = 0  # country
    node_type[dst.unique()] = 1  # product
    graph.ndata["node_type"] = node_type

    return graph


# Implement get_node_type_tensor utility function
def get_node_type_tensor(graph):
    """
    Assign node types for bipartite graph:
    Assumes countries are first N nodes and products are the rest.
    Returns tensor of 0s (country) and 1s (product) of length = num_nodes.
    """
    # num_nodes = graph.num_nodes()
    # num_edges = graph.num_edges()
    in_degrees = graph.in_degrees()
    out_degrees = graph.out_degrees()

    country_nodes = torch.where(
        out_degrees > in_degrees, torch.tensor(1.0), torch.tensor(0.0)
    )
    # Heuristic: country nodes tend to have more outgoing edges (exporting)

    node_types = country_nodes.to(torch.int64)
    return node_types


def closeness_per_node_type(
    g: dgl.DGLGraph, 
    node_types=None,
    f_debug=False
):
    """
    Computes closeness centrality per node in a bipartite graph,
    normalized using Borgatti & Everett's nonlinear denominator.

    Args:
        graph (dgl.DGLGraph): A DGL bidirected bipartite graph with `graph.ndata["node_type"]`.
        node_types (np.ndarray): Pre-extracted node types (0 = country, 1 = product).
        f_debug (bool): Whether to print debug information.

    Returns:
        torch.Tensor: Tensor of shape (N,) with closeness scores for all nodes.
    """
    # Step 1: Convert the DGL graph to an undirected NetworkX graph while retaining edge weights
    # We use undirected because closeness (even in weighted graphs) assumes symmetric path lengths.
    nx_graph = dgl.to_networkx(g, edge_attrs=["weight"]).to_undirected()

    # Step 2: Extract node types (0 = country, 1 = product) from DGL
    # Used to ensure we only compute closeness to *opposite* type
    if node_types is None:
        node_types = g.ndata["node_type"].cpu().numpy()
    num_nodes = g.num_nodes()

    # Step 3: Group nodes by their type for later use
    nodes_by_type = defaultdict(list)
    for nid, t in enumerate(node_types):
        nodes_by_type[t].append(nid)

    # Step 4: Allocate output tensor to hold closeness scores for each node
    closeness_scores = torch.zeros(num_nodes)

    # Step 5: Iterate through all nodes in the graph
    for i in range(num_nodes):
        this_type = node_types[i]
        other_type = 1 - this_type
        other_nodes = nodes_by_type[other_type]

        # Compute weighted shortest-paths lengths from node i to all reachable nodes
        # lengths = nx.single_source_shortest_path_length(nx_graph, i)
        lengths = nx.single_source_dijkstra_path_length(nx_graph, i, weight="weight")

        # Only consider distances to reachable opposite-type nodes
        reachable = [j for j in other_nodes if j in lengths]
        total_dist = sum(lengths[j] for j in reachable)

        # Compute closeness only if valid distances exist and total distance is positive
        if len(reachable) > 0 and total_dist > 0:
            avg_dist = total_dist / len(reachable)
            closeness_raw = 1.0 / avg_dist  # inverse of average distance
        else:
            closeness_raw = 0.0  # either no reachable nodes or undefined

        # Step 6: Apply Borgatti & Everett's nonlinear normalization
        nt = len(nodes_by_type[this_type])       # nodes of the same type
        no = len(nodes_by_type[other_type])      # nodes of the opposite type
        denom = nt + 2 * no - 2                  # theoretical max distance sum
        if f_debug:
            print(f"{i} weighted raw avg_dist:, {avg_dist}, closeness_raw:, {closeness_raw}, denom:, {denom}")

        closeness_scores[i] = closeness_raw / denom if denom > 0 else 0.0

    if f_debug:
        print("Closeness (per node type) computed successfully.")

    return closeness_scores


def bipartite_constraint(
    g: dgl.DGLGraph,
    node_types=None,
    f_debug=False,
) -> torch.Tensor:
    """
    Compute the bipartite constraint score for each node in a bipartite DGL graph.
    The score reflects how structurally redundant a node's neighborhood is, based
    on Burt's structural constraint concept adapted to bipartite graphs.

    Args:
        g (dgl.DGLGraph): The input bipartite graph (e.g., country-product graph),
                          assumed undirected or treated as undirected.
        node_types (Optional[np.ndarray or torch.Tensor]): A 1D array of node types (0 or 1),
                          matching DGL node IDs. If None, will attempt to read from g.ndata["node_type"].
        f_debug (bool): If True, prints diagnostic info.

    Returns:
        torch.Tensor: A (num_nodes, 1) tensor containing constraint scores per node.
                      Higher values indicate more redundancy (less structural holes).
    """
    # Convert graph to undirected NetworkX graph
    # We use undirected since structural redundancy is symmetric here
    nx_graph = g.to_networkx().to_undirected()
    num_nodes = g.num_nodes()

    # # Get node types if not passed explicitly
    # if node_types is None:
    #     node_types = g.ndata["node_type"].cpu().numpy()
    # elif isinstance(node_types, torch.Tensor):
    #     node_types = node_types.cpu().numpy()

    # Initialize constraint tensor
    constraint_tensor = torch.zeros(num_nodes)

    # Iterate over each node in the graph
    # This node will be treated as the "ego" in constraint terminology
    for ego in range(num_nodes):
        # Identify all direct neighbors of ego (these are its "alters")
        alters = list(nx_graph.neighbors(ego))
        if not alters:
            continue  # skip if ego has no neighbors

        # Assume uniform weights for now: each neighbor gets equal "investment"
        total_weight = len(alters)  # uniform weight

        # For each neighbor j (alter), compute the dyadic constraint c_ij
        for j in alters:
            pij = 1.0 / total_weight  # direct investment from ego to j

            # Compute indirect dependence through other alters
            # i.e., how much j is connected via common neighbors q ≠ j
            indirect_term = 0.0
            for q in alters:
                if q == ego:
                    continue  # skip self
                piq = 1.0 / total_weight  # direct investment to q
                q_neighbors = list(nx_graph.neighbors(q))
                if not q_neighbors:
                    continue  # skip q if it has no neighbors
                # contribution if j is also neighbor of q
                pqj = 1.0 / len(q_neighbors) if j in q_neighbors and len(q_neighbors) > 0 else 0.0
                indirect_term += piq * pqj

            # Compute dyadic constraint and add to ego's total
            cij = (pij + indirect_term) ** 2
            constraint_tensor[ego] += cij

        if f_debug:
            print(f"[debug] Node {ego} | alters = {alters} | constraint = {constraint_tensor[ego]:.4f}")

    return constraint_tensor


def weighted_one_step_hits(
    g: dgl.DGLGraph, 
    norm="l1", 
    f_debug=False
):
    """
    Compute one-step hub scores on a directed bipartite graph.

    This simplified HITS-like variant assumes a one-way edge structure
    (e.g., country → product). It uses outgoing weighted edge sums to
    calculate hub scores (e.g., export influence of countries), and
    skips the computation of authority scores since no reverse edges exist.

    Parameters:
        g (dgl.DGLGraph): Directed graph with g.edata["weight"]
        norm (str): Normalization type ('l1' or 'l2')
        f_debug (bool): Print debug information
    Returns:
        h: tensor of hub scores (countries)
        a: tensor of authority scores (products)
    """
    device = g.device
    num_nodes = g.num_nodes()

    # Allocate dummy node features (value=1) to enable directional message passing
    g.ndata["a_dummy"] = torch.ones(num_nodes, device=device)
    print("Node features:", g.ndata.keys())

    # Compute hub scores:
    #   For each node i, sum over weighted outgoing edges:
    #       h[i] = sum_j weight(i → j)
    g.update_all(
        dgl.function.v_mul_e("a_dummy", "weight", "m"), dgl.function.sum("m", "h")
    )

    h = g.ndata.pop("h")
    g.ndata.pop("a_dummy")

    # Optional normalization
    eps = 1e-10
    if norm == "l1":
        h /= h.sum() + eps
    elif norm == "l2":
        h /= torch.linalg.norm(h) + eps

    return h  # , a


def weighted_hits(g: dgl.DGLGraph, max_iter=100, tol=1e-8, norm="l1", f_debug=False):
    """
    Custom weighted HITS algorithm for directed DGL graph.

    Parameters:
        g (dgl.DGLGraph): Directed graph with g.edata["weight"]
        max_iter (int): Max number of iterations
        tol (float): Convergence threshold
        norm (str): Normalization type ('l1' or 'l2')
        verbose (bool): Print intermediate diagnostics

    Returns:
        hub_scores (torch.Tensor): (N,) tensor of hub scores
        auth_scores (torch.Tensor): (N,) tensor of authority scores
    """
    device = g.device
    num_nodes = g.num_nodes()

    # Initialize hub and authority scores
    h = torch.ones(num_nodes, device=device)
    a = torch.ones(num_nodes, device=device)
    if f_debug:
        print(f"Initial hub scores: {h}")
        print(f"Initial authority scores: {a}")

    for iter in range(max_iter):
        h_old = h.clone()
        a_old = a.clone()
        if f_debug:
            print(f"Iter {iter+1:03d}:")

        # Authority: a_j = sum_{i→j} w_{ij} * h_i
        g.ndata["h"] = h
        g.update_all(
            dgl.function.u_mul_e("h", "weight", "m"), dgl.function.sum("m", "a")
        )
        a = g.ndata.pop("a")
        if f_debug:
            print(f"Authority scores: {a}")
            print(f"[min={a.min():.4e}, max={a.max():.4e}")

        # Hub: h_i = sum_{i→j} w_{ij} * a_j
        g.ndata["a"] = a
        g.update_all(
            dgl.function.v_mul_e("a", "weight", "m"), dgl.function.sum("m", "h")
        )
        h = g.ndata.pop("h")
        if f_debug:
            print(f"Hub scores: {h}")
            print(f"[min={h.min():.4e}, max={h.max():.4e}")
            print("------before normalization------")
            print(
                f"[Iter {iter+1}] sum(h): {h.sum().item():.4f}, sum(a): {a.sum().item():.4f}"
            )

        # Normalize
        eps = 1e-10
        if norm == "l1":
            h /= h.sum() + eps
            a /= a.sum() + eps
        elif norm == "l2":
            h /= h.norm() + eps
            a /= a.norm() + eps

        # Check convergence
        delta = (h - h_old).abs().sum() + (a - a_old).abs().sum()
        if f_debug:
            print(f"Iter {iter+1:03d}, Δ = {delta:.4e}")

        if delta < tol:
            break

    if f_debug:
        print(f"Converged after {iter+1} iterations")
        print(f"Final hub scores: {h}")
        print(f"Final authority scores: {a}")

    return h, a


def dgl_graph_test(graph: dgl.DGLGraph, f_debug: bool = False):
    """"""
    src, dst = graph.edges()
    edges = set(zip(src.tolist(), dst.tolist()))
    reversed_edges = set(zip(dst.tolist(), src.tolist()))

    assert all(src[i] < dst[i] for i in range(len(src)))

    is_symmetric = edges == reversed_edges
    print("Graph is symmetric:", is_symmetric)
