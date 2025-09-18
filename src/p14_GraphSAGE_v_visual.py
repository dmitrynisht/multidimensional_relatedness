import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap
import hdbscan
import numpy as np
from scipy.stats import pearsonr
from p10_Chroma_dbm import ChromaEmbeddingStore
from p12_Utils import get_optuna_study_name
from p14_GraphSAGE_v04_dynamic_optuned import get_oredered_node_ids
from p14_GraphSAGE_v_evaluate import get_outlier_embeddings


def visualize_embeddings(embeddings, num_countries=None):
    """
    Visualize 2D PCA projection of country node embeddings only.

    Args:
        embeddings (Tensor): Output embeddings from the model (N_nodes x D)
        num_countries (int): Number of country nodes (assumed to be indexed first)
    """
    if num_countries is None:
        raise ValueError(
            "Please provide num_countries to identify which nodes are countries."
        )

    # Slice country embeddings only
    country_emb = embeddings[:num_countries]
    country_emb_np = country_emb.cpu().numpy()

    # Reduce dimensionality to 2D
    n_components = 2
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(country_emb_np)
    print(
        "Country embeddings reduced shape:",
        pd.DataFrame(reduced).drop_duplicates().shape,
    )
    # iterating over pca.explained_variance_ratio_ numpy array showing index and values of each element
    print(
        "Explained variance ratio:",
        [f"PC{i+1}: {v:.4f}" for i, v in enumerate(pca.explained_variance_ratio_)],
    )

    # Create DataFrame for Seaborn
    df_plot = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_plot["node_type"] = "country"

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_plot, x="PC1", y="PC2", hue="node_type", palette="tab10", legend=False
    )
    plt.title("Country Embeddings (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_embeddings_2PCA(
    embeddings,
    num_countries=None,
    std_multiplier=2.0,
    library="sns",
    random_state=None,
):
    """
    Visualize 2D PCA projection of country node embeddings only.

    Args:
        embeddings (Tensor): Output embeddings from the model (N_nodes x D)
        num_countries (int): Number of country nodes (assumed to be indexed first)
        std_multiplier (float): Multiplier for standard deviation to identify outliers (default: 2.0)
        library (str): Visualization library to use (default: "sns", alternative: "matplotlib")
    """
    if num_countries is None:
        raise ValueError(
            "Please provide num_countries to identify which nodes are countries."
        )

    # Slice country embeddings only
    country_emb = embeddings[:num_countries]
    country_emb_np = country_emb.cpu().numpy()

    # Reduce dimensionality to 2D
    n_components = 2
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    reduced = pca.fit_transform(country_emb_np)
    print(
        "Country embeddings reduced shape:",
        pd.DataFrame(reduced).drop_duplicates().shape,
    )
    # iterating over pca.explained_variance_ratio_ numpy array showing index and values of each element
    print(
        "Explained variance ratio:",
        [f"PC{i+1}: {v:.4f}" for i, v in enumerate(pca.explained_variance_ratio_)],
    )

    # Euclidean distance from the center (0,0)
    distances = np.linalg.norm(reduced, axis=1)

    # Outlier threshold
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + std_multiplier * std_dist

    # Identify outliers
    outlier_indices = np.where(distances > threshold)[0]

    # Plot
    plt.figure(figsize=(8, 6))
    if library == "sns":
        # Create DataFrame for Seaborn
        df_plot = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        df_plot["node_type"] = "country"

        sns.scatterplot(
            data=df_plot,
            x="PC1",
            y="PC2",
            hue="node_type",
            palette="tab10",
            legend=False,
        )
    elif library == "matplotlib":
        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            alpha=0.9,
            c="tab:blue",
            edgecolors="white",
            linewidth=0.5,
            label="Countries",
            s=30,
        )
        plt.scatter(
            reduced[outlier_indices, 0],
            reduced[outlier_indices, 1],
            color="red",
            edgecolors="white",
            linewidth=0.5,
            label="Outliers",
            s=30,
        )
    else:
        raise ValueError(
            f"Invalid library: {library}. Supported libraries: 'sns', 'matplotlib'"
        )

    # Draw a circle for the threshold
    circle = plt.Circle(
        (0, 0),
        threshold,
        color="green",
        fill=False,
        linestyle="--",
        linewidth=2,
        label=f"2 std dev threshold: {threshold:.2f}",
    )
    plt.gca().add_artist(circle)

    plt.title("Country Embeddings (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def lineplot_embedding_correlations_vs_targets(
    correlation_df: pd.DataFrame = None,
    target_columns: list = None,
    title="Embedding–Target Correlations Over Time",
    xlabel="Year",
    ylabel="Correlation",
    figsize=(10, 6),
):
    """
    Plots yearly embedding–target correlations from a long-form or wide-form correlation dataframe.

    Parameters:
        correlation_df (pd.DataFrame): DataFrame with a "year" column and multiple correlation columns.
        target_columns (list): List of column names to plot (e.g., ['emb_PC1_gdp_r', 'emb_PC1_hdi_r']).
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        figsize (tuple): Figure size.
    """
    # Melt the selected columns into long format
    melted_df = (
        correlation_df[["year"] + target_columns]
        .melt(id_vars="year", var_name="target", value_name="correlation")
        .dropna()
    )

    # Sort by year to ensure consistent line flow
    melted_df = melted_df.sort_values(by=["target", "year"])

    # Compute min/max
    y_min = melted_df["correlation"].min()
    # y_max = melted_df["correlation"].max()
    y_max = 1

    # Create the line plot
    plt.figure(figsize=figsize)
    sns.lineplot(data=melted_df, x="year", y="correlation", hue="target", marker="o")

    # Customize the appearance
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(y_min, y_max)

    # Legend styling
    plt.legend(
        title="Target Correlation",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()

    return y_min, y_max


def lineplot_embedding_correlations_faceted(
    correlation_df: pd.DataFrame = None,
    target_columns: list = None,
    benchmark_column: str = "eci_18__log_gdp_pcap",
    title="Embedding-Target Correlations vs MoR ECI (Faceted)",
    height=5,
    aspect=1.4,
    y_limits: tuple = None,
):
    """
    Plots Pearson and Spearman correlations of embeddings (PC1/PC2) with multiple targets,
    grouped by correlation type and principal component using faceted line plots.
    Also overlays MoR-style ECI benchmark on each subplot.

    Parameters:
        correlation_df (pd.DataFrame): Must include "year", embedding correlations, and the benchmark column.
        target_columns (list): Correlation columns to plot (e.g., ['emb_PC1_gdp_r', 'emb_PC1_gdp_s']).
        benchmark_column (str): Column name for MoR benchmark (e.g., 'eci_18__log_gdp_pcap').
        title (str): Overall title.
        height (float): Height of each subplot.
        aspect (float): Width/height ratio of each subplot.
    """

    # Melt embedding correlation columns
    melted_df = (
        correlation_df[["year"] + target_columns]
        .melt(id_vars="year", var_name="correlation_type", value_name="correlation")
        .dropna()
    )

    # Parse column names like 'emb_PC1_gdp_r'
    parsed = melted_df["correlation_type"].str.extract(r"emb_(PC\d)_(.+)_(r|s)")
    melted_df["PC"] = parsed[0]
    melted_df["target"] = parsed[1]
    melted_df["metric"] = parsed[2].map({"r": "Pearson", "s": "Spearman"})
    melted_df["target_label"] = (
        melted_df["target"]
        .str.replace("_", " ")  # e.g. 'log_gdp_per_capita' → 'log gdp per capita'
        .str.upper()  # → 'LOG GDP PER CAPITA'
    )

    # Benchmark line to replicate across subplots
    benchmark_df = correlation_df[["year", benchmark_column]].rename(
        columns={benchmark_column: "benchmark"}
    )

    # Merge to allow plotting
    melted_df = melted_df.merge(benchmark_df, on="year", how="left")

    # Plot using FacetGrid
    g = sns.FacetGrid(
        melted_df,
        col="metric",
        row="PC",
        hue="target_label",
        margin_titles=True,
        height=height,
        aspect=aspect,
        sharey=True,
    )

    g.map_dataframe(sns.lineplot, x="year", y="correlation", marker="o")

    # Overlay ECI benchmark as dashed line (repeat for each subplot)
    for ax, (pc, metric) in zip(g.axes.flat, melted_df.groupby(["PC", "metric"])):
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.plot(
            benchmark_df["year"],
            benchmark_df["benchmark"],
            linestyle="--",
            color="black",
            # label="MoR ECI benchmark",
            linewidth=1.5,
            zorder=1,  # Keeps it below dots
        )
        if y_limits:
            ax.set_ylim(*y_limits)  # <- apply fixed y-axis scale

    # Final touches
    g.set_axis_labels("Year", "Correlation")
    g.set_titles(row_template="{row_name}", col_template="{col_name} Correlation")

    # Manually get original handles
    handles, labels = g.axes[0][0].get_legend_handles_labels()
    # Add custom benchmark handle
    benchmark_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="MoR ECI benchmark",
    )
    handles.append(benchmark_handle)
    labels.append("ECI (MoR) vs log GDP (Pearson)")

    # Deduplicate labels
    seen = set()
    unique_handles_labels = [
        (h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))
    ]
    handles, labels = zip(*unique_handles_labels)

    # Add new unified legend
    g.add_legend(
        handles=handles,
        labels=labels,
        title="Correlation Target",
        # loc="upper right",
        bbox_to_anchor=(0.8, 0.9),
        loc="upper left",
        # loc="center left",
        # bbox_to_anchor=(0.73, 0.78),
        # # borderaxespad=0.0
    )

    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(title, fontsize=16, fontweight="bold")

    plt.show()


def visualize_country_embeddings_detect_outliers(
    rolling_exports_years=None,
    n_years=None,
    x_cp=None,
    gdp_df=None,
    num_countries=0,
    f_init=False,
    prefix="sage-mor",
    version=None,
    target_score=None,
    top_features=None,
    vectordb_location=None,
    location_df=None,
    **kwargs,
):
    f_deterministic = kwargs.get("f_deterministic", False)
    random_seed = 42 if f_deterministic else None
    year_seed = random_seed if f_deterministic else None
    
    outlier_embedding_countries = np.empty((0, 3))
    for year in rolling_exports_years[:n_years]:
        # print(f"\n{'=' * 60}\nTraining GNN-MoR for year {year}...")
        study_name = get_optuna_study_name(
            prefix=prefix,
            top_features=top_features,
            target_score=target_score,
            year=year,
            f_init=f_init,
            version=version,
            **kwargs,
        )
        print("Study name:", study_name)
        embeddings_single_year_collection = ChromaEmbeddingStore(
            db_name=study_name, db_path=vectordb_location
        )
        embeddings = embeddings_single_year_collection.load()

        # Filter export data and GDP for this year
        x_cp_single_year = x_cp[x_cp["year"] == year]
        y = gdp_df[gdp_df["year"] == year]["log_gdp_per_capita"].values

        # Visualize embeddings using PCA, and potential outliers
        year_seed = year_seed + 1 if f_deterministic else None
        visualize_embeddings_2PCA(
            embeddings=embeddings, 
            num_countries=num_countries, 
            library="matplotlib", 
            random_state=year_seed,
        )

        # Create global node ID mapping
        country_ids, product_ids = get_oredered_node_ids(
            x_cp_single_year, num_countries
        )

        # Select outlier embeddings
        single_year_outlier_countries, pca_2 = get_outlier_embeddings(
            embeddings=embeddings, num_countries=num_countries, country_ids=country_ids
        )
        print(single_year_outlier_countries.shape)
        outlier_embedding_countries = np.concatenate(
            (outlier_embedding_countries, single_year_outlier_countries)
        )

        # Pearson correlation
        corr, p_value = pearsonr(pca_2[:, 0], y)
        print(f"PCA1 Correlation with GDP:")
        print(f"   Pearson r = {corr:.4f}, p = {p_value:.4f}")

        # Show outlier countries
        print("\nOutlier countries:")
        for idx, id, dist in single_year_outlier_countries:
            # print(idx, id, dist, sep="\t")
            location = location_df[location_df["country_id"] == int(id)]
            print(
                f"Outlier {int(idx)}: Distance = {dist:.2f}, Country ID = {int(id)}: {location['country_name'].values[0]}, ({location['country_code'].values[0]})"
            )

        print("-" * 60, "\n\n")

    return outlier_embedding_countries


def visualize_embeddings_with_umap(
    rolling_exports_years=None,
    n_years=None,
    x_cp=None,
    sindices_df=None,
    num_countries=0,
    f_init=False,
    prefix="sage-mor",
    version=None,
    target_score=None,
    top_features=None,
    vectordb_location=None,
    color_by=None,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    random_state=42,
    **kwargs,
):
    """
    Projects embeddings into 2D using UMAP and visualizes them without clustering.

    Args:
        rolling_exports_years (list): List of years to train the model on
        n_years (int): Number of years to train the model on
        x_cp (pd.DataFrame): DataFrame containing all target indicators, with 'year' and target columns.
        sindices_df (pd.DataFrame): DataFrame containing all target indicators, with 'year' and target columns.
        color_by (str): Optional column in metadata to color points
        umap_n_neighbors (int): UMAP neighborhood size
        umap_min_dist (float): UMAP spacing control
        random_state (int): Reproducibility

    Returns:
        pd.DataFrame: UMAP projections with metadata (if provided)
    """
    # if hasattr(embeddings, "detach"):
    #     embeddings = embeddings.detach().cpu().numpy()
    records = []

    for year in rolling_exports_years[:n_years]:
        # print(f"\n{'=' * 60}\nTraining GNN-MoR for year {year}...")
        study_name = get_optuna_study_name(
            prefix=prefix,
            top_features=top_features,
            target_score=target_score,
            year=year,
            f_init=f_init,
            version=version,
            **kwargs,
        )
        print("Study name:", study_name)
        embeddings_single_year_collection = ChromaEmbeddingStore(
            db_name=study_name, db_path=vectordb_location
        )
        embeddings = embeddings_single_year_collection.load()

        # Filter export data for this year, create global node ID mapping
        x_cp_single_year = x_cp[x_cp["year"] == year]
        country_ids, _ = get_oredered_node_ids(x_cp_single_year, num_countries)

        # Project embeddings into 2D using UMAP
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        umap_coords = reducer.fit_transform(embeddings[:num_countries])

        df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
        df["country_id"] = country_ids
        df["year"] = year

        if sindices_df is not None and year in sindices_df["year"].values:
            metadata = sindices_df[year].reset_index(drop=True)
            df = pd.concat([metadata, df], axis=1)

        # UMAP Visualization
        plt.figure(figsize=(10, 6))
        if color_by and sindices_df is not None and color_by in df.columns:
            plt.scatter(
                df["UMAP1"],
                df["UMAP2"],
                c=df[color_by],
                cmap="viridis",
                s=50,
            )
            plt.colorbar(label=color_by)
        else:
            plt.scatter(df["UMAP1"], df["UMAP2"], s=50)

        plt.title(f"UMAP Projection for Year {year} (No Clustering)")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Applying HDBSCAN clustering (optional)
        f_with_clustering = kwargs.get("f_with_clustering", False)
        if f_with_clustering:
            min_cluster_size = kwargs.get("min_cluster_size", 5)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            cluster_labels = clusterer.fit_predict(umap_coords)

            df["cluster"] = cluster_labels

            # HDBSCAN clusters Visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(
                df["UMAP1"], 
                df["UMAP2"], 
                c=df["cluster"], 
                cmap="tab10", 
                s=50
            )
            plt.title("UMAP Projection with HDBSCAN Clusters")
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")
            plt.colorbar(label="Cluster")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        records.append(df)

    # result_df = pd.DataFrame(records)

    return pd.concat(records, ignore_index=True)


def visualize_embeddings_with_umap_v2(
    rolling_exports_years=None,
    n_years=None,
    x_cp=None,
    sindices_df=None,
    num_countries=0,
    f_init=False,
    prefix="sage-mor",
    version=None,
    target_score=None,
    top_features=None,
    vectordb_location=None,
    color_by=None,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    **kwargs,
):
    """
    Projects and visualizes country node embeddings using UMAP, optionally followed by HDBSCAN clustering.

    This function reduces high-dimensional embeddings into 2D space using UMAP with various neighborhood sizes,
    and optionally applies HDBSCAN clustering on the resulting projections. It generates side-by-side UMAP
    visualizations colored by a selected target (e.g., log GDP), with optional HDBSCAN cluster overlays.
    Each point is labeled with the corresponding country code.

    Args:
        rolling_exports_years (list): List of years to process (chronologically ordered).
        n_years (int): Number of years from the list to include.
        x_cp (pd.DataFrame): Country-product export DataFrame with a 'year' column.
        sindices_df (pd.DataFrame): Socio-economic indicators with columns: 'year', 'country_id', and targets like 'log_gdp_per_capita'.
        num_countries (int): Number of country nodes (used to slice embeddings).
        f_init (bool): Whether the embeddings are for initialization (used in naming convention).
        prefix (str): Prefix used for the study/embedding name.
        version (str or int): Version tag for the study naming.
        target_score (str): Metric tag (e.g., 'corr', 'r2') used for naming.
        top_features (list): Optional list of structural features used during training (used in naming).
        vectordb_location (str): Path to the ChromaDB where embeddings are stored.
        color_by (str): Column name in `sindices_df` used to color UMAP points (e.g., 'log_gdp_per_capita').
        umap_n_neighbors (int): Default UMAP neighborhood size (overridden by hardcoded settings inside the loop).
        umap_min_dist (float): UMAP parameter controlling how tightly points are packed.
        random_state (int): Seed for reproducibility.
        **kwargs:
            f_with_clustering (bool): If True, applies HDBSCAN clustering.
            min_cluster_size (int): HDBSCAN parameter for minimum cluster size.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing UMAP coordinates, cluster labels (if used),
                    socio-economic metadata, and country identifiers for each projection.
    """
    original_rc = plt.rcParams.copy()  # backup current settings

    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['grid.alpha'] = 0.25
    plt.style.use('seaborn-v0_8-poster')

    f_with_clustering = kwargs.get("f_with_clustering", False)
    f_deterministic = kwargs.get("f_deterministic", False)
    random_seed = 42 if f_deterministic else None
    year_seed = random_seed if f_deterministic else None
    records = []

    for year in rolling_exports_years[:n_years]:
        # print(f"\n{'=' * 60}\nTraining GNN-MoR for year {year}...")
        study_name = get_optuna_study_name(
            prefix=prefix,
            top_features=top_features,
            target_score=target_score,
            year=year,
            f_init=f_init,
            version=version,
            **kwargs,
        )
        print("Study name:", study_name)
        embeddings_single_year_collection = ChromaEmbeddingStore(
            db_name=study_name, db_path=vectordb_location
        )
        embeddings = embeddings_single_year_collection.load()

        # Filter export data for this year, create global node ID mapping
        x_cp_single_year = x_cp[x_cp["year"] == year]
        country_ids, _ = get_oredered_node_ids(x_cp_single_year, num_countries)
        if sindices_df is not None and year in sindices_df["year"].values:
            metadata = sindices_df[sindices_df["year"] == year].reset_index(drop=True)

        umap_neighbor_settings = [5, 15, 30]
        year_seed = year_seed + 1 if f_deterministic else None

        # ─── UMAP-only plots (1 row, 3 columns) ───
        fig_umap, axes_umap = plt.subplots(1, 3, figsize=(18, 6))

        # ─── HDBSCAN plots (2 rows, 3 columns) ───
        if f_with_clustering:
            fig_cluster_u5, ax_u5 = plt.subplots(1, 1, figsize=(14, 12))
            fig_cluster, axes_cluster = plt.subplots(3, 3, figsize=(21, 18))

        for i, n_neighbors in enumerate(umap_neighbor_settings):
            # Project embeddings into 2D using UMAP
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=umap_min_dist,
                random_state=year_seed,
            )
            umap_coords = reducer.fit_transform(embeddings[:num_countries])

            df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
            df["country_id"] = country_ids
            df["year"] = year
            df["umap_n_neighbors"] = n_neighbors

            if sindices_df is not None and year in sindices_df["year"].values:
                df = pd.merge(metadata, df, on="country_id", how="inner")

            # ─── UMAP Plot (no clustering) ───
            ax = axes_umap[i]
            scatter = ax.scatter(
                df["UMAP1"], 
                df["UMAP2"],
                c=df[color_by] if color_by in df.columns else "gray",
                cmap="viridis", 
                s=50
            )
            for _, row in df.iterrows():
                ax.text(row["UMAP1"], row["UMAP2"], row["country_code"], fontsize=6)
            ax.set_title(f"UMAP (n_neighbors={n_neighbors})")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.grid(True)

            # - HDBSCAN Plot (with clustering) - 
            if f_with_clustering:
                min_cluster_size = kwargs.get("min_cluster_size", 5)

                # - 1. HDBSCAN on UMAP coordinates - 
                clusterer_umap = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                cluster_labels_umap = clusterer_umap.fit_predict(umap_coords)
                df["umap_cluster"] = cluster_labels_umap

                # - 2. HDBSCAN on raw high-dimensional embeddings - 
                clusterer_raw = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                cluster_labels_raw = clusterer_raw.fit_predict(embeddings[:num_countries])
                df["hdb_cluster"] = cluster_labels_raw

                # - Row 0: HDBSCAN clusters from UMAP
                ax0 = axes_cluster[0, i]
                scatter_c = ax0.scatter(
                    df["UMAP1"], df["UMAP2"],
                    c=df["umap_cluster"], cmap="tab10", s=55
                )
                # for _, row in df.iterrows():
                #     ax0.text(row["UMAP1"], row["UMAP2"], row["country_code"], fontsize=6)
                ax0.set_title(f"HDBSCAN on UMAP\n(n_neighbors={n_neighbors})")
                ax0.set_xlabel("UMAP1")
                ax0.set_ylabel("UMAP2")
                ax0.grid(True)

                # if i == 1:
                umap_labels = sorted(df["umap_cluster"].dropna().unique())
                umap_colors = plt.cm.tab10(np.linspace(0, 1, len(umap_labels)))
                legend_umap = [
                    Line2D([0], [0], marker='o', color='w', label=str(lbl),
                        markerfacecolor=col, markersize=8)
                    for lbl, col in zip(umap_labels, umap_colors)
                ]
                # legend_umap = [
                #     Line2D([0], [0], marker='o', color='w',  # label=str(lbl),
                #         markerfacecolor=col, markersize=8)
                #     for lbl, col in zip(umap_labels, umap_colors)
                # ]
                ax0.legend(
                    handles=legend_umap,
                    title="UMAP\nClusters",
                    loc='upper left',
                    bbox_to_anchor=(1.00, 1),
                    borderaxespad=0.
                    # loc='lower left', 
                    # bbox_to_anchor=(0, -1.3),
                    # ncol=len(legend_umap),  # Spread items in one row
                )

                if i == 0:
                    scatter_c = ax_u5.scatter(
                        df["UMAP1"], df["UMAP2"],
                        c=df["umap_cluster"], cmap="tab10", s=80
                    )
                    for _, row in df.iterrows():
                        ax_u5.text(row["UMAP1"], row["UMAP2"], row["country_code"], fontsize=12)
                    ax_u5.set_title(f"HDBSCAN on UMAP — Year {year}\n(n_neighbors={n_neighbors})")
                    ax_u5.set_xlabel("UMAP1")
                    ax_u5.set_ylabel("UMAP2")
                    ax_u5.grid(True)

                    ax_u5.legend(
                        handles=legend_umap,
                        title="UMAP\nClusters",
                        loc='upper left',
                        bbox_to_anchor=(1.00, 1),
                        borderaxespad=0.
                    )

                # - Row 1: HDBSCAN clusters from raw embeddings -
                ax1 = axes_cluster[1, i]
                scatter_raw = ax1.scatter(
                    df["UMAP1"], df["UMAP2"],
                    c=df["hdb_cluster"], cmap="tab10", s=55
                )
                # for _, row in df.iterrows():
                #     ax1.text(row["UMAP1"], row["UMAP2"], row["country_code"], fontsize=6)
                ax1.set_title(f"HDBSCAN on Raw Embeddings\n(n_neighbors={n_neighbors})")
                ax1.set_xlabel("UMAP1")
                ax1.set_ylabel("UMAP2")
                ax1.grid(True)

                hdb_labels = sorted(df["hdb_cluster"].dropna().unique())
                hdb_colors = plt.cm.tab10(np.linspace(0, 1, len(hdb_labels)))
                legend_hdb = [
                    Line2D([0], [0], marker='o', color='w', label=str(lbl),
                        markerfacecolor=col, markersize=8)
                    for lbl, col in zip(hdb_labels, hdb_colors)
                ]
                # legend_hdb = [
                #     Line2D([0], [0], marker='o', color='w',  # label=str(lbl),
                #         markerfacecolor=col, markersize=8)
                #     for lbl, col in zip(hdb_labels, hdb_colors)
                # ]
                ax1.legend(
                    handles=legend_hdb, 
                    title="HDBSCAN\nClusters",
                    loc='upper left',
                    bbox_to_anchor=(1.00, 1),
                    borderaxespad=0.
                    # loc='lower left', 
                    # bbox_to_anchor=(0, -0.60),
                    # ncol=len(legend_hdb),  # Spread items in one row
                )

                # - Row 2: GDP coloring -
                ax2 = axes_cluster[2, i]
                scatter_gdp = ax2.scatter(
                    df["UMAP1"], df["UMAP2"], 
                    c=df[color_by], cmap="viridis", s=53
                )
                # for _, row in df.iterrows():
                #     ax2.text(row["UMAP1"], row["UMAP2"], row["country_code"], fontsize=6)
                ax2.set_title(f"Log GDP (n_neighbors={n_neighbors})")
                ax2.set_xlabel("UMAP1")
                ax2.set_ylabel("UMAP2")
                ax2.grid(True)

                # if i == 1:
                # ─── Add legend for Row 2: GDP coloring ───
                cbar = fig_cluster.colorbar(
                    scatter_gdp,
                    ax=ax2,
                    orientation='horizontal',
                    fraction=0.05,
                    pad=0.18
                )
                cbar.set_label(color_by)

            records.append(df)

        fig_umap.tight_layout()
        fig_umap.suptitle(f"UMAP Projections — Year {year}", y=1.02, fontsize=14)

        if f_with_clustering:
            # fig_cluster_u5.suptitle(f"UMAP + HDBSCAN — Year {year}", y=1.01, fontsize=18)
            fig_cluster_u5.tight_layout()
            fig_cluster.tight_layout()  # (rect=[0, 0.4, 1, 1], h_pad=3)
            fig_cluster.subplots_adjust(hspace=0.35)
            fig_cluster.suptitle(f"UMAP + HDBSCAN — Year {year}", y=1.02, fontsize=14)
        
        plt.show()

    plt.rcParams.update(original_rc)  # restore original settings

    return pd.concat(records, ignore_index=True)
