import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr, spearmanr
from p10_Chroma_dbm import ChromaEmbeddingStore
from p12_Utils import get_optuna_study_name


def evaluate_embedding_pc1_vs_gdp(embeddings, gdp_df, num_countries=144):
    # Extract country embeddings (assumed to be the first N rows)
    country_embeddings = embeddings[:num_countries]

    # Ensure it's on CPU and convert to NumPy
    country_np = country_embeddings.numpy()

    # Match GDP values
    y = gdp_df["log_gdp_per_capita"].values  # shape: [144]

    # --------- PCA Correlation (MoR-style) ---------
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(country_np).squeeze()

    correlation, p_value = pearsonr(pc1, y)
    print(f"PCA First Component Correlation with GDP:")
    print(f"   Pearson r = {correlation:.4f}, p = {p_value:.4f}")

    # --------- Linear Regression (Predictive Power) ---------
    model = LinearRegression()
    model.fit(country_np, y)
    r2_score = model.score(country_np, y)
    print(f"Linear Regression R² Score:")
    print(f"   R² = {r2_score:.4f}")


def evaluate_embedding_npcs_vs_gdp(
    embeddings, gdp_df, num_countries=144, num_components=5
):
    """
    Evaluates embeddings using multiple PCA components instead of only the first one.

    - Uses PCA to extract `num_components` principal components from embeddings.
    - Performs a multiple linear regression of the selected components against log GDP per capita.
    - Computes Pearson correlations for each principal component individually.
    - Computes R² score to assess how well embeddings explain GDP.

    Args:
        embeddings (torch.Tensor): Learned node embeddings (shape: [num_nodes, embedding_dim]).
        gdp_df (pd.DataFrame): DataFrame containing 'log_gdp_per_capita' for correlation.
        num_countries (int): Number of country nodes.
        num_components (int): Number of PCA components to use for regression.
    """
    # Extract country embeddings (assumed to be first `num_countries` rows)
    country_embeddings = embeddings[:num_countries]  # .cpu().numpy()
    country_np = country_embeddings.numpy()

    # Extract log GDP per capita values
    y = gdp_df["log_gdp_per_capita"].values  # shape: [num_countries]

    # --------- PCA Transformation ---------
    pca = PCA(n_components=num_components)
    transformed_components = pca.fit_transform(
        country_np
    )  # shape [num_countries, num_components]

    # Print variance explained by each component
    print(
        f"PCA explained variance ratios: {pca.explained_variance_ratio_[:num_components]}"
    )

    # --------- Multiple Linear Regression ---------
    model = LinearRegression()
    model.fit(transformed_components, y)
    r2_score = model.score(transformed_components, y)
    print(f"Multiple Linear Regression R² Score (using {num_components} PCs):")
    print(f"   R² = {r2_score:.4f}")


def evaluate_embedding_pc5_vs_gdp(embeddings, gdp_df, num_countries=144):
    pca = PCA(n_components=32)
    country_embeddings = embeddings[:144]
    components = pca.fit_transform(country_embeddings.numpy())  # shape [144, 32]

    for i in range(5):  # top 5 components
        pc = components[:, i]
        corr, p = pearsonr(pc, gdp_df["log_gdp_per_capita"].values)
        print(f"PC{i+1} vs GDP: r = {corr:.4f}, p = {p:.4f}")


def get_outlier_embeddings(
    embeddings, num_countries=144, country_ids=None, std_multiplier=2.0
):
    """
    Get the outlier embeddings based on Euclidean distance from the center (0,0) and standard deviation multiplier.
    Args:
        embeddings (torch.Tensor): Learned node embeddings (shape: [num_nodes, embedding_dim]).
        num_countries (int): Number of country nodes.
        country_ids (np.ndarray): Array of country IDs corresponding to the embeddings.
        std_multiplier (float): Multiplier for standard deviation to identify outliers (default: 2.0).

    Returns:
        outliers_array (np.ndarray): Array of shape (top_n, 3) containing indices, IDs, and distances.
        reduced (np.ndarray): PCA reduced embeddings for visualization.
    """

    # Slice country embeddings only
    country_emb = embeddings[:num_countries]
    country_emb_np = country_emb.cpu().numpy()

    # Reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(country_emb_np)

    # Euclidean distance from the center (0,0)
    distances = np.linalg.norm(reduced, axis=1)

    # Outlier threshold
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + std_multiplier * std_dist

    # Identify outliers and their distances
    outlier_indices = np.where(distances > threshold)[0]
    outlier_distances = distances[outlier_indices]

    # Sort outlier_indices by distance in descending order
    sorted_idx = np.argsort(outlier_distances)[::-1]
    sorted_outlier_indices = outlier_indices[sorted_idx]

    outliers_array = np.column_stack(
        (
            sorted_outlier_indices,
            country_ids[sorted_outlier_indices],
            distances[sorted_outlier_indices],
        )
    )

    return outliers_array, reduced


def get_top_n_embeddings(embeddings, num_countries=144, country_ids=None, top_n=3):
    """
    Get the top N outlier embeddings based on Euclidean distance from the center (0,0).
    Args:
        embeddings (torch.Tensor): Learned node embeddings (shape: [num_nodes, embedding_dim]).
        num_countries (int): Number of country nodes.
        country_ids (np.ndarray): Array of country IDs corresponding to the embeddings.
        top_n (int): Number of top outliers to return.

    Returns:
        outliers_array (np.ndarray): Array of shape (top_n, 3) containing indices, IDs, and distances.
    """

    # Slice country embeddings only
    country_emb = embeddings[:num_countries]
    country_emb_np = country_emb.cpu().numpy()

    # Reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(country_emb_np)

    # Euclidean distance from the center (0,0)
    distances = np.linalg.norm(reduced, axis=1)

    # Sort by distance in descending order
    outlier_indices = distances.argsort()[::-1]
    outliers_array = np.column_stack(
        (
            outlier_indices[:top_n],
            country_ids[outlier_indices[:top_n]],
            distances[outlier_indices[:top_n]],
        )
    )

    return outliers_array


def evaluate_pca_components_vs_targets(embeddings, targets_dict, num_countries, year):
    """
    Evaluate PCA components against various targets (e.g., GDP, population).
    Args:
        embeddings (torch.Tensor): Learned node embeddings (shape: [num_nodes, embedding_dim]).
        targets_dict (dict): Dictionary containing target values for each target (e.g., {"gdp": gdp_vals, "pop": pop_vals}).
        num_countries (int): Number of country nodes.
        year (int): Year for which to evaluate the targets.

    Returns:
        row (dict): Dictionary containing correlation and Spearman values for each target.
    """
    country_emb_np = embeddings[:num_countries].cpu().numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(country_emb_np)
    pc1, pc2 = reduced[:, 0], reduced[:, 1]

    row = {"year": year}
    for target_name, target_vals in targets_dict.items():
        target = target_vals[:num_countries]

        r1, _ = pearsonr(pc1, target)
        s1, _ = spearmanr(pc1, target)
        r2, _ = pearsonr(pc2, target)
        s2, _ = spearmanr(pc2, target)

        # Naming: [Embedding]_[PC#]_[Target]_[Metric]
        row[f"emb_PC1_{target_name}_r"] = r1
        row[f"emb_PC1_{target_name}_s"] = s1
        row[f"emb_PC2_{target_name}_r"] = r2
        row[f"emb_PC2_{target_name}_s"] = s2

    return row


def get_embeddings_correlations_df(
    rolling_exports_years=None,
    sindices_df=None,
    num_countries=None,
    f_init=False,
    prefix="sage-mor",
    version=None,
    target_columns=None,
    target_score=None,
    top_features=None,
    vectordb_location=None,
    **kwargs,
):
    """
    Batch evaluate correlation of embedding PC1 vs targets across years.

    Args:
        years (list or range): Years to process.
        sindices_df (pd.DataFrame): DataFrame with 'year', 'country_id', targets.
        num_countries (int): Number of country nodes.
        target_columns (list): List of target columns to evaluate.
        embedding_store_getter (callable): Function(year) that returns embeddings for that year.

    Returns:
        pd.DataFrame: Yearly correlations.
    """
    f_deterministic = kwargs.get("f_deterministic", False)
    random_seed = 42 if f_deterministic else None

    target_columns = target_columns if target_columns else ["log_gdp_per_capita"]
    version = version if version else 0
    records = []

    for year in rolling_exports_years:
        # print(f"Processing year {year}...")
        study_name = get_optuna_study_name(
            prefix=prefix,
            top_features=top_features,
            target_score=target_score,
            year=year,
            f_init=f_init,
            version=version,
            **kwargs,
        )
        print(f"Embeddings collection name: {study_name}")
        embeddings_single_year_collection = ChromaEmbeddingStore(
            db_name=study_name, db_path=vectordb_location
        )
        print(f"Collection: \ndb_name={study_name}\ndp_path={vectordb_location}\nstore.collection.name={embeddings_single_year_collection.collection.name}")
        embeddings = embeddings_single_year_collection.load()
        # embeddings = embedding_store_getter(year)

        # Prepare target values
        sindices_single_year = sindices_df[sindices_df["year"] == year].copy()
        sindices_single_year = sindices_single_year.sort_values(
            "country_id"
        ).reset_index(drop=True)

        country_emb = embeddings[:num_countries].cpu()
        country_emb_np = country_emb.numpy()

        # I have found inconsistency: in the "objective_corr" function, which is executed during "hyperparameters tuning" step I am using a `pca = PCA(n_components=1)`
        n_components = 1
        if n_components == 1:
            pca = PCA(n_components=1, random_state=random_seed)
            reduced = pca.fit_transform(country_emb_np)
            pc1 = reduced.squeeze()

            # ###############
            # print("pc1 type:", type(pc1))
            # pca = PCA(n_components=2)
            # reduced = pca.fit_transform(country_emb_np)
            # pc1, pc2 = reduced[:, 0], reduced[:, 1]
            # print("pc1 type:", type(pc1))
            # assert False, "Debugging"
            # ##############

            mask_pc1 = ~np.isnan(pc1)
            row = {"year": year}
            for target_name in target_columns:
                target = sindices_single_year[target_name].values[:num_countries]

                # pearsonr and spearmanr require no missing values, will apply NaN mask for target values
                mask_t = ~np.isnan(target)
                mask_1 = mask_pc1 & mask_t

                # Pearson and Spearman correlation
                if mask_1.sum() > 2:  # Need at least 2 pairs to compute correlation
                    # Here the score is correlation coefficient
                    corr_p, pp_value = pearsonr(pc1[mask_1], target[mask_1])
                    r_pc1 = abs(corr_p)
                    corr_s, sp_value = spearmanr(pc1[mask_1], target[mask_1])
                    s_pc1 = abs(corr_s)
                else:
                    r_pc1, s_pc1 = np.nan, np.nan

                r_pc2, s_pc2 = np.nan, np.nan

                # Column naming: heci__target_r/s_vN
                row[f"emb_PC1__{target_name}_r_v{version}"] = r_pc1
                row[f"emb_PC1__{target_name}_s_v{version}"] = s_pc1
                row[f"emb_PC2__{target_name}_r_v{version}"] = r_pc2
                row[f"emb_PC2__{target_name}_s_v{version}"] = s_pc2

        else:
            pca = PCA(n_components=2, random_state=random_seed)
            reduced = pca.fit_transform(country_emb_np)
            pc1, pc2 = reduced[:, 0], reduced[:, 1]

            # We apply masks for embeddings, but actually at this point we expect embeddings to be all there
            mask_pc1 = ~np.isnan(pc1)
            mask_pc2 = ~np.isnan(pc2)

            row = {"year": year}

            for target_name in target_columns:
                target = sindices_single_year[target_name].values[:num_countries]

                # pearsonr and spearmanr require no missing values, will apply NaN mask for target values
                mask_t = ~np.isnan(target)
                mask_1 = mask_pc1 & mask_t
                mask_2 = mask_pc2 & mask_t

                # Pearson and Spearman correlation
                if mask_1.sum() > 2:  # Need at least 2 pairs to compute correlation
                    # Here the score is correlation coefficient
                    corr_p, pp_value = pearsonr(pc1[mask_1], target[mask_1])
                    r_pc1 = abs(corr_p)
                    corr_s, sp_value = spearmanr(pc1[mask_1], target[mask_1])
                    s_pc1 = abs(corr_s)
                else:
                    r_pc1, s_pc1 = np.nan, np.nan

                if mask_2.sum() > 2:  # Need at least 2 pairs to compute correlation
                    r_pc2, _ = pearsonr(pc2[mask_2], target[mask_2])
                    s_pc2, _ = spearmanr(pc2[mask_2], target[mask_2])
                else:
                    r_pc2, s_pc2 = np.nan, np.nan

                # Column naming: heci__target_r/s_vN
                row[f"emb_PC1__{target_name}_r_v{version}"] = r_pc1
                row[f"emb_PC1__{target_name}_s_v{version}"] = s_pc1
                row[f"emb_PC2__{target_name}_r_v{version}"] = r_pc2
                row[f"emb_PC2__{target_name}_s_v{version}"] = s_pc2

        records.append(row)

    df_result = pd.DataFrame(records)

    return df_result


def train_regressions_on_embeddings(
    rolling_exports_years=None,
    n_years=None,
    sindices_df=None,
    num_countries=0,
    f_init=False,
    study_prefix="sage-mor",
    version=None,
    top_features=None,
    target_score="corr",
    vectordb_location=None,
    target_columns=None,
    cv=5,
    scoring="r2",
    # f_debug=False,
    **kwargs,
):
    """
    Train regression models on full embeddings per year and multiple targets.

    Args:
        sindices_df (pd.DataFrame): DataFrame containing all target indicators, with 'year' and target columns.
        target_columns (list): List of target column names to use for regression.
        rolling_exports_years (list): List of years to iterate through.
        vectordb_location (str): Path to ChromaDB where embeddings are stored.
        study_prefix (str): Prefix used in study name.
        top_features (list): Topological features list used in training.
        target_score (str): Scoring metric used in study name (e.g., 'corr', 'r2').
        version (str or int): Version used in study naming.
        num_countries (int): Number of countries (embeddings to slice).
        cv (int): Cross-validation folds.
        scoring (str): Scoring metric for cross_val_score.

    Returns:
        pd.DataFrame: DataFrame with columns ["year", f"{model}_{target}_{scoring}", ...]
    """
    f_deterministic = kwargs.get("f_deterministic", False)
    random_seed = 42 if f_deterministic else None
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=random_seed),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_seed),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1, ),  # deterministic by design, make sure to check random seed for cross validdation
        # "ModelName": ModelClass(...),
    }
    cv = kwargs.get("cv", 5)
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
    scoring = kwargs.get("scoring", "r2")
    f_debug = kwargs.get("f_debug", False)
    f_regr_analysis = kwargs.get("f_regr_analysis", False)

    records = []

    for year in rolling_exports_years[:n_years]:
        row = {"year": year}

        study_name = get_optuna_study_name(
            prefix=study_prefix,
            top_features=top_features,
            target_score=target_score,
            year=year,
            f_init=f_init,
            version=version,
            **kwargs,
        )
        # if f_debug:
        print(f"DEBUG: Embeddings name: {study_name}")

        embeddings_single_year_collection = ChromaEmbeddingStore(
            db_name=study_name, db_path=vectordb_location
        )
        if f_debug:
            print(
                "DEBUG: Collection list:",
                *embeddings_single_year_collection.list_collections(),
                "\n",
                sep="\n",
            )

        embeddings = embeddings_single_year_collection.load()
        country_emb = embeddings[:num_countries]
        country_emb_np = country_emb.cpu().numpy()

        # Prepare target values
        sindices_single_year = sindices_df[sindices_df["year"] == year].copy()
        sindices_single_year = sindices_single_year.sort_values(
            "country_id"
        ).reset_index(drop=True)
        if sindices_single_year.empty:
            print(f"[WARN] No targets for year {year}. Skipping.")
            continue

        for target_col in target_columns:
            if target_col not in sindices_single_year.columns:
                print(f"[WARN] Target {target_col} missing in sindices_df.")
                continue

            # Prepare target values
            y_raw = sindices_single_year[target_col].values
            notna_mask = ~np.isnan(y_raw)
            if not np.any(notna_mask):
                print(
                    f"[WARN] All values missing for target {target_col} in year {year}. Skipping."
                )
                continue

            # Removing missing values
            y = y_raw[notna_mask]
            # Ensuring allignment of embeddings and targets
            country_emb_notna = country_emb_np[notna_mask]
            for model_name, model in models.items():
                try:
                    # Cross-validated score
                    scores = cross_val_score(
                        model, 
                        country_emb_notna, 
                        y, 
                        cv=cv_splitter, 
                        scoring=scoring
                    )
                    key = f"{model_name}_{target_col}_{scoring}"
                    row[key] = scores.mean()

                    if f_regr_analysis:
                        print(f"model: {model_name}, target: {target_col}")

                    # Fit model and calculate Pearson r
                    model.fit(country_emb_notna, y)
                    y_pred = model.predict(country_emb_notna)
                    r_val, _ = pearsonr(y, y_pred)
                    row[f"{model_name}_{target_col}_corr"] = r_val

                    if f_regr_analysis:
                        print(f"model: {model_name, year}\nnp.std(y_pred) = {np.std(y_pred)}, np.std(y) = {np.std(y)}, np.isnan(y_pred).any() = {np.isnan(y_pred).any()}, np.isnan(y).any() = {np.isnan(y).any()}")
                        
                        if np.std(y_pred) == 0:
                            print(f"==>> Constant prediction for {model_name} on {target_col} in year {year}")
                        else:
                            print(f"==>> Pearson r = {r_val:.4f}")

                except Exception as e:
                    row[f"{model_name}_{target_col}_{scoring}"] = np.nan

                    debug_msg = "".join([": ", str(e)]) if f_debug else ""
                    print(
                        f"[ERROR] {model_name} failed on {target_col} for year {year}{debug_msg}"
                    )

        records.append(row)

        if f_debug:
            print(f"DEBUG: The debug mode is on. Skipping the rest of the years.")
            break

    result_df = pd.DataFrame(records)

    return result_df


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
