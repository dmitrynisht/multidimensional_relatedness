import optuna
from optuna import logging
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from p12_Utils import set_seed
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from p10_Chroma_dbm import ChromaEmbeddingStore
from p12_Utils import get_optuna_study_name, get_optuna_storage_uri
from p14_GraphSAGE_v04_dynamic_optuned import (
    objective_r2, 
    objective_corr, 
    objective_corr_sye, 
    objective_corr_sye_m_loss
)
from p14_GraphSAGE_v04_dynamic_optuned import (
    optuna_train_TemporalSAGE_pure_vdb,
    optuna_train_TemporalSAGE_pure_vdb_sye,
    optuna_train_TemporalSAGE_pure_vdb_sye_m_loss,
    get_oredered_node_ids,
)
from p14_GraphSAGE_v05_static_optuned import (
    objective_corr_sye_u_loss,
    optuna_train_sye_sage_pure_conv_u_loss_bi_ms
)
from p14_GraphSAGE_v_visual import (
    visualize_embeddings_2PCA,
    lineplot_embedding_correlations_vs_targets,
    lineplot_embedding_correlations_faceted,
    visualize_country_embeddings_detect_outliers,
    visualize_embeddings_with_umap,
    visualize_embeddings_with_umap_v2,
)
from p14_GraphSAGE_v_evaluate import (
    get_outlier_embeddings,
    get_embeddings_correlations_df,
    train_regressions_on_embeddings,
)


def load_or_create_optuna_study(
    study_name=None,
    storage_uri=None,
    direction="maximize",
    seed=None,
    n_warmup_steps=0,
    f_create_if_not_exists=False,
    **kwargs,
):
    """
    Load an existing Optuna study or create a new one if it doesn't exist.
    Args:
        direction (str): The optimization direction ('minimize' or 'maximize').
        study_name (str): The name of the study.
        storage_uri (str): The URI for the storage backend.
        sampler (optuna.samplers.BaseSampler): The sampler to use for the study.
        **kwargs: Additional arguments to pass to `optuna.create_study`.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
        f_it_is_first_run = False
        print(f"Study '{study_name}' loaded successfully.\n")
    except Exception as e:
        print(f"Study '{study_name}' not found, because:\n==>> {e}.\n")
        # return "For testing aids, skipped the study creation."
        if f_create_if_not_exists:
            print(f"Creating a new study '{study_name}'...\n")
            sampler = TPESampler(seed=seed)
            pruner = MedianPruner(n_warmup_steps=n_warmup_steps)
            study = optuna.create_study(
                direction=direction,
                study_name=study_name,
                storage=storage_uri,
                sampler=sampler,
                pruner=pruner,
            )
            f_it_is_first_run = True
        else:
            print("==>> Study creation skipped.\n")
            study = None
            f_it_is_first_run = False

    return study, f_it_is_first_run


def training_models__storing_embeddings(
    rolling_exports_years=None,
    n_years=None,
    x_cp=None,
    # gdp_df=None,
    # num_countries=0,
    f_init=False,
    os_path=None,
    prefix="sage-mor",
    version=None,
    optuna_study_location=None,
    target_score=None,
    top_features=None,
    vectordb_location=None,
    # location_df=None,
    trial_name=None,
    **kwargs,
):
    f_debug = kwargs.pop("f_debug", False)
    if "node_type_encoding" in kwargs:
        node_type_encoding = kwargs.pop("node_type_encoding", "onehot")
    for year in rolling_exports_years[:n_years]:
        print(f"\n{'=' * 60}\nTraining GNN-MoR for year {year}...")
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

        # Filter export data and GDP for this year
        x_cp_single_year = x_cp[x_cp["year"] == year]
        # y = gdp_df[gdp_df["year"] == year]["log_gdp_per_capita"].values

        # Define Optuna study location
        storage_uri = get_optuna_storage_uri(
            study_name=study_name,
            os_path=os_path,
            optuna_study_location=optuna_study_location,
        )

        # Load Optuna study
        f_create_if_not_exists = False  # We expect the study to exist for each year
        study, f_it_is_first_run = load_or_create_optuna_study(
            study_name=study_name,
            storage_uri=storage_uri,
            f_create_if_not_exists=f_create_if_not_exists,
        )
        best_params = study.best_params
        if "node_type_encoding" not in best_params:
            best_params["node_type_encoding"] = node_type_encoding

        # db_name = f"sage-mor-{'-'.join(top_features)}-t_{target_score}-{year}{'-init' if f_init else ''}"
        sage_dynamic_single_year_store = ChromaEmbeddingStore(
            db_name=study_name, db_path=vectordb_location
        )
        print(f"Embedding collection name: {sage_dynamic_single_year_store.db_name}")
        print(f"Embedding collection is not None: {sage_dynamic_single_year_store is not None}; Embedding store type: {type(sage_dynamic_single_year_store)}")
        embeddings_single_year_collection = sage_dynamic_single_year_store
        return_embeddings = False  # This arg is actually not used in the function, but it should indicate intention to store ebeddings in the vector database

        if "updated_loss" in kwargs:
            # model, embeddings = optuna_train_TemporalSAGE_pure_vdb_sye(
            #     **extract_best_params(
            #         df=x_cp_single_year,
            #         f_debug=f_debug,
            #         return_embeddings=return_embeddings,
            #         embedding_store=embeddings_single_year_collection,
            #         trial_name=trial_name,
            #         best_params=best_params,
            #     ),
            #     **kwargs,
            # )
            model, embeddings = optuna_train_TemporalSAGE_pure_vdb_sye_m_loss(
                **extract_best_params(
                    df=x_cp_single_year,
                    f_debug=f_debug,
                    return_embeddings=return_embeddings,
                    embedding_store=embeddings_single_year_collection,
                    trial_name=trial_name,
                    best_params=best_params,
                ),
                **kwargs,
            )
        else:
            model, embeddings = optuna_train_TemporalSAGE_pure_vdb(
                **extract_best_params(
                    df=x_cp_single_year,
                    f_debug=f_debug,
                    return_embeddings=return_embeddings,
                    embedding_store=embeddings_single_year_collection,
                    trial_name=trial_name,
                    best_params=best_params,
                )
            )

    pass


def embeddings_evaluation_visualization(
    rolling_exports_years=None,
    n_years=None,
    x_cp=None,
    gdp_df=None,
    num_countries=0,
    f_init=False,
    # os_path=None,
    prefix="sage-mor",
    version=None,
    # optuna_study_location=None,
    target_score=None,
    top_features=None,
    vectordb_location=None,
    location_df=None,
    target_correlations_df=None,
    target_columns=None,
    **kwargs,
):
    version = version if version else 0

    embeddings_correlations_df = get_embeddings_correlations_df(
        rolling_exports_years=rolling_exports_years,
        sindices_df=gdp_df,
        num_countries=num_countries,
        f_init=f_init,
        prefix=prefix,
        version=version,
        target_columns=target_columns,
        target_score=target_score,
        top_features=top_features,
        vectordb_location=vectordb_location,
        **kwargs,
    )

    full_correlations_df = target_correlations_df.merge(
        embeddings_correlations_df, on="year", how="left"
    )

    reg_scores_df = train_regressions_on_embeddings(
        rolling_exports_years=rolling_exports_years,
        n_years=n_years,
        sindices_df=gdp_df,
        num_countries=num_countries,
        f_init=f_init,
        study_prefix=prefix,
        version=version,
        top_features=top_features,
        vectordb_location=vectordb_location,
        target_columns=target_columns,
        target_score=target_score,
        **kwargs,
    )

    # Showing correlations with all Socioeconomic Indicators, and ECI(MoR) as a benchmark
    show_sei_correlations_cols = ["eci_18__log_gdp_pcap"]
    show_sei_correlations_cols.extend(
        [
            # *full_correlations_df.columns[6:]
            *full_correlations_df.columns[2:]
        ]
    )
    # Debug
    print(f"show_sei_correlations_cols: {show_sei_correlations_cols}")
    #
    y_limits = lineplot_embedding_correlations_vs_targets(
        correlation_df=full_correlations_df,
        target_columns=show_sei_correlations_cols,
        title="Embeddings (PC1, PC2) vs Socioeconomic Indicators (Pearson r & Spearman r)",
    )
    # # Debugging
    # print(f"y_limits: {y_limits}")
    # #
    # return None, None

    # Showing only the correlations with GDP per capita
    show_gdp_correlations_cols = [
        "eci_18__log_gdp_pcap",
        f"emb_PC1__log_gdp_per_capita_r_v{version}",
        f"emb_PC1__log_gdp_per_capita_s_v{version}",
        f"emb_PC2__log_gdp_per_capita_r_v{version}",
        f"emb_PC2__log_gdp_per_capita_s_v{version}",
    ]
    _ = lineplot_embedding_correlations_vs_targets(
        correlation_df=full_correlations_df,
        target_columns=show_gdp_correlations_cols,
        title="Embeddings (PC1, PC2) and ECI vs GDP per capita (Pearson r)",
    )

    # Showing only the correlations with GDP per capita, HCI, and ECI(MoR) as a benchmark
    # Pearson r and Spearman r correlations in different columns
    # PC1 and PC2 components in different lines
    show_gdp_correlations_cols = [
        f"emb_PC1__log_gdp_per_capita_r_v{version}",
        f"emb_PC1__log_gdp_per_capita_s_v{version}",
        f"emb_PC2__log_gdp_per_capita_r_v{version}",
        f"emb_PC2__log_gdp_per_capita_s_v{version}",
        f"emb_PC1__hci_r_v{version}",
        f"emb_PC1__hci_s_v{version}",
        f"emb_PC2__hci_r_v{version}",
        f"emb_PC2__hci_s_v{version}",
    ]
    lineplot_embedding_correlations_faceted(
        correlation_df=full_correlations_df,
        target_columns=show_gdp_correlations_cols,
        benchmark_column="eci_18__log_gdp_pcap",
        y_limits=y_limits,
    )

    # Showing only the correlations with GINI, and ECI(MoR) as a benchmark
    # Pearson r and Spearman r correlations in different columns
    # PC1 and PC2 components in different lines
    show_gdp_correlations_cols = [
        f"emb_PC1__gini_r_v{version}",
        f"emb_PC1__gini_s_v{version}",
        f"emb_PC2__gini_r_v{version}",
        f"emb_PC2__gini_s_v{version}",
        f"emb_PC1__gini_disp_r_v{version}",
        f"emb_PC1__gini_disp_s_v{version}",
        f"emb_PC2__gini_disp_r_v{version}",
        f"emb_PC2__gini_disp_s_v{version}",
    ]
    lineplot_embedding_correlations_faceted(
        correlation_df=full_correlations_df,
        target_columns=show_gdp_correlations_cols,
        benchmark_column="eci_18__log_gdp_pcap",
        y_limits=y_limits,
    )

    # Sanity check
    print("kwargs: ", kwargs)
    # Using PCA to visualize the embeddings
    # PC1 and PC2 components in different lines
    outlier_embedding_countries = visualize_country_embeddings_detect_outliers(
        rolling_exports_years=rolling_exports_years,
        n_years=n_years,
        x_cp=x_cp,
        gdp_df=gdp_df,
        num_countries=num_countries,
        f_init=f_init,
        prefix=prefix,
        version=version,
        target_score=target_score,
        top_features=top_features,
        vectordb_location=vectordb_location,
        location_df=location_df,
        **kwargs,
    )

    # umap_projections_df = visualize_embeddings_with_umap(
    #     rolling_exports_years=rolling_exports_years,
    #     n_years=n_years,
    #     x_cp=x_cp,
    #     sindices_df=gdp_df,
    #     num_countries=num_countries,
    #     f_init=f_init,
    #     prefix=prefix,
    #     version=version,
    #     target_score=target_score,
    #     top_features=top_features,
    #     vectordb_location=vectordb_location,
    #     **kwargs,
    # )
    umap_projections_df = visualize_embeddings_with_umap_v2(
        rolling_exports_years=rolling_exports_years,
        n_years=n_years,
        x_cp=x_cp,
        sindices_df=gdp_df,
        num_countries=num_countries,
        f_init=f_init,
        prefix=prefix,
        version=version,
        target_score=target_score,
        top_features=top_features,
        vectordb_location=vectordb_location,
        **kwargs,
    )

    return (
        full_correlations_df,
        outlier_embedding_countries,
        reg_scores_df,
        umap_projections_df,
    )


def run_training_evaluation_steps(
    rolling_exports_years=None,
    n_years=None,
    x_cp=None,
    gdp_df=None,
    num_countries=0,
    f_init=False,
    os_path=None,
    prefix="sage-mor",
    version=None,
    optuna_study_location=None,
    target_score=None,
    top_features=None,
    vectordb_location=None,
    trial_name=None,
    location_df=None,
    target_correlations_df=None,
    target_columns=None,
    **kwargs,
):
    f_skip_training = kwargs.get("f_skip_training", False)
    if f_skip_training:
        pass
    else:
        training_models__storing_embeddings(
            rolling_exports_years=rolling_exports_years,
            n_years=n_years,
            x_cp=x_cp,
            # gdp_df=sindices_df,
            # num_countries=num_countries,
            f_init=f_init,
            os_path=os_path,
            prefix=prefix,
            version=version,
            optuna_study_location=optuna_study_location,
            target_score=target_score,
            top_features=top_features,
            vectordb_location=vectordb_location,
            # location_df=location_df,
            trial_name=trial_name,
            **kwargs,
        )

    (
        full_correlations_df,
        outlier_embedding_countries,
        reg_scores_df,
        umap_projections_df,
    ) = embeddings_evaluation_visualization(
        rolling_exports_years=rolling_exports_years,
        n_years=n_years,
        x_cp=x_cp,
        gdp_df=gdp_df,
        num_countries=num_countries,
        f_init=f_init,
        # os_path=os_path,
        prefix=prefix,
        version=version,
        # optuna_study_location=optuna_study_location,
        target_score=target_score,
        top_features=top_features,
        vectordb_location=vectordb_location,
        location_df=location_df,
        target_correlations_df=target_correlations_df,
        target_columns=target_columns,
        **kwargs,
    )

    return (
        full_correlations_df,
        outlier_embedding_countries,
        reg_scores_df,
        umap_projections_df,
    )


def run_tuning_steps(
    random_seed=None,
    n_warmup_steps=0,
    direction="maximize",
    n_trials=1,
    study_expected_best_value=0.0,
    f_enqueue_hyperparams=True,
    f_debug=False,
    study_hyperparameters=None,
    f_use_initial_hyperparams=True,
    study_name=None,
    f_init=False,
    os_path=None,
    optuna_study_location=None,
    rolling_exports_years=None,
    n_years=None,
    x_cp=None,
    gdp_df=None,
    num_countries=0,
    target_score=None,
    f_iter_top_features=False,
    top_features=None,
    trial_name=None,
    **kwargs,
):

    if f_use_initial_hyperparams:
        study_name = study_name
        storage_uri = get_optuna_storage_uri(
            study_name=study_name,
            os_path=os_path,
            optuna_study_location=optuna_study_location,
        )
        try:
            study_hyperparameters = (
                {
                    "num_epochs": 7,
                    "lr": 0.0001124186209579306,
                    "lambda_temporal": 0.10789142699330445,
                    "retain_graph": False,
                    "hidden_feats": 66,
                    "out_feats": 24,
                    "dropout": 0.4537832369630465,
                    "num_layers": 2,
                    "agg_layer_0": "pool",
                    "agg_layer_1": "pool",
                    "gru_num_layers": 3,
                    "use_relu": False,
                    "use_random_noise": True,
                    "node_type_encoding": "onehot",
                }
                if study_hyperparameters is None
                else study_hyperparameters
            )
            study = optuna.load_study(study_name=study_name, storage=storage_uri)

            study_hyperparameters = study.best_params
        except Exception as e:
            print(f"Study '{study_name}' not found\n")

    print("study initial hyperparameters:", study_hyperparameters)

    target_score = target_score if bool(target_score) else "r2"
    studies_by_year = {}
    for year in rolling_exports_years[:n_years]:
        print(f"\n{'=' * 60}\nRunning GNN-MoR for year {year}...")
        study_name = get_optuna_study_name(
            top_features=top_features,
            target_score=target_score,
            year=year,
            f_init=f_init,
            **kwargs,
        )
        # study_name = f"sage-mor-{'-'.join(top_features)}-t_{target_score}-{year}{'-init' if f_init else ''}"
        print("Study name:", study_name)

        # Filter export data and GDP for this year
        x_cp_single_year = x_cp[x_cp["year"] == year]
        y = gdp_df[gdp_df["year"] == year]["log_gdp_per_capita"].values

        # Define Optuna study location
        storage_uri = get_optuna_storage_uri(
            study_name=study_name,
            os_path=os_path,
            optuna_study_location=optuna_study_location,
        )

        # Create Optuna study
        f_create_if_not_exists = True  # We are creating a new study for each year
        study, f_it_is_first_run = load_or_create_optuna_study(
            study_name=study_name,
            storage_uri=storage_uri,
            f_create_if_not_exists=f_create_if_not_exists,
            seed=random_seed,
            n_warmup_steps=n_warmup_steps,
            direction=direction,
        )
        studies_by_year[year] = study

        study_best_trial = run_batch_of_trials(
            f_it_is_first_run=f_it_is_first_run,  # At this stage it should be False
            study_hyperparameters=study_hyperparameters,
            f_debug=f_debug,
            f_enqueue_hyperparams=f_enqueue_hyperparams,
            n_trials=n_trials,
            study=study,
            df=x_cp_single_year,
            y=y,
            num_countries=num_countries,
            study_expected_best_value=study_expected_best_value,
            target_score=target_score,
            f_iter_top_features=f_iter_top_features,
            top_features=top_features,
            trial_name=trial_name,
            **kwargs,
        )

        # Enqueue best hyperparameters for the next study
        study_hyperparameters = study_best_trial.params

    return studies_by_year


def run_batch_of_trials(
    f_it_is_first_run=False,
    study_hyperparameters=None,
    f_debug=False,
    f_enqueue_hyperparams=False,
    n_trials=1,
    study=None,
    df=None,
    y=None,
    num_countries=0,
    study_expected_best_value=0.0,
    target_score=None,
    f_iter_top_features=False,
    top_features=None,
    trial_name=None,
    **kwargs,
):
    """
    Run a batch of trials for the Optuna study.

    Args:
        f_it_is_first_run (bool): Whether this is the first run of the study.
        study_hyperparameters (dict): The hyperparameters for the study.
        f_debug (bool): Whether to run in debug mode.
        f_enqueue_hyperparams (bool): Whether to enqueue hyperparameters for the next study.
        n_trials (int): The number of trials to run.
        study (optuna.study.Study): The Optuna study.
        df (pandas.DataFrame): The dataframe.
        y (pandas.Series): The target variable.
        num_countries (int): The number of countries.
        study_expected_best_value (float): The expected best value for the study.
        target_score (float): The target score.
        f_iter_top_features (bool): Whether to iterate over top features.
        top_features (list): The top features.
        trial_name (str): The trial prefix name, which would be shown in the report messages during the execution. If not provided, it will be set to the default value "Trial ".
    Returns:
        best_trial (optuna.trial.FrozenTrial): The best trial from the study.

    """
    if study is None:
        return

    if df is None:
        return

    if y is None:
        return

    if num_countries == 0:
        return

    # if f_it_is_first_run and f_enqueue_hyperparams and study_hyperparameters is None:
    #     return

    # if f_debug and f_enqueue_hyperparams and study_hyperparameters is None:
    #     return

    if f_debug:
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    if (
        (f_it_is_first_run or f_debug)
        and f_enqueue_hyperparams
        and study_hyperparameters is not None
    ):
        study.enqueue_trial(study_hyperparameters)

    target_score = target_score if bool(target_score) else "r2"
    if target_score == "r2":
        # this call is equivalent to initial:
        # lambda trial: objective
        objective_func = objective_r2
        top_features = top_features if bool(top_features) else ["nd"]
    elif target_score == "corr":
        objective_func = objective_corr
        top_features = top_features if bool(top_features) else ["nd", "cd"]
    elif target_score == "corr_sye":
        print("Will be used 'objective_corr_sye' objective function with updated loss-calculation")
        objective_func = objective_corr_sye
        top_features = top_features if bool(top_features) else ["nd", "cd"]
    elif target_score == "corr_sye_m_loss":
        print("Will be used 'objective_corr_sye_m_loss' objective function with min loss-calculation")
        objective_func = objective_corr_sye_m_loss
        top_features = top_features if bool(top_features) else ["nd", "cd"]
    elif target_score == "corr_sye_u_loss":
        print("Will be used 'objective_corr_sye_u_loss' objective function with unsupervised loss-calculation")
        objective_func = objective_corr_sye_u_loss
        top_features = top_features if bool(top_features) else ["nd", "cd"]
    elif target_score == "corr_sye_u_loss_bi_ms":
        print("Will be used 'objective_corr_sye_u_loss_bi_ms' objective function with unsupervised loss-calculation and bipartite metrics")
        objective_func = objective_corr_sye_u_loss
        top_features = top_features if bool(top_features) else ["wosh"]
        kwargs["learn_embeddings_func"] = optuna_train_sye_sage_pure_conv_u_loss_bi_ms

    print("Starting optimization...")
    if f_it_is_first_run:
        set_seed(42)
    study.optimize(
        lambda trial: objective_func(
            trial=trial,
            df=df,
            y=y,
            num_countries=num_countries,
            f_debug=f_debug,
            f_iter_top_features=f_iter_top_features,
            top_features=top_features,
            trial_name=trial_name,
            **kwargs,
        ),
        n_trials=n_trials,
    )  # , n_trials=5)
    print("")
    print("=" * 60, sep="\n")
    print("Best trial:", study.best_trial)
    print("Best trial:", study.best_value)
    print("Best trial:", study.best_params)

    assert (
        study.best_value >= study_expected_best_value
    ), "Best trial is not good enough"

    return study.best_trial


def get_user_attrs_hyperparams(
    user_attr_name="hyperparameters_r2_9999_fast", study=None
):
    """
    Get the hyperparameters from the study's user attributes.
    If the study or the user attribute is not found, return the default hyperparameters values, which are known to wokr with r-squared of 0.9100.
    study_hyperparameters_v1 are named after the first the best trial found in the study, but I know there exist better hyperparameters.
    Args:
        user_attr_name (str): The name of the user attribute.
        study (optuna.study.Study): The study to get the hyperparameters from.
    Returns:
        dict: The hyperparameters as a dictionary.
    """
    try:
        study_hyperparameters_v1 = None
        if user_attr_name in study.user_attrs:
            print(f"{user_attr_name} was found in user_attrs dict")
            study_hyperparameters_v1 = study.user_attrs.get(user_attr_name)
        else:
            raise KeyError("The user_attr_name is not in user_attrs dict")
    except:
        print(
            f"The study doesn't exist or {user_attr_name} is not in user_attrs dict and was returned from a template"
        )
        study_hyperparameters_v1 = {
            "num_epochs": 5,
            "lr": 0.004587784068375952,
            "lambda_temporal": 0.8090768579596355,
            "retain_graph": True,
            "hidden_feats": 159,
            "out_feats": 22,
            "dropout": 0.21704941181926313,
            "num_layers": 2,
            "agg_layer_0": "pool",
            "agg_layer_1": "pool",
            "gru_num_layers": 1,
            "use_relu": True,
            "use_random_noise": True,
            "node_type_encoding": "onehot",
        }  # 0.99
        # {
        #     "num_epochs": 7,
        #     "lr": 0.0061295007078484625,
        #     "lambda_temporal": 0.6049722646002988,
        #     "retain_graph": True,
        #     "hidden_feats": 102,
        #     "out_feats": 15,
        #     "dropout": 0.11725380007571568,
        #     "num_layers": 2,
        #     "agg_layer_0": "gcn",
        #     "agg_layer_1": "pool",
        #     "gru_num_layers": 2,
        #     "use_relu": False,
        #     "use_random_noise": True,
        #     "node_type_encoding": "none",
        # } # 0.7103

    return study_hyperparameters_v1


def depricated_run():
    """This is how initially was done before introducing the function run_batch_of_trials."""
    # print("Starting optimization...")
    # if f_it_is_first_run:
    #     set_seed(42)
    # study.optimize(
    #     lambda trial: objective(
    #         trial=trial, df=x_cp, y=y, num_countries=num_countries, f_debug=f_debug
    #     ),
    #     n_trials=n_trials,
    # )  # , n_trials=5)
    # print("")
    # print("=" * 60, sep="\n")
    # print("Best trial:", study.best_trial)
    # print("Best trial:", study.best_value)
    # print("Best trial:", study.best_params)

    # assert study.best_value >= study_expected_best_value, "Best trial is not good enough"

    # # Example of the output:
    # [I 2025-04-07 03:25:05,807] Trial 13 finished with value: 0.9047807367276105 and parameters: {'num_epochs': 2, 'lr': 0.005182394897119479, 'lambda_temporal': 0.8080075821063994, 'retain_graph': False, 'hidden_feats': 127, 'out_feats': 24, 'dropout': 0.14350535268204173, 'num_layers': 3, 'agg_layer_0': 'pool', 'agg_layer_1': 'gcn', 'agg_layer_2': 'gcn', 'gru_num_layers': 1, 'use_relu': True, 'use_random_noise': True, 'node_type_encoding': 'onehot'}. Best is trial 13 with value: 0.9047807367276105.
    # [I 2025-04-07 03:55:50,802] Trial 22 finished with value: 0.8865713647274407 and parameters: {'num_epochs': 5, 'lr': 0.006302937335757843, 'lambda_temporal': 0.6194373887242279, 'retain_graph': False, 'hidden_feats': 128, 'out_feats': 18, 'dropout': 0.09430645783140884, 'num_layers': 3, 'agg_layer_0': 'pool', 'agg_layer_1': 'gcn', 'agg_layer_2': 'gcn', 'gru_num_layers': 1, 'use_relu': True, 'use_random_noise': True, 'node_type_encoding': 'onehot'}. Best is trial 13 with value: 0.9047807367276105.
    # [I 2025-04-07 04:02:36,657] Trial 26 finished with value: 0.8355919271375576 and parameters: {'num_epochs': 5, 'lr': 0.002110940159710195, 'lambda_temporal': 0.6675553523312674, 'retain_graph': False, 'hidden_feats': 123, 'out_feats': 39, 'dropout': 0.39682177214054215, 'num_layers': 4, 'agg_layer_0': 'pool', 'agg_layer_1': 'gcn', 'agg_layer_2': 'pool', 'agg_layer_3': 'lstm', 'gru_num_layers': 2, 'use_relu': True, 'use_random_noise': True, 'node_type_encoding': 'onehot'}. Best is trial 13 with value: 0.9047807367276105.
    # [I 2025-04-07 04:15:35,945] Trial 29 finished with value: 0.8403543145812609 and parameters: {'num_epochs': 3, 'lr': 0.0007886014449181702, 'lambda_temporal': 0.7090441521583267, 'retain_graph': True, 'hidden_feats': 113, 'out_feats': 28, 'dropout': 0.08658550138927229, 'num_layers': 3, 'agg_layer_0': 'pool', 'agg_layer_1': 'pool', 'agg_layer_2': 'gcn', 'gru_num_layers': 2, 'use_relu': False, 'use_random_noise': False, 'node_type_encoding': 'none'}. Best is trial 13 with value: 0.9047807367276105.
    # [I 2025-04-07 04:16:02,035] Trial 31 finished with value: 0.9036740591986112 and parameters: {'num_epochs': 3, 'lr': 0.007533227438051657, 'lambda_temporal': 0.6184198937204007, 'retain_graph': False, 'hidden_feats': 126, 'out_feats': 19, 'dropout': 0.15538395635914484, 'num_layers': 5, 'agg_layer_0': 'pool', 'agg_layer_1': 'gcn', 'agg_layer_2': 'gcn', 'agg_layer_3': 'mean', 'agg_layer_4': 'pool', 'gru_num_layers': 1, 'use_relu': True, 'use_random_noise': True, 'node_type_encoding': 'onehot'}. Best is trial 13 with value: 0.9047807367276105.

    # [I 2025-04-07 04:16:12,414] Trial 32 finished with value: 0.9185270656386892 and parameters: {'num_epochs': 4, 'lr': 0.007427791712711574, 'lambda_temporal': 0.787775121005695, 'retain_graph': False, 'hidden_feats': 128, 'out_feats': 19, 'dropout': 0.13984655829278747, 'num_layers': 4, 'agg_layer_0': 'pool', 'agg_layer_1': 'gcn', 'agg_layer_2': 'gcn', 'agg_layer_3': 'mean', 'gru_num_layers': 1, 'use_relu': True, 'use_random_noise': True, 'node_type_encoding': 'onehot'}. Best is trial 32 with value: 0.9185270656386892.
    # [I 2025-04-07 04:22:07,609] Trial 42 finished with value: 0.9133571778580173 and parameters: {'num_epochs': 6, 'lr': 0.006945812784688647, 'lambda_temporal': 0.03296639019623693, 'retain_graph': False, 'hidden_feats': 128, 'out_feats': 22, 'dropout': 0.1355732859609593, 'num_layers': 4, 'agg_layer_0': 'mean', 'agg_layer_1': 'pool', 'agg_layer_2': 'gcn', 'agg_layer_3': 'mean', 'gru_num_layers': 1, 'use_relu': False, 'use_random_noise': False, 'node_type_encoding': 'onehot'}.
    # [I 2025-04-07 04:27:32,869] Trial 48 finished with value: 0.9032565491196316 and parameters: {'num_epochs': 9, 'lr': 0.00032802835555538744, 'lambda_temporal': 0.8219352110115995, 'retain_graph': False, 'hidden_feats': 128, 'out_feats': 24, 'dropout': 0.16360877267839466, 'num_layers': 3, 'agg_layer_0': 'lstm', 'agg_layer_1': 'gcn', 'agg_layer_2': 'gcn', 'gru_num_layers': 1, 'use_relu': True, 'use_random_noise': False, 'node_type_encoding': 'onehot'}. Best is trial 32 with value: 0.9185270656386892.
    # Timing:
    # [Trial  0] 2025-04-07 02:46:31
    # [Trial 10] 2025-04-07 03:23:29
    # [Trial 48] 2025-04-07 04:27:32
    pass


def extract_best_params(
    df=None,
    f_debug=False,
    return_embeddings=False,
    embedding_store=None,
    trial_name=None,
    best_params=None,
):
    """
    Extracts the best parameters from the Optuna study and prepares them for model training.
    Args:
        df (pd.DataFrame): DataFrame containing the input data.
        f_debug (bool): Flag for debugging mode.
        embedding_store: Placeholder for embedding store (not used here).
        best_params (dict): Dictionary containing the best parameters from the Optuna study.
    Returns:
        dict: Dictionary containing the parameters for model training.
    """
    kwargs = {
        "df": df,
        "f_debug": f_debug,
        "return_embeddings": return_embeddings,
        "embedding_store": embedding_store,
        "trial_name": trial_name if bool(trial_name) else "Trial",
        "trial": SimpleNamespace(
            number="deployment"
        ),  # no active Optuna trial here, passing a placeholder
        "num_epochs": best_params["num_epochs"],
        "lambda_temporal": best_params["lambda_temporal"],
        "use_retain_graph": best_params["retain_graph"],
        "hidden_feats": best_params["hidden_feats"],
        "out_feats": best_params["out_feats"],
        "use_random_noise": best_params["use_random_noise"],
        "node_type_encoding": best_params["node_type_encoding"],
        "node_type_dim": best_params.get("node_type_dim", 0),
        "lr": best_params["lr"],
        "dropout": best_params["dropout"],
        "num_layers": best_params["num_layers"],
        "agg_list": [
            best_params[f"agg_layer_{i}"] for i in range(best_params["num_layers"])
        ],
        "gru_num_layers": best_params["gru_num_layers"],
        "use_relu": best_params["use_relu"],
    }

    return kwargs


def model_evaluation(
    return_embeddings, embedding_store, embeddings, num_countries, y, f_pc1_corr=False
):
    """
    Evaluate the model using linear regression and return the R^2 score.
    """
    if return_embeddings:
        pass
    else:
        embeddings = embedding_store.load()

    country_emb = embeddings[:num_countries]
    country_emb_np = country_emb.numpy()

    model = LinearRegression()
    model.fit(country_emb_np, y)
    r2_score = model.score(country_emb_np, y)
    print(f"Linear Regression R² Score:")
    print(f"   R² = {r2_score:.9f}")

    if f_pc1_corr:
        # --------- PCA Correlation (MoR-style) ---------
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(country_emb_np).squeeze()

        correlation, p_value = pearsonr(pc1, y)
        print(f"PCA First Component Correlation with GDP:")
        print(f"   Pearson r = {correlation:.4f}, p = {p_value:.4f}")

    pass


# def get_optuna_study_name(
#     prefix="sage-mor",
#     top_features=None,
#     target_score=None,
#     year=None,
#     f_init=None,
#     version=None,
# ):
#     """
#     Generate a study name based on the provided parameters.
#     Args:
#         prefix (str): The prefix for the study name.
#         top_features (list): List of top features.
#         target_score (str): The target score.
#         year (int): The year for the study.
#         f_init (bool): Flag indicating if it's an initial study.
#         version (str): The version of the study.
#     Returns:
#         str: The generated study name.
#     """
#     if f_init:
#         postfix = "init"
#     else:
#         postfix = f"train-v{version}"

#     study_name = f"{prefix}-{'-'.join(top_features)}-t_{target_score}-{year}-{postfix}"

#     return study_name


# def get_optuna_storage_uri(study_name, os_path, optuna_study_location):
#     """
#     Get the SQLite storage URI for Optuna.
#     """
#     db_name = ".".join([study_name, "db"])
#     print(f"Optuna db name: {db_name}")

#     # Full path to the SQLite file (e.g., study name = "graphsage_v1")
#     db_path = os_path.join(optuna_study_location, db_name)
#     # Normalize the path for SQLite URI format
#     db_path = os_path.abspath(db_path).replace("\\", "/")

#     # Convert to SQLite URI format
#     storage_uri = f"sqlite:///{db_path}"
#     print(f"Optuna storage location:\n\t{storage_uri}")

#     return storage_uri


def get_study_trials_df(study=None, f_verbose=True):
    """
    Get the trials DataFrame from the Optuna study.
    """
    if study is None:
        return None

    if f_verbose:
        print(f"Study name: {study.study_name}")
        # print(f"Study best value: {study.best_value}")
        print(
            f"Best trial Number: {study.best_trial.number}. Value: {study.best_value:.5f}. Duration: {study.best_trial.duration} seconds"
        )

    study_trials_df = study.trials_dataframe(attrs=("value", "duration")).sort_values(
        "value", ascending=False
    )
    # print(f"Trials DataFrame:\n{trials_df}")

    return study_trials_df


def get_study_params_by_trial_number(study=None, trial_number=None):
    """
    Get the parameters of a trial by its number.
    """
    if study is None:
        return None

    params = study.trials[trial_number].params
    # print(f"Trial {trial_number} parameters: {params}")

    return params


def get_study_best_params(study=None):
    """
    Get the parameters of the best trial.
    """
    if study is None:
        return None

    best_params = study.best_params
    # print(f"Best trial parameters: {best_params}")

    return best_params
