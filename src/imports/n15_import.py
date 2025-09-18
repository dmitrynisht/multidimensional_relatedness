import sys
import os
import gc  # garbage collector
import pandas as pd
import dgl
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from dgl.nn import SAGEConv

# for calculating the correlation coefficient R-squared
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Path to 'src' folder is defined by 'config' and 'config_notebooks' files
from p01_raw_data_imports import read_hs92_parquet_data, save_hs92_parquet_data
from p01_raw_data_imports import read_hs92_stata_data, save_hs92_stata_data
from p01_raw_data_imports import switch_to_dtype, custom_float_formatter, df_stats
from p01_raw_data_imports import restore_cache, staging_cache

# from p06_mor_analysis import calculate_r_squared, calculate_yearly_eci_gdp_correlation, check_global_eci_convergence
# from p06_mor_analysis import scatterplot_ECI_vs_GDP, lineplot_ECI_vs_GDP_correlation, plot_eci_convergence

from p10_Chroma_dbm import (
    ChromaEmbeddingStore,
    get_chroma_client,
    store_final_embeddings,
    load_all_embeddings,
)
from p11_Optuna import (
    load_or_create_optuna_study,
    get_user_attrs_hyperparams,
    run_batch_of_trials,
    extract_best_params,
    model_evaluation,
    run_tuning_steps,
    get_study_trials_df,
    run_training_evaluation_steps,
)
from p12_Utils import (
    get_optuna_study_name, 
    get_optuna_storage_uri,
    set_seed,
)
from p14_GraphSAGE_v03_dynamic_chrdb import train_TemporalSAGE_pure_vdb

import optuna
from p14_GraphSAGE_v04_dynamic_optuned import (
    objective_r2,
    objective_corr,
    objective_corr_sye,
    optuna_train_TemporalSAGE_pure_vdb,
    optuna_train_TemporalSAGE_pure_vdb_sye,
)
from types import SimpleNamespace

# For model evaluation
from p14_GraphSAGE_v04_dynamic_optuned import (
    eval_TemporalSAGE_pure_vdb,
    get_oredered_node_ids,
)
from p14_GraphSAGE_v05_static_optuned import (
    get_graph_statistics,
)
from p14_GraphSAGE_v_evaluate import (
    evaluate_embedding_pc1_vs_gdp,
    evaluate_embedding_npcs_vs_gdp,
    evaluate_embedding_pc5_vs_gdp,
    get_outlier_embeddings,
    get_top_n_embeddings,
    get_embeddings_correlations_df,
    train_regressions_on_embeddings,
)
from p14_GraphSAGE_v_visual import (
    visualize_embeddings,
    visualize_embeddings_2PCA,
    lineplot_embedding_correlations_vs_targets,
    lineplot_embedding_correlations_faceted,
    visualize_embeddings_with_umap,
    visualize_embeddings_with_umap_v2,
)

# Assign the custom formatter to Pandas options
pd.options.display.float_format = custom_float_formatter

# Define the relative path to the output data folder
output_file_location = os.path.join("..", "data", "15_SAGE_gold")

# Define the relative path to the input data folder
input_file_location = os.path.join("..", "data", "07_sindices_processed")

# Define the relative path to the vector database data folder
vectordb_location = os.path.join("..", "data", "10_chromadb")
# Ensure the directory exists
os.makedirs(vectordb_location, exist_ok=True)

# Define the relative path to the optuna studies folder
optuna_study_location = os.path.join("..", "data", "11_optuna_studies")
# Ensure the directory exists
os.makedirs(optuna_study_location, exist_ok=True)

print(f"os.getcwd():\n{os.getcwd()}", "=" * 60, sep="\n")
print("input_file_location:", input_file_location, "=" * 60, sep="\n")
print("output_file_location:", output_file_location, "=" * 60, sep="\n")
print("vectordb_location:", vectordb_location, "=" * 60, sep="\n")
print("optuna_study_location:", optuna_study_location, "=" * 60, sep="\n")

pass
