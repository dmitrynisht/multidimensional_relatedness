from n15_import import *

import os

# Updating modules with the latest changes in case they were modified
from importlib import reload

import p10_Chroma_dbm, p11_Optuna, p12_Utils, p14_GraphSAGE_v03_dynamic_chrdb, p14_GraphSAGE_v_evaluate, p14_GraphSAGE_v_visual, p14_GraphSAGE_v04_dynamic_optuned, p14_GraphSAGE_v05_static_optuned

reload(p10_Chroma_dbm)
get_chroma_client = p10_Chroma_dbm.get_chroma_client
store_final_embeddings = p10_Chroma_dbm.store_final_embeddings
load_all_embeddings = p10_Chroma_dbm.load_all_embeddings
ChromaEmbeddingStore = p10_Chroma_dbm.ChromaEmbeddingStore

reload(p11_Optuna)
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

reload(p12_Utils)
from p12_Utils import (
    get_optuna_study_name, 
    get_optuna_storage_uri,
    set_seed
)

reload(p14_GraphSAGE_v03_dynamic_chrdb)
from p14_GraphSAGE_v03_dynamic_chrdb import train_TemporalSAGE_pure_vdb

reload(p14_GraphSAGE_v_evaluate)
from p14_GraphSAGE_v_evaluate import (
    evaluate_embedding_pc1_vs_gdp,
    evaluate_embedding_npcs_vs_gdp,
    evaluate_embedding_pc5_vs_gdp,
    get_outlier_embeddings,
    get_top_n_embeddings,
    get_embeddings_correlations_df,
    train_regressions_on_embeddings,
)

reload(p14_GraphSAGE_v_visual)
from p14_GraphSAGE_v_visual import (
    visualize_embeddings,
    visualize_embeddings_2PCA,
    lineplot_embedding_correlations_vs_targets,
    lineplot_embedding_correlations_faceted,
    visualize_embeddings_with_umap,
    visualize_embeddings_with_umap_v2,
)

reload(p14_GraphSAGE_v04_dynamic_optuned)
from p14_GraphSAGE_v04_dynamic_optuned import (
    objective_r2,
    objective_corr,
    objective_corr_sye,
    optuna_train_TemporalSAGE_pure_vdb,
    optuna_train_TemporalSAGE_pure_vdb_sye,
)

# For model evaluation
from p14_GraphSAGE_v04_dynamic_optuned import (
    eval_TemporalSAGE_pure_vdb,
    get_oredered_node_ids,
)

from p14_GraphSAGE_v05_static_optuned import (
    get_graph_statistics,
)

print(f"os.getcwd():\n{os.getcwd()}", "=" * 60, sep="\n")
print("input_file_location:", input_file_location, "=" * 60, sep="\n")
print("output_file_location:", output_file_location, "=" * 60, sep="\n")
print("vectordb_location:", vectordb_location, "=" * 60, sep="\n")

pass
