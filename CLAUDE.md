# Claude Code Context: Multidimensional Approach to the Principle of Relatedness Research Project

## Project Overview
This is a Master's thesis project focused on **multidimensional relatedness** using economic complexity analysis combined with Graph Neural Networks (GraphSAGE) and vector databases (ChromaDB). The research analyzes economic relationships between countries and products using both traditional Method of Reflections (MOR) and modern GNN approaches.

## Architecture & Technologies
- **Graph Neural Networks**: GraphSAGE for learning embeddings with topological features
- **Vector Database**: ChromaDB for persistent storage of node embeddings
- **Economic Analysis**: Method of Reflections (MOR) for complexity indices
- **Optimization**: Optuna for hyperparameter tuning
- **Core Stack**: PyTorch/DGL, pandas, NetworkX, Jupyter notebooks

## Data Pipeline Structure
**All data processing is executed through Jupyter notebooks** - not just experiments but the entire pipeline.

### Pipeline Stages (01-14)
- `01_raw/` - Raw economic data (exports, GDP, population, social indices)
- `02_ingested/` - Processed input data  
- `03_preprocessed/` - Cleaned and validated datasets
- `04_mor_implemented/` - Method of Reflections results
- `05_mor_silver/` - Refined complexity indices
- `06_mor_gold/` - Final validated MOR results
- `07_sindices_processed/` - Social indices (HDI, GINI, HCI)
- `11_optuna_studies/` - Hyperparameter optimization studies
- `15_SAGE_gold/` - GraphSAGE model outputs

### Execution Management
- **dataworkflow.xlsx**: Personal reference guide containing:
  - `order`: Execution sequence for jupyter files
  - `file`: Corresponding jupyter notebook names
  - `input`/`output`: Folder and file names for I/O data
  - `flow`: Approximate order of variables created (precise for orders 1-8, less precise for order 9+)

## Current Main Experiment
**File**: `n15_GraphSAGE_mor_training_nd-cd-ht-pr_features+dims+u_loss.ipynb`

### Naming Convention Breakdown
- `nd-cd-ht-pr`: Topological features used for GNN
  - `nd`: node degree (number of neighbors)
  - `cd`: centrality degree (closeness centrality)
  - `ht`: HITS score
  - `pr`: PageRank score
- `dims`: Dimensional features also used (currently only 'ph' - Product hierarchy, expandable in future)
- `u_loss`: Unsupervised loss function variant

### Loss Function Variants
- `0_loss`: Loss not computed (returns zeros)
- `a_loss`: Loss computed but no minimum selection (last epoch embeddings)
- `m_loss`: Loss computed with minimum selection
- `u_loss`: Unsupervised loss as per Hamilton et al. (2017) GraphSAGE paper (**used in thesis results**)

## Code Organization

### Source Modules (`src/`)
- `p01-p14`: Processing modules organized by pipeline stage
- `p10_Chroma_dbm.py`: ChromaDB wrapper (`ChromaEmbeddingStore` class)
- `p14_GraphSAGE_v*`: GraphSAGE implementation variants
- `imports/`: Import modules for notebook integration

### Notebooks (`notebooks/`)
- `n01-n14`: Analysis notebooks corresponding to pipeline stages
- `config_notebooks.py`: Notebook configuration
- **Main experiment**: `n15_GraphSAGE_mor_training_nd-cd-ht-pr_features+dims+u_loss.ipynb`

## Key Economic Indicators
- **ECI**: Economic Complexity Index
- **PCI**: Product Complexity Index  
- **RCA**: Revealed Comparative Advantage
- **Social Indices**: HDI, GINI coefficient, Human Capital Index

## ChromaDB Integration
- Persistent storage of PyTorch tensor embeddings
- Vector similarity search capabilities
- Metadata tracking for node IDs
- Integration with dynamic GraphSAGE training pipeline
- Collection management for different model variants

## Development Context
- **Research Focus**: Economic complexity through network relationships
- **Methodology**: Combining traditional economics methods with modern ML
- **Data Sources**: International trade data, World Bank indicators, UNDP indices
- **Output**: Country and product embeddings for complexity analysis

## Core Function Architecture

### Primary Training Functions (src/p11_Optuna.py)

#### `run_tuning_steps` (lines 453-573)
**Purpose**: Hyperparameter optimization across multiple years using Optuna

**Key Parameters**:
- `target_score`: Determines loss function variant
  - `"r2"`: Uses `objective_r2` (default, with `["nd"]` features)
  - `"corr"`: Uses `objective_corr` (with `["nd", "cd"]` features)  
  - `"corr_sye_u_loss"`: **Uses `objective_corr_sye_u_loss` with unsupervised loss** (current implementation)
- `n_years`: Number of years to process from `rolling_exports_years`
- `rolling_exports_years`: Year range (typically 2012-2019)

**Implementation Flow**:
1. Creates separate Optuna studies for each year
2. Uses best hyperparameters from previous year as initial hyperparameters for next year
3. Returns dictionary of studies indexed by year (`studies_by_year`)

#### `run_training_evaluation_steps` (lines 377-450)
**Purpose**: Final model training using best hyperparameters and evaluation

**Two-Phase Process**:
1. **Training Phase** (`training_models__storing_embeddings`): 
   - Uses best hyperparameters from tuning
   - Trains final models and stores embeddings in ChromaDB
2. **Evaluation Phase** (`embeddings_evaluation_visualization`):
   - Evaluates embeddings against economic indicators
   - Creates UMAP visualizations and correlation analysis

**Returns**: `(full_correlations_df, outlier_embedding_countries, reg_scores_df, umap_projections_df)`

### Unsupervised Loss Implementation (src/p14_GraphSAGE_v05_static_optuned.py)

#### `objective_corr_sye_u_loss` (lines 338-473)
**Core unsupervised learning objective function**

**Loss Computation Flow**:
1. **Training Function**: `optuna_train_sye_sage_pure_conv_u_loss`
   - Creates `GraphSAGEUnsupervisedLoss(num_negative=5)` instance
   - Implements contrastive learning with negative sampling

2. **Loss Calculation** (`compute_static_consistency_loss_v1`):
   ```python
   # Extract positive edges from graph
   src_pos, dst_pos = graph.edges()
   
   # Sample 5 negative edges per positive edge
   neg_dst = torch.randint(low=0, high=embeddings.shape[0], 
                          size=(src_pos.shape[0], 5))
   
   # Compute unsupervised loss using negative sampling
   loss = unsup_loss(z_u, z_v, z_neg)
   ```

3. **Evaluation Objective**:
   - Uses PCA to reduce country embeddings to 2 components
   - Computes Pearson correlation between first PC and GDP per capita
   - Maximizes absolute correlation: `score = abs(corr)`

#### GraphSAGE Unsupervised Learning Principles
- Follows Hamilton et al. (2017) approach
- **Contrastive Learning**: Positive edges should have similar embeddings, negative edges should be dissimilar
- **Negative Sampling**: 5 random negative samples per positive edge
- **Objective**: Learn embeddings where economically related countries cluster together

### Notebook Integration Flow
1. **Initial Tuning**: `init_studies_by_year = run_tuning_steps(target_score="corr_sye_u_loss", ...)`
2. **Training Tuning**: `training_studies_by_year = run_tuning_steps(...)` with refined hyperparameters
3. **Final Training & Evaluation**: `run_training_evaluation_steps(...)` → ChromaDB storage → correlation analysis

## Important Files for Reference
- `dataworkflow.xlsx`: Execution roadmap and I/O mapping
- `config.py`: Environment setup and path management
- `requirements.txt`: Dependencies including chromadb, torch, optuna
- `README.md`: Project documentation with loss variant explanations