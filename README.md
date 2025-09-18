# A Multidimensional Approach to the Principle of Relatedness

## Abstract

Despite recent progress in economic complexity research, traditional measures such as the Economic Complexity Index (ECI) compress rich trade structures into single scalar values, potentially limiting interpretability and expressiveness. This thesis explores whether graph-based representation learning can capture the structural heterogeneity of international trade networks more effectively. We propose a bipartite graph neural network (GNN) framework trained on country–product export data. The resulting node embeddings are evaluated against established socio-economic indicators, with embedding quality assessed using both unsupervised techniques and regression-based metrics. 

**Key Finding**: Results show that the learned embeddings only weakly correlate with conventional indicators such as GDP per capita or Human Capital Index and fail to outperform the ECI benchmark. However, the framework demonstrates flexibility, reproducibility, and potential for future extensions.

## Research Contribution

This work provides:
- **Methodological Framework**: First systematic application of GraphSAGE to economic complexity analysis
- **Empirical Evidence**: Demonstrates limitations of current GNN approaches for economic development modeling  
- **Reproducible Pipeline**: Complete workflow from raw trade data to embedding evaluation
- **Technical Insights**: Identifies topological feature design as a critical bottleneck for performance
- **Future Research Foundation**: Clear directions for improvement and extension

## Key Findings

### Primary Results
- **Weak Correlations**: GNN embeddings show minimal alignment with socio-economic indicators (GDP, GINI, HCI)
- **Benchmark Comparison**: Failed to outperform traditional Economic Complexity Index (ECI)
- **Evaluation Methods**: Consistent poor performance across PCA, UMAP, HDBSCAN, and regression analysis
- **R² Values**: Negative across nearly all indicators and models, indicating limited explanatory power

### Technical Insights
- **Topological Feature Limitation**: Poor performance likely stems from inadequate topological feature design
- **Mono-partite vs Bipartite**: Current features (node degree, centrality, HITS, PageRank) may not capture bipartite network structure effectively
- **Feature Engineering**: Generic graph metrics do not transfer well to economic trade networks

## Methodology Overview

### Graph Neural Network Architecture
- **Model**: GraphSAGE with unsupervised learning (Hamilton et al., 2017)
- **Network Structure**: Bipartite graph (countries ↔ products)
- **Training Approach**: Method of Reflections - single-year network snapshots
- **Loss Function**: Unsupervised contrastive loss with negative sampling (5 negatives per positive edge)

### Topological Features Used
- `nd`: Node degree (number of neighbors)
- `cd`: Centrality degree (closeness centrality)
- `ht`: HITS score
- `pr`: PageRank score

### Dimensional Features Used
- `ph`: Product hierarchy features
  - Encodes HS classification structure (X.XXX.YYYY format)
  - Captures two meaningful hierarchy levels from product taxonomy
  - Currently the only dimensional feature used, but framework supports expansion

### Evaluation Framework
- **Linear Projection**: PCA correlation analysis with socio-economic indicators
- **Nonlinear Clustering**: UMAP + HDBSCAN for economic grouping validation
- **Regression Analysis**: SVR and ElasticNet for predictive performance assessment
- **Benchmark Comparison**: Against Economic Complexity Index (ECI)

## Requirements

### System Requirements
- **OS**: Windows 11 (tested)
- **Python**: 3.11.11 (tested)
- **Hardware**: CPU sufficient

### Key Dependencies
- **torch==2.1.0+cpu** (latest Windows-supported version)
- **dgl==2.2.1** (latest Windows-supported version)
- **ChromaDB**: Vector database for embedding storage
- **Optuna**: Hyperparameter optimization
- **NetworkX, pandas**: Graph and data manipulation

### Installation
```bash
# Clone repository
git clone [repository-url]
cd multidimensional_relatedness

# Install dependencies
pip install -r requirements.txt

# Setup environment
python config.py

# Create ChromaDB directory
mkdir data/10_chromadb
```

## Usage

### Data Requirements
The research pipeline requires two main trade data files from Harvard Dataverse:
- `hs92_country_country_product_year_4_2010_2014.dta`
- `hs92_country_country_product_year_4_2015_2019.dta`

**Source**: The Growth Lab at Harvard. (2025). International Trade Data (HS, 92) (Version 15.0) [Dataset]. Harvard Dataverse. https://doi.org/10.7910/DVN/T4CHWJ

These files must be placed in the `data/01_raw/` directory for the pipeline to function correctly. All data sources used in this research are properly referenced in the thesis paper.

### Directory Setup
The `data/10_chromadb/` directory must be created manually before running any experiments. This folder is excluded from git tracking (`.gitignore`) because ChromaDB generates files with generic names that don't clearly identify which experiment they belong to, making version control impractical.

```bash
# Create ChromaDB directory (Linux/macOS)
mkdir -p data/10_chromadb

# Create ChromaDB directory (Windows)
mkdir data\10_chromadb
```

This directory will store vector database files generated during pipeline execution.

### Data Pipeline Execution
All processing is executed through Jupyter notebooks following the order specified in `dataworkflow.xlsx`:

1. **Data Ingestion** (n01-n02, n07): Raw trade data, GDP, population data, social indices
2. **Preprocessing** (n03): Data cleaning, validation
3. **Economic Analysis** (n04-n06): Method of Reflections, ECI calculation  
4. **GNN Training** (n15): GraphSAGE model training and evaluation

### Main Experiment
```bash
# Primary training and evaluation notebook
jupyter notebook notebooks/n15_GraphSAGE_mor_training_nd-cd-ht-pr_features+dims+u_loss.ipynb
```

### Key Execution Files
- `dataworkflow.xlsx`: Execution roadmap with input/output mappings
- `config.py`: Environment setup and path configuration
- Main experiment: `n15_GraphSAGE_mor_training_nd-cd-ht-pr_features+dims+u_loss.ipynb`

## Project Structure

```
├── data/
│   ├── 01_raw/              # Raw economic data (World Bank, trade data)
│   ├── 02_ingested/         # Processed input data
│   ├── 03_preprocessed/     # Cleaned datasets
│   ├── 04_mor_implemented/  # Method of Reflections results
│   ├── 05_mor_silver/       # Refined complexity indices
│   ├── 06_mor_gold/         # Final ECI results
│   ├── 07_sindices_processed/ # Social indices (HDI, GINI, HCI)
│   ├── 10_chromadb/         # ChromaDB vector database files (created during pipeline execution)
│   ├── 11_optuna_studies/   # Hyperparameter optimization studies
│   └── 15_SAGE_gold/        # GraphSAGE model outputs
├── src/
│   ├── p01-p14*.py         # Processing modules by pipeline stage
│   ├── p10_Chroma_dbm.py   # ChromaDB integration
│   ├── p14_GraphSAGE_v*.py # GraphSAGE implementations
│   └── imports/            # Import scripts for notebook integration
│       └── n*_import*.py   # Notebook-specific import modules
├── notebooks/
│   ├── n01-n15*.ipynb      # Analysis notebooks
│   └── config_notebooks.py # Notebook configuration
├── dataworkflow.xlsx       # Execution roadmap
├── config.py              # Environment setup
└── requirements.txt       # Dependencies
```

## Reproducibility

### Complete Pipeline Reproduction
1. Follow `dataworkflow.xlsx` execution order
2. Run notebooks n01 through n15 sequentially
3. Main results generated by `n15_GraphSAGE_mor_training_nd-cd-ht-pr_features+dims+u_loss.ipynb`

### Notebook Execution Considerations
Many Jupyter notebooks contain intentional `assert False` statements that were added during the research phase to enable selective execution of specific cell ranges. While this approach was beneficial for iterative development and testing, it prevents the use of "Run All" functionality in Jupyter notebooks. Users reproducing the pipeline will need to manually execute cells using "Run Current Cell and Below" or comment out these assertions as needed. The number of `assert False` statements varies by notebook depending on the research workflow requirements.

### Key Configuration
Main experiment configuration used for thesis results:
- **Years**: 2012-2019 (8-year analysis window)
- **Countries**: 144 countries with reliable trade data
- **Products**: 1178 products (HS92 4-digit product classification)
- **Hyperparameter Optimization**: Optuna across multiple years
- **Embedding Storage**: ChromaDB vector database

## Future Work Directions

### Methodological Improvements
1. **Bipartite-Specific Features**: Replace mono-partite topological metrics with bipartite network measures
2. **Hierarchical Modeling**: Enhance product taxonomy utilization beyond current HS classification structure. Products in the current dataset exhibit similar complexity levels, limiting the discriminative power of hierarchical features. This area requires more attention to effectively leverage multi-level product categorization for economic complexity modeling
3. **Feature Enrichment**: Integrate institutional, innovation, or geographical variables
4. **Advanced Loss Functions**: Experiment with different unsupervised objectives beyond neighborhood prediction
5. **Temporal Dynamics**: Train GNNs across multiple years rather than single-year snapshots

## References

- Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. NeurIPS.
- Hidalgo, C. A., & Hausmann, R. (2009). The building blocks of economic complexity. PNAS.
- McInnes, L., Healy, J., & Melville, J. (2020). UMAP: Uniform manifold approximation and projection for dimension reduction.

## Citation

If you use this framework or build upon this research, please cite:
```
[Your thesis citation details]
```

## Contact

[
    Dzmitry Nisht,
    dzmitrynisht@hotmail.com,
    https://www.linkedin.com/in/dzmitry-nisht/,
    +351939863502,
]

---

**Notes**:
- This research demonstrates that current GNN approaches require significant methodological refinement to effectively model economic complexity. The negative results provide valuable insights for future research directions in the intersection of machine learning and development economics.

- This repository includes a supplementary file named `Claude.md`.  
The content of this file did not contribute to the research design, analysis, or results presented in the thesis. It was created only after the research work had been completed, as part of an exploratory exercise to examine how an external AI assistant (Claude) interprets and structures project-related information.  

- All scientific contributions, methodologies, and findings contained in this repository are solely the work of the author. The inclusion of `Claude.md` is intended exclusively for transparency and to provide potential guidance for future researchers interested in experimenting with AI-assisted project documentation. 