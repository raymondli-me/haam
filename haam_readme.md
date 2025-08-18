# HAAM - Human-AI Accuracy Model Analysis Package

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package implementing the Double Machine Learning Lens Model Equation (DML-LME) for analyzing how humans and AI achieve perceptual accuracy in high-dimensional settings.

## 🎯 What is HAAM?

HAAM implements the Human-AI Accuracy Model framework for understanding how humans and AI systems use information when making judgments. It answers questions like:

- What features (principal components) do humans vs AI rely on?
- How much do human and AI judgments align with ground truth?
- What semantic topics characterize the information being used?
- How can we visualize these relationships interactively?

### Key Features

- 📊 **Cross-fitted post-lasso regression** for debiased coefficient estimation
- 🔍 **Automatic topic modeling** to interpret principal components using c-TF-IDF
- 📈 **Interactive visualizations** including framework diagrams, coefficient grids, and 3D UMAP plots
- 💾 **Comprehensive metrics export** including top/bottom topics for all 200 PCs
- 🔄 **R compatibility** via reticulate

### Recent Updates (v1.1.0)

- **Enhanced UMAP parameters**: Now uses `n_neighbors=5`, `min_dist=0.0`, `metric='cosine'` for better clustering
- **Improved HDBSCAN clustering**: `min_cluster_size=10`, `min_samples=2` matching BERTopic-style parameters
- **c-TF-IDF implementation**: Uses the BERTopic formula for better topic extraction
- **Generic criterion labeling**: All "SC" references changed to "Y" for flexibility
- **Dynamic metrics calculation**: R², PoMA, and other metrics calculated from actual data
- **Manual PC naming**: New `pc_names` parameter for custom PC labels
- **Metrics summary export**: New `create_metrics_summary()` method exports comprehensive JSON

## 🚀 Installation

### Quick Install (when package is published)

```bash
pip install haam
```

### Install from GitHub

```bash
pip install git+https://github.com/raymondli-me/haam.git
```

### Local Development Install

```bash
git clone https://github.com/raymondli-me/haam.git
cd haam
pip install -e .
```

### Dependencies

Core requirements:
- `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `scipy`
- `matplotlib`, `seaborn`, `plotly` (visualizations)
- `sentence-transformers` (embeddings)
- `umap-learn`, `hdbscan` (dimensionality reduction & clustering)

## 📖 Quick Start

### Basic Usage

```python
from haam import HAAM

# Your data
criterion = [...]        # Ground truth (e.g., social class)
ai_judgment = [...]      # AI predictions/ratings
human_judgment = [...]   # Human ratings
texts = [...]           # Text documents (optional if you have embeddings)

# Run analysis with one line
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=texts,
    n_components=200
)

# Export all results
haam.export_all_results('./output')
```

### Example with Custom PC Names

```python
# Create visualizations with manual PC naming
pc_names = {
    0: "Formality",      # PC1
    1: "Complexity",     # PC2  
    3: "Sentiment",      # PC4
    6: "Education"       # PC7
}

# Create main visualization with custom names
haam.create_main_visualization(pc_names=pc_names)

# PCs without custom names will show "-" in the visualization
```

### Google Colab Example

```python
# Install
!pip install sentence-transformers umap-learn hdbscan plotly

# Mount Drive and load data
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/your_data.csv')

# Run analysis
from haam import HAAM

haam = HAAM(
    criterion=df['target_variable'],  # Any criterion variable
    ai_judgment=df['ai_rating'],
    human_judgment=df['human_rating'],
    texts=df['text_content'].tolist()
)

# Export results including metrics summary
haam.export_all_results('/content/drive/MyDrive/haam_output')

# Get comprehensive metrics
metrics = haam.create_metrics_summary()
print(f"AI Model R² (CV): {metrics['model_performance']['AI']['r2_cv']:.3f}")
print(f"Human Model R² (CV): {metrics['model_performance']['HU']['r2_cv']:.3f}")
```

## 🔧 Core Functions

### Main Analysis Class

```python
# Initialize and run analysis
haam = HAAM(
    criterion,           # Ground truth variable
    ai_judgment,         # AI predictions
    human_judgment,      # Human ratings
    texts=None,          # Optional: texts for topic modeling
    embeddings=None,     # Optional: pre-computed embeddings
    n_components=200,    # Number of PCA components
    auto_run=True        # Run full pipeline automatically
)
```

### Key Methods

- **`export_all_results(output_dir)`** - Export all results and visualizations
- **`create_main_visualization(pc_names=None)`** - Generate interactive HAAM framework diagram
- **`create_mini_grid()`** - Create coefficient grid for all 200 PCs
- **`explore_pc_topics(pc_indices)`** - Explore topic associations for specific PCs
- **`plot_pc_effects(pc_idx)`** - Bar chart showing PC effects on outcomes
- **`create_umap_visualization(color_by='Y')`** - 3D UMAP visualization
- **`create_metrics_summary()`** - Export comprehensive metrics as JSON

### Topic Analysis

```python
# Explore topics for top PCs with enhanced c-TF-IDF
pc_topics = haam.explore_pc_topics(
    pc_indices=[4, 7, 1, 5, 172],  # 0-based indexing
    n_topics=10
)

# Get high/low topics for a specific PC
topics = haam.topic_analyzer.get_pc_high_low_topics(
    pc_idx=4,    # PC5 (0-based)
    n_high=5,    # Top 5 high topics
    n_low=5      # Top 5 low topics
)

# Topics now use BERTopic-style c-TF-IDF for better quality
```

### Visualization Options

```python
# Create all visualizations with custom PC names
pc_names = {
    0: "Writing Style",
    3: "Technical Terms",
    6: "Emotional Tone"
}

haam.create_main_visualization(pc_names=pc_names)  # Shows custom names
haam.create_mini_grid()                           # Grid showing all 200 PCs

# UMAP with different coloring (now uses Y instead of SC)
haam.create_umap_visualization(color_by='Y')    # Color by criterion
haam.create_umap_visualization(color_by='AI')   # Color by AI judgment
haam.create_umap_visualization(color_by='PC1')  # Color by PC1 scores

# PC effects bar chart
fig = haam.plot_pc_effects(pc_idx=4)  # For PC5 (0-based)
```

### Metrics Export

```python
# Create comprehensive metrics summary
metrics = haam.create_metrics_summary(output_file='metrics.json')

# Access specific metrics
print(f"Y Model Performance: R²={metrics['model_performance']['Y']['r2_cv']:.3f}")
print(f"AI PoMA: {metrics['mediation_analysis']['AI']['proportion_mediated']:.1f}%")
print(f"Human PoMA: {metrics['mediation_analysis']['HU']['proportion_mediated']:.1f}%")
```

## 📊 Output Structure

Running `export_all_results()` creates:

```
output_directory/
├── haam_results_coefficients_[timestamp].csv  # All PC coefficients
├── haam_results_summary_[timestamp].csv      # Model performance metrics
├── pc_topic_exploration.csv                  # Topic associations for all 200 PCs
├── haam_metrics_summary.json                 # Comprehensive metrics export
├── haam_main_visualization.html              # Interactive framework diagram
├── haam_mini_grid.html                       # Coefficient grid (200 PCs)
├── haam_umap_3d_Y.html                      # UMAP colored by criterion
├── haam_umap_3d_AI.html                     # UMAP colored by AI
└── haam_umap_3d_HU.html                     # UMAP colored by human
```

## 🔬 Advanced Usage

### Using Pre-computed Embeddings

```python
# If you already have embeddings
embeddings = np.load('embeddings.npy')

haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    embeddings=embeddings,  # Skip text processing
    n_components=200
)
```

### Custom PC Selection

```python
# Different ranking methods for selecting top PCs
top_pcs_triple = haam.analysis.get_top_pcs(ranking_method='triple')  # Default
top_pcs_by_ai = haam.analysis.get_top_pcs(ranking_method='AI')      # By AI importance
top_pcs_by_human = haam.analysis.get_top_pcs(ranking_method='HU')   # By human importance
```

### Access Detailed Results

```python
# Coefficient DataFrame
coef_df = haam.results['coefficients']

# Model performance
summary = haam.results['model_summary']

# Access specific model results (Y is the generic criterion)
y_results = haam.analysis.results['debiased_lasso']['SC']  # Internal still uses SC
print(f"Y R² (CV): {y_results['r2_cv']:.4f}")
print(f"Selected PCs: {y_results['selected']}")

# Get topic information for all PCs
all_topics = haam.results.get('all_pc_topics', {})
```

## 🌉 R Integration

Use HAAM from R via reticulate:

```r
library(reticulate)

# Install HAAM
py_install("git+https://github.com/raymondli-me/haam.git")

# Import
haam <- import("haam")

# Run analysis
analysis <- haam$HAAM(
  criterion = criterion_vector,
  ai_judgment = ai_vector,
  human_judgment = human_vector,
  texts = as.list(text_vector),
  n_components = 200L
)

# Create visualizations with custom PC names
pc_names <- list(
  "0" = "Formality",
  "3" = "Complexity"
)
analysis$create_main_visualization(pc_names = pc_names)

# Export results
analysis$export_all_results("./output")

# Access results in R
model_summary <- analysis$results$model_summary
coefficients <- py_to_r(analysis$results$coefficients)
```

## 📈 Interpreting Results

### Key Metrics

- **R² (CV)**: Cross-validated R-squared - use this over in-sample R²
- **N_selected**: Number of PCs selected by lasso
- **Coefficients**: Standardized effect of each PC on outcomes
- **PoMA**: Proportion of Modeled Accuracy (mediation through PCs)

### Understanding PCs

Each PC is characterized by:
- **HIGH topics**: Topics/keywords associated with high PC values (using c-TF-IDF)
- **LOW topics**: Topics/keywords associated with low PC values
- **Coefficients**: How strongly the PC predicts each outcome (Y, AI, Human)
- **Names**: Optional custom labels for interpretability

### Visualizations Guide

1. **Main Visualization**: Shows top 9 PCs as mediators between criterion and judgments
   - Now displays "Y" instead of "SC" for generic criterion
   - Shows custom PC names when provided, otherwise shows "-"
   - Dynamically calculates R², PoMA, and other metrics

2. **Mini Grid**: Overview of all 200 PCs - size indicates magnitude, color indicates sign

3. **UMAP**: Shows how outcomes vary in the embedding space
   - Enhanced parameters for better clustering visualization
   - Can color by Y (criterion), AI, or Human judgments

## 🐛 Troubleshooting

### Common Issues

**Module not found:**
```python
import sys
sys.path.append('/path/to/haam')
```

**Memory issues with large datasets:**
```python
# Use fewer components
haam = HAAM(..., n_components=100)

# Or pre-compute embeddings
embeddings = HAAMAnalysis.generate_embeddings(texts)
```

**No topics in visualization:**
```python
# Make sure to provide texts, not just embeddings
haam = HAAM(..., texts=your_texts)
```

## 📝 Citation

If you use HAAM in your research, please cite:

```bibtex
@article{li2025dml,
  title = {High-Dimensional Perception with the Double Machine Learning Lens Model Equation (DML-LME)},
  author = {Li, Raymond V. and Biesanz, Jeremy C.},
  journal = {Psychometrika},
  year = {2025}
}

@software{haam_package,
  title = {HAAM: Human-AI Accuracy Model Analysis Package},
  author = {Li, Raymond V. and Biesanz, Jeremy C.},
  year = {2025},
  url = {https://github.com/raymondli-me/haam}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💬 Contact

For questions or issues, please open an issue on [GitHub](https://github.com/raymondli-me/haam/issues).