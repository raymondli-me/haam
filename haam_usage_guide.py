"""
HAAM Package - Comprehensive Usage Guide
========================================

This guide shows how to use the HAAM package in various environments.
"""

# =============================================================================
# GOOGLE COLAB QUICK START
# =============================================================================

# Cell 1: Installation
!pip install numpy pandas scikit-learn statsmodels scipy matplotlib seaborn plotly sentence-transformers umap-learn hdbscan

# Cell 2: Load the package files (if not installed via pip)
# Option A: If you have the files in Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy the package files to Colab
!cp -r /content/drive/MyDrive/haam_package /content/

# Add to Python path
import sys
sys.path.append('/content/haam_package')

# Option B: Download directly from GitHub (when available)
# !git clone https://github.com/raymondli-me/haam.git
# sys.path.append('/content/haam')

# Cell 3: Import and use
from haam_main import HAAM
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('/content/drive/MyDrive/your_data.csv')

# Extract variables
criterion = df['social_class'].values
ai_judgment = df['ai_rating'].values
human_judgment = df['human_rating'].values
texts = df['text'].tolist() if 'text' in df else None

# Run analysis
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=texts,
    n_components=200,
    auto_run=True
)

# View results
print("Model Summary:")
print(haam.results['model_summary'])

print("\nTop 9 PCs:")
print(haam.results['top_pcs'])

# Create visualizations
main_viz = haam.create_main_visualization('/content/drive/MyDrive/haam_output')
mini_grid = haam.create_mini_grid('/content/drive/MyDrive/haam_output')

print(f"Main visualization saved to: {main_viz}")
print(f"Mini grid saved to: {mini_grid}")

# Export all results
outputs = haam.export_all_results('/content/drive/MyDrive/haam_output')


# =============================================================================
# DETAILED PYTHON USAGE
# =============================================================================

import numpy as np
import pandas as pd
from haam_main import HAAM

# 1. BASIC USAGE - With text data for topic modeling
# ---------------------------------------------------

# Load data
data = pd.read_csv('your_data.csv')

# Initialize and run analysis
haam = HAAM(
    criterion=data['social_class'],
    ai_judgment=data['ai_rating'], 
    human_judgment=data['human_rating'],
    texts=data['essay_text'].tolist(),
    n_components=200,
    auto_run=True  # Automatically runs full pipeline
)

# Access results
coefficients_df = haam.results['coefficients']  # DataFrame of all PC coefficients
model_summary = haam.results['model_summary']   # R², number selected, etc.
top_pcs = haam.results['top_pcs']              # Indices of top 9 PCs

# Export everything at once
output_paths = haam.export_all_results('./haam_results')


# 2. ADVANCED USAGE - Manual control
# ----------------------------------

# Initialize without auto-run
haam = HAAM(
    criterion=data['social_class'],
    ai_judgment=data['ai_rating'],
    human_judgment=data['human_rating'], 
    texts=data['essay_text'].tolist(),
    n_components=200,
    auto_run=False
)

# Run analysis step by step
results = haam.run_full_analysis()

# Get top PCs using different ranking methods
top_pcs_triple = haam.analysis.get_top_pcs(n_top=9, ranking_method='triple')  # Default
top_pcs_by_ai = haam.analysis.get_top_pcs(n_top=9, ranking_method='AI')      # Ranked by AI
top_pcs_by_human = haam.analysis.get_top_pcs(n_top=9, ranking_method='HU')   # Ranked by Human

# Explore topics for specific PCs
pc_topics = haam.explore_pc_topics(
    pc_indices=[4, 7, 1, 5, 172],  # 0-based indices
    n_topics=20  # Show top 10 and bottom 10 topics
)
print(pc_topics)

# Create individual visualizations
main_viz_path = haam.create_main_visualization('./visualizations')
mini_grid_path = haam.create_mini_grid('./visualizations')

# Create PC effects plot for a specific PC
fig = haam.plot_pc_effects(pc_idx=4, save_path='./pc5_effects.png')

# Create UMAP visualizations with different coloring
umap_sc = haam.create_umap_visualization(n_components=3, color_by='SC')
umap_ai = haam.create_umap_visualization(n_components=3, color_by='AI')
umap_pc1 = haam.create_umap_visualization(n_components=3, color_by='PC1')


# 3. USING PRE-COMPUTED EMBEDDINGS
# ---------------------------------

# If you already have embeddings
embeddings = np.load('embeddings.npy')  # Shape: (n_samples, embedding_dim)

haam = HAAM(
    criterion=data['social_class'],
    ai_judgment=data['ai_rating'],
    human_judgment=data['human_rating'],
    embeddings=embeddings,  # Pass embeddings instead of texts
    n_components=200
)

# Note: Topic analysis won't be available without texts


# 4. WORKING WITH MISSING DATA
# ----------------------------

# HAAM handles missing values in outcomes
criterion = data['social_class'].values  # Can contain NaN
ai_judgment = data['ai_rating'].values   # Can contain NaN  
human_judgment = data['human_rating'].values  # Can contain NaN

haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=data['essay_text'].tolist(),
    n_components=200
)

# The analysis will automatically handle missing values


# 5. CUSTOM ANALYSIS WORKFLOW
# ---------------------------

# Access the underlying components for custom analysis
analysis = haam.analysis  # HAAMAnalysis object
topic_analyzer = haam.topic_analyzer  # TopicAnalyzer object
visualizer = haam.visualizer  # HAAMVisualizer object

# Get PCA features
pca_features = analysis.results['pca_features']  # Shape: (n_samples, n_components)
variance_explained = analysis.results['variance_explained']  # Variance by each PC

# Get detailed debiased lasso results
sc_results = analysis.results['debiased_lasso']['SC']
print(f"SC model selected {sc_results['n_selected']} PCs")
print(f"SC model R² (CV): {sc_results['r2_cv']:.4f}")

# Get selected PC indices for each outcome
sc_selected = sc_results['selected']  # Array of selected PC indices
ai_selected = analysis.results['debiased_lasso']['AI']['selected']
hu_selected = analysis.results['debiased_lasso']['HU']['selected']

# Get topic associations for all PCs
all_associations = topic_analyzer.get_pc_topic_associations()

# Get high/low topics for a specific PC
pc_topics = topic_analyzer.get_pc_high_low_topics(
    pc_idx=4,  # PC5 (0-based)
    n_high=10,
    n_low=10,
    p_threshold=0.01  # Only very significant topics
)


# =============================================================================
# R USAGE WITH RETICULATE
# =============================================================================

# In R:

# install.packages("reticulate")
library(reticulate)

# Set Python environment (optional)
# use_python("/usr/local/bin/python3")
# use_virtualenv("myenv")
# use_condaenv("myenv")

# Source the Python files
source_python("haam_main.py")
source_python("haam_core.py") 
source_python("haam_topics.py")
source_python("haam_visualizations.py")

# Or if installed as package:
# py_install("git+https://github.com/raymondli-me/haam.git")
# haam <- import("haam")

# Load your data
data <- read.csv("your_data.csv")

# Create Python objects from R data
criterion <- r_to_py(data$social_class)
ai_judgment <- r_to_py(data$ai_rating)
human_judgment <- r_to_py(data$human_rating)
texts <- r_to_py(as.list(data$essay_text))

# Run analysis
haam_analysis <- HAAM(
  criterion = criterion,
  ai_judgment = ai_judgment,
  human_judgment = human_judgment,
  texts = texts,
  n_components = 200L,  # Use L for integers
  auto_run = TRUE
)

# Access results (automatically converted to R objects)
model_summary <- haam_analysis$results$model_summary
coefficients <- haam_analysis$results$coefficients
top_pcs <- haam_analysis$results$top_pcs

# Create visualizations
main_viz <- haam_analysis$create_main_visualization("./output")
mini_grid <- haam_analysis$create_mini_grid("./output")

# Export all results
outputs <- haam_analysis$export_all_results("./output")

# Explore PC topics
pc_topics <- haam_analysis$explore_pc_topics(
  pc_indices = list(4L, 7L, 1L),
  n_topics = 10L
)

# Convert results to R dataframe
pc_topics_df <- py_to_r(pc_topics)


# =============================================================================
# COMMON WORKFLOWS
# =============================================================================

# WORKFLOW 1: Quick analysis with all defaults
# --------------------------------------------
haam = HAAM(criterion, ai_judgment, human_judgment, texts=texts)
haam.export_all_results()


# WORKFLOW 2: Focus on specific PCs
# ---------------------------------
haam = HAAM(criterion, ai_judgment, human_judgment, texts=texts)

# Get PCs that are important for AI
top_ai_pcs = haam.analysis.get_top_pcs(n_top=20, ranking_method='AI')

# Explore their topics
for pc_idx in top_ai_pcs[:5]:
    topics = haam.topic_analyzer.get_pc_high_low_topics(pc_idx)
    print(f"\nPC{pc_idx+1}:")
    print(f"  HIGH: {topics['high'][0]['keywords'] if topics['high'] else 'None'}")
    print(f"  LOW: {topics['low'][0]['keywords'] if topics['low'] else 'None'}")


# WORKFLOW 3: Compare models
# --------------------------
# Get all coefficients
coef_df = haam.results['coefficients']

# Find PCs where AI and Human disagree
disagreement = []
for i in range(len(coef_df)):
    ai_coef = coef_df.iloc[i]['AI_coef']
    hu_coef = coef_df.iloc[i]['HU_coef']
    
    # Different signs and both significant
    if np.sign(ai_coef) != np.sign(hu_coef) and abs(ai_coef) > 0.05 and abs(hu_coef) > 0.05:
        disagreement.append(i)

print(f"PCs where AI and Human disagree: {disagreement}")


# WORKFLOW 4: Generate embeddings only
# ------------------------------------
from haam_core import HAAMAnalysis

texts = ["text 1", "text 2", "text 3"]
embeddings = HAAMAnalysis.generate_embeddings(texts)
print(f"Embeddings shape: {embeddings.shape}")


# =============================================================================
# INTERPRETING RESULTS
# =============================================================================

# The main outputs to examine:

# 1. Model Summary - Overall performance
#    - R²(CV): Cross-validated R-squared (use this, not in-sample)
#    - N_selected: Number of PCs selected by lasso

# 2. Top PCs - Most important principal components
#    - These explain the most variance in outcomes
#    - Check their topic associations for interpretation

# 3. Coefficients DataFrame - All PC coefficients
#    - Shows which PCs matter for each outcome
#    - Look for patterns across outcomes

# 4. Topic Associations - What each PC represents
#    - HIGH topics: High values on this PC
#    - LOW topics: Low values on this PC
#    - Use keywords to interpret PC meaning

# 5. Visualizations
#    - Main viz: Shows top 9 PCs and their relationships
#    - Mini grid: Shows all 200 PCs at once
#    - UMAP: Shows how outcomes vary in embedding space


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# Issue: "No module named 'haam'"
# Solution: Make sure files are in Python path
import sys
sys.path.append('/path/to/haam/folder')

# Issue: Memory error with large datasets
# Solution: Use fewer components or batch processing
haam = HAAM(criterion, ai_judgment, human_judgment, 
           embeddings=embeddings,  # Pre-compute embeddings
           n_components=100)  # Use fewer components

# Issue: No topic information in visualizations
# Solution: Make sure to provide texts, not just embeddings
haam = HAAM(criterion, ai_judgment, human_judgment, 
           texts=texts,  # Required for topic analysis
           embeddings=embeddings)

# Issue: Can't see visualizations in Colab
# Solution: Open the HTML files
from IPython.display import IFrame
IFrame(src='./haam_main_visualization.html', width=1000, height=600)