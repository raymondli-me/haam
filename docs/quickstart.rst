Quick Start
===========

Mini-Example: Human-GPT Perception of Anger in Reddit Comments
--------------------------------------------------------------

.. code-block:: python

   import subprocess
   import sys
   subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud", "-q"])
   !pip install git+https://github.com/raymondli-me/haam.git
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.gridspec as gridspec
   from matplotlib.patches import Patch
   import os
   import re
   from haam import HAAM
   import warnings
   warnings.filterwarnings('ignore')


   github_url_processed = 'https://raw.githubusercontent.com/raymondli-me/haam/main/data/anger_family_with_angry_word_count.csv'

   try:
       df_loaded_github = pd.read_csv(github_url_processed)
       df_filtered_github = df_loaded_github.copy() # Use a consistent variable name
       print(f"✓ Loaded data from GitHub CSV: {df_filtered_github.shape[0]} rows, {df_filtered_github.shape[1]} columns")
   except Exception as e:
       print(f"Error loading data from GitHub: {e}")
       print(f"Please ensure the URL is correct: {github_url_processed}")
       df_filtered_github = None # Set to None to prevent further execution if file not found

   if df_filtered_github is not None:
       # Prepare data for HAAM using the loaded data
       # Ensure column names match those in your CSV
       if 'angry_word_count' in df_filtered_github.columns and \
          'human_sum_score' in df_filtered_github.columns and \
          'gpt_sum_score' in df_filtered_github.columns and \
          'text' in df_filtered_github.columns:

           texts_github = df_filtered_github['text'].values.tolist()
           criterion_github = df_filtered_github['angry_word_count'].values.astype(float)  # X: angry word count
           human_judgment_github = df_filtered_github['human_sum_score'].values.astype(float)  # HU: human ratings
           ai_judgment_github = df_filtered_github['gpt_sum_score'].values.astype(float)  # AI: GPT ratings

           haam_github = HAAM(
               criterion=criterion_github,        # X: The ground truth variable (angry word count in this case)
               ai_judgment=ai_judgment_github,    # AI: AI system's predictions/ratings
               human_judgment=human_judgment_github, # HU: Human ratings/judgments
               texts=texts_github,                # Raw text data for topic modeling
               n_components=200,                  # Number of principal components to extract
               min_cluster_size=10,               # HDBSCAN parameter for topic clustering
               min_samples=2,                     # HDBSCAN parameter for core points
               umap_n_components=3,               # 3D UMAP for topic embedding
               standardize=True,                  # Standardize variables for DML
               sample_split_post_lasso=False,     # Use full sample for max power
               auto_run=True                      # Run analysis immediately
           )

           results_github = haam_github.create_comprehensive_pc_analysis(
               #pc_indices=specific_pcs,          # Which PCs to analyze (commented = use first 15)
               k_topics=3,                        # Number of topics per pole in word clouds
               max_words=100,                     # Maximum words per word cloud
               generate_wordclouds=True,          # Create word cloud table (like your Colab script)
               generate_3d_umap=True,             # Create 3D UMAP with PC arrows
               umap_arrow_k=1,                    # Single topic endpoints for arrows (clean visualization)
               show_data_counts=True,             # Show "HU: n=3" warnings for sparse data
               output_dir='haam_wordclouds_final_aligned_sparse_github', # Output directory
               display=True                       # Show plots in notebook/colab
               )

Understanding the Output
------------------------

The analysis produces several key outputs:

1. **Word Clouds**: Show which topics are associated with high vs low PC scores, colored by validity:
   
   - Dark red: Consensus high (top quartile for X, HU, and AI)
   - Light red: Any high (top quartile for HU & AI only)
   - Dark blue: Consensus low (bottom quartile for all)
   - Light blue: Any low (bottom quartile for HU & AI only)
   - Grey: Mixed signals

2. **3D UMAP Visualization**: Interactive plot showing topic clusters with PC directional arrows

3. **PoMA Analysis**: Percentage of accuracy flowing through measured cues vs direct pathways

4. **Model Performance**: R² values showing how well PCs predict X, HU, and AI judgments

Simple Usage
------------

For basic analysis without the full visualization pipeline:

.. code-block:: python

   from haam import HAAM
   import numpy as np
   
   # Your data
   criterion = np.array([...])        # Ground truth
   ai_judgment = np.array([...])      # AI predictions
   human_judgment = np.array([...])   # Human ratings
   texts = [...]                      # Optional: text data
   
   # Run analysis
   haam = HAAM(
       criterion=criterion,
       ai_judgment=ai_judgment,
       human_judgment=human_judgment,
       texts=texts,  # Optional
       auto_run=True
   )
   
   # View results
   print(haam.results['model_summary'])
   print(f"Top PCs: {haam.results['top_pcs']}")
   
   # Create main visualization
   haam.create_main_visualization()

Key Methods
-----------

- ``create_comprehensive_pc_analysis()``: Full analysis with word clouds and UMAP
- ``create_main_visualization()``: Interactive dashboard of top PCs
- ``create_3d_umap_with_pc_arrows()``: 3D topic space with PC directions
- ``create_pc_wordclouds()``: Word clouds for specific PCs
- ``export_all_results()``: Save all results to disk