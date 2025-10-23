#!/usr/bin/env python3
"""
HAAM Basic Example
==================
Minimal example showing core HAAM (Human AI Accuracy Model) workflow for one construct (Prestige).
"""

import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from haam import HAAM

# Set seed for reproducibility (CRITICAL for consistent results)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Generate embeddings from text
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

# 3. Extract variables for prestige construct
X = df['self_prestige'].values          # Self-report
Y_HU = df['judge_prestige'].values      # Human judgment
Y_AI = df['AI_Pres_mean'].values        # AI judgment

# Remove any NaN values (clean data)
valid_mask = ~(np.isnan(X) | np.isnan(Y_HU) | np.isnan(Y_AI))
X = X[valid_mask]
Y_HU = Y_HU[valid_mask]
Y_AI = Y_AI[valid_mask]
embeddings = embeddings[valid_mask]
texts = [df['text'].tolist()[i] for i in range(len(df)) if valid_mask[i]]

# 4. Run HAAM with exact settings from Brunswik Newsletter analysis
haam = HAAM(
    criterion=X,
    human_judgment=Y_HU,
    ai_judgment=Y_AI,
    embeddings=embeddings,
    texts=texts,
    n_components=50,
    standardize=True,
    sample_split_post_lasso=False,  # Use full sample for max power
    min_cluster_size=3,              # HDBSCAN parameter for topic clustering
    min_samples=1,                   # HDBSCAN parameter for core points
    auto_run=True
)

# 5. Access results
results = haam.analysis.results

# Print key findings
print("\n" + "="*60)
print("PRESTIGE - KEY RESULTS")
print("="*60)

# Zero-order correlations
r_X_HU = np.corrcoef(X, Y_HU)[0, 1]
r_X_AI = np.corrcoef(X, Y_AI)[0, 1]
r_HU_AI = np.corrcoef(Y_HU, Y_AI)[0, 1]
print(f"\nValidity--Human:  r = {r_X_HU:.3f}")
print(f"Validity--AI:     r = {r_X_AI:.3f}")
print(f"Human--AI:        r = {r_HU_AI:.3f}")

# LASSO feature selection
n_X = results['debiased_lasso']['X']['n_selected']
n_AI = results['debiased_lasso']['AI']['n_selected']
n_HU = results['debiased_lasso']['HU']['n_selected']
print(f"\nPCs selected by LASSO:")
print(f"  Validity:  {n_X:>2d} / 50")
print(f"  AI:        {n_AI:>2d} / 50")
print(f"  Human:     {n_HU:>2d} / 50")

# R² values (post-LASSO CV)
r2_X = results['debiased_lasso']['X'].get('r2_cv_post_lasso',
                                           results['debiased_lasso']['X']['r2_cv'])
r2_AI = results['debiased_lasso']['AI'].get('r2_cv_post_lasso',
                                             results['debiased_lasso']['AI']['r2_cv'])
r2_HU = results['debiased_lasso']['HU'].get('r2_cv_post_lasso',
                                             results['debiased_lasso']['HU']['r2_cv'])
print(f"\nCross-validated R² (post-LASSO):")
print(f"  Validity:  {r2_X:.3f}")
print(f"  AI:        {r2_AI:.3f}")
print(f"  Human:     {r2_HU:.3f}")

print("\n✓ Analysis complete!")
print("="*60 + "\n")
