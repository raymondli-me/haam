#!/usr/bin/env python3
"""
HAAM Basic Example
==================
Minimal example showing core HAAM (Human AI Accuracy Model) workflow for one construct (Prestige).
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from haam import HAAM

# Set seed for reproducibility
np.random.seed(42)

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Generate embeddings from text
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

# 3. Extract variables for prestige construct
X = df['self_prestige'].values          # Self-report
Y_HU = df['judge_prestige'].values      # Human judgment
Y_AI = df['AI_Pres_mean'].values        # AI judgment

# 4. Run HAAM
haam = HAAM(
    criterion=X,
    human_judgment=Y_HU,
    ai_judgment=Y_AI,
    embeddings=embeddings,
    texts=df['text'].tolist(),
    n_components=50,
    standardize=True,
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
print(f"\nValidity--Human:  r = {r_X_HU:.3f}")
print(f"Validity--AI:     r = {r_X_AI:.3f}")

# LASSO feature selection
n_X = results['debiased_lasso']['X']['n_selected']
n_AI = results['debiased_lasso']['AI']['n_selected']
n_HU = results['debiased_lasso']['HU']['n_selected']
print(f"\nPCs selected: Validity={n_X}/50, AI={n_AI}/50, Human={n_HU}/50")

# R² values
r2_X = results['debiased_lasso']['X'].get('r2_cv_post_lasso',
                                           results['debiased_lasso']['X']['r2_cv'])
print(f"\nCross-validated R²: {r2_X:.3f}")

print("\n✓ Analysis complete!")
print("="*60 + "\n")
