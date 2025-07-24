#!/usr/bin/env python3
"""
Simple HAAM Example
==================
Minimal example showing basic HAAM usage with the new v1.1.0 features.
"""

import pandas as pd
import numpy as np
from haam import HAAM

# Load your data
df = pd.read_csv('your_data.csv')

# Extract required columns
criterion = df['your_criterion_column'].values  # e.g., social class
ai_judgment = df['ai_predictions'].values
human_judgment = df['human_ratings'].values
embeddings = df[[f'embed_{i}' for i in range(384)]].values  # Your embeddings
texts = df['text'].values.tolist()  # Optional but recommended for topics

# Run HAAM analysis
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    embeddings=embeddings,
    texts=texts,
    n_components=200
)

# Access results
print("Model Summary:")
print(haam.results['model_summary'])

# Create visualizations (default: PC names are "-")
haam.create_main_visualization('./output')

# After interpreting results, add meaningful PC names
pc_names = {
    4: "Lifestyle & Work",      # PC5 (0-based index)
    7: "Professional Topics",   # PC8
    1: "Writing Style",         # PC2
    # ... add more based on your interpretation
}

# Re-create visualization with names
haam.create_main_visualization('./output', pc_names=pc_names)

# Export comprehensive metrics
metrics = haam.visualizer.create_metrics_summary('./output/metrics.json')
print(f"\nKey Metrics:")
print(f"R²(Y): {metrics['model_performance']['Y']['r2_cv']:.3f}")
print(f"R²(AI): {metrics['model_performance']['AI']['r2_cv']:.3f}")
print(f"R²(HU): {metrics['model_performance']['HU']['r2_cv']:.3f}")