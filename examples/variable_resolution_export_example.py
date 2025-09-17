#!/usr/bin/env python3
"""
Example: Converting HAAM Analysis to Variable Resolution Data Standard
=====================================================================

This example demonstrates how to use HAAM to analyze text data with
human and AI ratings, then export the results to the Variable Resolution
Data Standard format for visualization.
"""

import numpy as np
import pandas as pd
from haam import HAAM, haam_to_variable_resolution
import json

# Generate example data
np.random.seed(42)
n_samples = 500

# Simulate a scenario where we're analyzing sentiment ratings
# Criterion: "true" sentiment score (ground truth)
criterion = np.random.normal(50, 15, n_samples)  # Mean=50, SD=15

# Human ratings: somewhat correlated with truth but with noise
human_judgment = criterion + np.random.normal(0, 10, n_samples)

# AI ratings: different correlation pattern
ai_judgment = criterion * 0.9 + np.random.normal(5, 8, n_samples)

# Generate example texts with varying sentiment
texts = []
for i, score in enumerate(criterion):
    if score > 65:
        texts.append(f"Text {i}: This is an extremely positive message with great enthusiasm and joy!")
    elif score > 50:
        texts.append(f"Text {i}: A moderately positive statement showing some optimism.")
    elif score > 35:
        texts.append(f"Text {i}: A neutral comment without strong emotions either way.")
    else:
        texts.append(f"Text {i}: A somewhat negative text expressing concerns and disappointment.")

print("="*80)
print("HAAM TO VARIABLE RESOLUTION EXPORT EXAMPLE")
print("="*80)

# Step 1: Run HAAM analysis
print("\n1. Running HAAM analysis...")
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=texts,
    n_components=50,  # Extract 50 PCs
    min_cluster_size=15,  # Larger clusters for this example
    auto_run=True
)

# Step 2: Convert to Variable Resolution format
print("\n2. Converting to Variable Resolution Data Standard...")

# Example 1: Basic conversion with defaults
vr_data = haam_to_variable_resolution(
    haam,
    output_file="examples/output/haam_sentiment_analysis.json",
    title="Sentiment Analysis: Human vs AI Ratings",
    description="Comparison of human and AI sentiment ratings with ground truth",
    author="HAAM Example Script",
    include_pcs=10  # Include first 10 PCs
)

print(f"\n✓ Basic export complete!")
print(f"  - Variables defined: {len(vr_data['schema']['variables'])}")
print(f"  - Data items: {len(vr_data['data']['items'])}")
print(f"  - Has clustering: {'clusters' in vr_data['schema']}")
print(f"  - Has 3D positions: {'positioning' in vr_data['schema']}")

# Example 2: Export with all PCs
vr_data_all_pcs = haam_to_variable_resolution(
    haam,
    output_file="examples/output/haam_all_pcs.json",
    title="Full PC Analysis",
    description="Export with all principal components",
    include_all_pcs=True
)

print(f"\n✓ Full PC export complete!")
print(f"  - Total PCs included: {sum(1 for k in vr_data_all_pcs['schema']['variables'] if k.startswith('PC'))}")

# Example 3: Manual data export (without HAAM instance)
print("\n3. Manual data export example...")

# Create additional variables
confidence_scores = np.random.beta(8, 2, n_samples) * 100  # Skewed towards high confidence
response_time = np.random.gamma(2, 2, n_samples) * 1000  # Response time in ms

# Get PCA features and positions from HAAM
pca_features = haam.analysis.results['pca_features']
positions = None
clusters = None

if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer:
    if hasattr(haam.topic_analyzer, 'umap_embeddings'):
        positions = haam.topic_analyzer.umap_embeddings
    if hasattr(haam.topic_analyzer, 'clusters'):
        # Create cluster dict
        clusters = {
            "ids": haam.topic_analyzer.clusters,
            "labels": {i: f"Topic {i}" for i in np.unique(haam.topic_analyzer.clusters) if i != -1}
        }

# Use the converter class directly for more control
from haam import HAAMToVariableResolution

converter = HAAMToVariableResolution()
vr_data_custom = converter.convert_from_data(
    criterion=criterion,
    human_judgment=human_judgment,
    ai_judgment=ai_judgment,
    texts=texts,
    ids=[f"text_{i:04d}" for i in range(n_samples)],  # Custom IDs
    pca_features=pca_features,
    clusters=clusters,
    positions=positions,
    additional_variables={
        "confidence": confidence_scores,
        "response_time": response_time
    },
    title="Enhanced Sentiment Analysis",
    description="Sentiment analysis with additional metrics",
    include_pcs=15
)

# Validate before saving
is_valid, errors = converter.validate()
if is_valid:
    print("\n✓ Data validation passed!")
    converter.save_to_file("examples/output/haam_enhanced.json")
else:
    print("\n✗ Validation errors:")
    for error in errors:
        print(f"  - {error}")

# Example 4: Show structure of exported data
print("\n4. Structure of exported data:")
print(f"\nMetadata:")
print(f"  - Title: {vr_data['metadata']['title']}")
print(f"  - Created: {vr_data['metadata']['created']}")
print(f"  - Dataset size: {vr_data['metadata'].get('datasetSize', 'N/A')}")

print(f"\nVariables ({len(vr_data['schema']['variables'])} total):")
for var_name, var_def in list(vr_data['schema']['variables'].items())[:5]:
    print(f"  - {var_name}: {var_def['displayName']} ({var_def['type']})")
    if 'range' in var_def:
        print(f"    Range: [{var_def['range'][0]:.2f}, {var_def['range'][1]:.2f}]")

print(f"\nSample data item:")
if vr_data['data']['items']:
    item = vr_data['data']['items'][0]
    print(f"  - ID: {item['id']}")
    print(f"  - Content: {item['content'][:60]}...")
    print(f"  - Values: {list(item['values'].keys())}")
    if 'cluster' in item:
        print(f"  - Cluster: {item['cluster']}")
    if 'position' in item:
        print(f"  - Position: (x={item['position']['x']:.2f}, "
              f"y={item['position']['y']:.2f}, z={item['position']['z']:.2f})")

print("\n✓ Examples complete!")
print("\nThe exported JSON files can now be uploaded to the Variable Resolution")
print("visualization engine for interactive exploration of the HAAM analysis results.")

# Example 5: Quick statistics on the exported data
print("\n5. Quick statistics on correlations:")
# Calculate some correlations from the data
import scipy.stats as stats

corr_human_ai = stats.pearsonr(human_judgment, ai_judgment)[0]
corr_human_truth = stats.pearsonr(human_judgment, criterion)[0]
corr_ai_truth = stats.pearsonr(ai_judgment, criterion)[0]

print(f"  - Human-AI correlation: {corr_human_ai:.3f}")
print(f"  - Human-Truth correlation: {corr_human_truth:.3f}")
print(f"  - AI-Truth correlation: {corr_ai_truth:.3f}")

# Show PC1 correlations if available
if 'PC1' in vr_data['data']['items'][0]['values']:
    pc1_values = [item['values']['PC1'] for item in vr_data['data']['items']]
    print(f"\nPC1 correlations:")
    print(f"  - PC1-Human: {stats.pearsonr(pc1_values, human_judgment)[0]:.3f}")
    print(f"  - PC1-AI: {stats.pearsonr(pc1_values, ai_judgment)[0]:.3f}")
    print(f"  - PC1-Truth: {stats.pearsonr(pc1_values, criterion)[0]:.3f}")