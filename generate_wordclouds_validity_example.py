#!/usr/bin/env python3

"""
Word Cloud Generation with Validity Coloring - HAAM Analysis
===========================================================
This script demonstrates the new validity coloring feature for word clouds.
It shows which topics are valid vs perceived social class markers.
"""

# Install wordcloud if not already installed
!pip install wordcloud

# Mount Google Drive (if not already mounted)
from google.colab import drive
import os
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haam import HAAM
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("WORD CLOUD GENERATION WITH VALIDITY COLORING")
print("="*80)

# ==============================================================================
# LOAD DATA AND RUN HAAM ANALYSIS
# ==============================================================================

print("\n1. LOADING DATA AND RUNNING HAAM ANALYSIS...")
print("-"*60)

# Load your data
filename = 'combined_social_class_dataset_v2.csv'
data_path = f'/content/drive/MyDrive/2025_06_30_anonymized_data_dmllme_2025_06_30/NV_Embed_Combined/{filename}'

df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Extract data
texts = df['text'].values.tolist()
embedding_cols = [f'embed_dim_{i}' for i in range(4096)]
embeddings = df[embedding_cols].values
social_class = df['social_class'].astype(float).values
ai_rating = df['ai_rating'].astype(float).values
human_rating = df['SC_RATING_11'].astype(float).values

# Initialize HAAM
haam = HAAM(
    criterion=social_class,
    ai_judgment=ai_rating,
    human_judgment=human_rating,
    embeddings=embeddings,
    texts=texts,
    n_components=200,
    min_cluster_size=10,
    min_samples=2,
    umap_n_components=3,
    standardize=True,
    sample_split_post_lasso=False,
    auto_run=True
)

print("✓ HAAM analysis complete!")

# ==============================================================================
# DEMONSTRATE VALIDITY COLORING
# ==============================================================================

print("\n2. DEMONSTRATING VALIDITY COLORING FEATURE...")
print("-"*60)

# Your specific PCs
specific_pcs = [2, 1, 3, 4, 5]  # First 5 from your list

# Create output directories
pole_dir = 'wordclouds_pole_coloring'
validity_dir = 'wordclouds_validity_coloring'
os.makedirs(pole_dir, exist_ok=True)
os.makedirs(validity_dir, exist_ok=True)

print("\nGenerating word clouds for PC3 to show the difference...")
print("\n" + "="*60)

# ==============================================================================
# PC3 WITH POLE COLORING (Traditional)
# ==============================================================================

print("A. PC3 with POLE COLORING (traditional red/blue gradients)")
print("-"*40)

fig_pole, _, _ = haam.create_pc_wordclouds(
    pc_idx=2,  # PC3 (0-based)
    k=15,
    max_words=150,
    figsize=(12, 6),
    output_dir=pole_dir,
    display=True,  # Display this one
    color_mode='pole'  # Traditional coloring
)

print("\n✓ This shows high pole topics in red, low pole topics in blue")
print("  All topics in each pole get the same color gradient")

# ==============================================================================
# PC3 WITH VALIDITY COLORING (New feature)
# ==============================================================================

print("\n" + "="*60)
print("B. PC3 with VALIDITY COLORING (shows valid vs perceived markers)")
print("-"*40)

fig_validity, _, _ = haam.create_pc_wordclouds(
    pc_idx=2,  # PC3 (0-based)
    k=15,
    max_words=150,
    figsize=(12, 6),
    output_dir=validity_dir,
    display=True,  # Display this one
    color_mode='validity'  # NEW validity coloring
)

print("\n✓ This shows which topics are:")
print("  - DARK RED: Valid high SC markers (Y+HU+AI all agree)")
print("  - LIGHT RED: Perceived high SC (only HU+AI, not Y)")
print("  - DARK BLUE: Valid low SC markers (Y+HU+AI all agree)")
print("  - LIGHT BLUE: Perceived low SC (only HU+AI, not Y)")
print("  - GREY: Mixed or weak signals")

# ==============================================================================
# GRID COMPARISON
# ==============================================================================

print("\n" + "="*60)
print("3. CREATING GRID VISUALIZATIONS FOR COMPARISON...")
print("-"*60)

# Grid with traditional pole coloring
print("\nCreating grid with POLE coloring...")
grid_pole = haam.create_top_pcs_wordcloud_grid(
    pc_indices=specific_pcs[:9],
    k=10,
    max_words=50,
    output_file=os.path.join(pole_dir, 'grid_pole.png'),
    display=False,
    color_mode='pole'
)
print("✓ Saved: grid_pole.png")

# Grid with validity coloring
print("\nCreating grid with VALIDITY coloring...")
grid_validity = haam.create_top_pcs_wordcloud_grid(
    pc_indices=specific_pcs[:9],
    k=10,
    max_words=50,
    output_file=os.path.join(validity_dir, 'grid_validity.png'),
    display=True,  # Display this one
    color_mode='validity'
)
print("✓ Saved: grid_validity.png")

# ==============================================================================
# BATCH GENERATION FOR ALL YOUR PCs
# ==============================================================================

print("\n" + "="*60)
print("4. BATCH GENERATING ALL YOUR REQUESTED PCs...")
print("-"*60)

# All your requested PCs
all_specific_pcs = [2, 1, 3, 4, 5, 14, 13, 11, 12, 46, 9, 17, 16, 20, 105]

print(f"\nGenerating validity-colored word clouds for {len(all_specific_pcs)} PCs...")
print(f"PCs (1-based): {[pc+1 for pc in all_specific_pcs]}")

# Batch generate with validity coloring
output_paths = haam.create_all_pc_wordclouds(
    pc_indices=all_specific_pcs,
    k=15,
    max_words=150,
    figsize=(12, 6),
    output_dir=os.path.join(validity_dir, 'batch'),
    display=False,
    color_mode='validity'
)

print(f"\n✓ Generated {len(output_paths)} word clouds with validity coloring")

# ==============================================================================
# INTERPRETATION GUIDE
# ==============================================================================

print("\n" + "="*80)
print("INTERPRETING VALIDITY COLORS")
print("="*80)

print("""
The validity coloring reveals important insights:

1. VALID MARKERS (Dark colors):
   - These words/topics genuinely indicate social class
   - They're consistent across objective truth and subjective ratings
   - Research value: These are real social class indicators

2. PERCEIVED MARKERS (Light colors):
   - These represent stereotypes or biases
   - Humans and AI think they indicate social class, but they don't
   - Research value: These reveal biases in perception

3. MIXED SIGNALS (Grey colors):
   - Inconsistent associations across measures
   - May indicate complex or ambiguous markers

Example interpretations:
- "Harvard" in dark red = genuinely indicates high social class
- "Designer bags" in light red = perceived as high class but not actually
- "Food stamps" in dark blue = genuinely indicates low social class  
- "Fast food" in light blue = stereotypically low class but not predictive
""")

print("\n✓ All word clouds saved to:")
print(f"  - Pole coloring: {pole_dir}/")
print(f"  - Validity coloring: {validity_dir}/")

print("\n🎨 TIP: Use validity coloring to study bias and stereotypes in your data!")
print("📊 TIP: Compare pole vs validity coloring to see which associations are real vs perceived")

print("\n✅ Done!")