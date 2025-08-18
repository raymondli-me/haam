#!/usr/bin/env python3

"""
Word Cloud Generation Script with VALIDITY COLORING
===================================================
This script generates word clouds using validity coloring to show
which topics are valid vs perceived social class markers.

Color meanings:
- Dark Red: Valid high SC markers (Y+HU+AI all high)
- Light Red: Perceived high SC (HU+AI high, Y not)
- Dark Blue: Valid low SC markers (Y+HU+AI all low)
- Light Blue: Perceived low SC (HU+AI low, Y not)
- Dark Grey: Mixed strong signals
- Light Grey: Mixed weak signals
"""

# Install wordcloud if not already installed
!pip install wordcloud
# Install HAAM package from GitHub
!pip install git+https://github.com/raymondli-me/haam.git

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
# LOAD EXISTING HAAM ANALYSIS
# ==============================================================================

print("\n1. LOADING EXISTING HAAM ANALYSIS...")
print("-"*60)

# Load the same data as in your initial analysis
filename = 'combined_social_class_dataset_v2.csv'
data_path = f'/content/drive/MyDrive/2025_06_30_anonymized_data_dmllme_2025_06_30/NV_Embed_Combined/{filename}'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Extract data with same column names as initial analysis
texts = df['text'].values.tolist()
embedding_cols = [f'embed_dim_{i}' for i in range(4096)]
embeddings = df[embedding_cols].values
social_class = df['social_class'].astype(float).values
ai_rating = df['ai_rating'].astype(float).values
human_rating = df['SC_RATING_11'].astype(float).values

print(f"\nData summary:")
print(f"  Total essays: {len(texts)}")
print(f"  Valid human ratings: {(~np.isnan(human_rating)).sum()} ({(~np.isnan(human_rating)).sum()/len(human_rating)*100:.1f}%)")

# ==============================================================================
# RE-INITIALIZE HAAM WITH SAME PARAMETERS
# ==============================================================================

print("\n2. RE-INITIALIZING HAAM...")
print("-"*60)
print("Note: This will reload the analysis to access the topic analyzer")

# Initialize HAAM with exact same parameters as your initial run
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

print("✓ HAAM analysis reloaded successfully!")
print(f"✓ Topic analyzer available: {hasattr(haam, 'topic_analyzer') and haam.topic_analyzer is not None}")

# ==============================================================================
# VALIDITY COLORING EXPLANATION
# ==============================================================================

print("\n" + "="*80)
print("VALIDITY COLORING EXPLAINED")
print("="*80)
print("""
Validity coloring reveals which topics are:
- VALID markers: Consistent across ground truth (Y) and perceptions (HU/AI)
- PERCEIVED markers: Only in perceptions, not ground truth (stereotypes/biases)

Colors:
- Dark Red (#8B0000): Valid high SC - genuinely indicates high social class
- Light Red (#FF6B6B): Perceived high SC - stereotype of high class
- Dark Blue (#00008B): Valid low SC - genuinely indicates low social class  
- Light Blue (#6B9AFF): Perceived low SC - stereotype of low class
- Dark Grey (#4A4A4A): Mixed strong signals
- Light Grey (#B0B0B0): Mixed weak signals
""")

# ==============================================================================
# GENERATE WORD CLOUDS WITH VALIDITY COLORING
# ==============================================================================

print("\n3. GENERATING WORD CLOUDS WITH VALIDITY COLORING...")
print("-"*60)

# Define the specific PCs you want (0-based indices)
# You specified: PC3,2,4,5,6,15,14,12,13,47,10,18,17,21,106
specific_pcs = [2, 1, 3, 4, 5, 14, 13, 11, 12, 46, 9, 17, 16, 20, 105]  # 0-based indices

print(f"Generating validity-colored word clouds for {len(specific_pcs)} specific PCs:")
print(f"PC indices (1-based): {[pc+1 for pc in specific_pcs]}")

# Create output directory for word clouds
wordcloud_dir = 'haam_wordclouds_validity'
os.makedirs(wordcloud_dir, exist_ok=True)

# Parameters for word cloud generation
k_topics = 3  # Number of topics to include from each pole
max_words = 150  # Maximum words in each word cloud
figsize = (12, 6)  # Size of each word cloud figure

# Generate word clouds for each specific PC
print("\nGenerating individual word clouds with validity coloring...")
successful_pcs = []
failed_pcs = []

for i, pc_idx in enumerate(specific_pcs):
    print(f"\n  [{i+1}/{len(specific_pcs)}] Creating validity-colored word cloud for PC{pc_idx + 1}...")
    
    try:
        fig, high_path, low_path = haam.create_pc_wordclouds(
            pc_idx=pc_idx,
            k=k_topics,
            max_words=max_words,
            figsize=figsize,
            output_dir=wordcloud_dir,
            display=True,
            color_mode='validity'  # USE VALIDITY COLORING
        )
        plt.close(fig)  # Close figure to save memory
        
        print(f"    ✓ High pole: {high_path}")
        print(f"    ✓ Low pole: {low_path}")
        successful_pcs.append(pc_idx)
        
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")
        failed_pcs.append(pc_idx)

print(f"\n✓ Successfully generated validity-colored word clouds for {len(successful_pcs)} PCs")
if failed_pcs:
    print(f"✗ Failed for {len(failed_pcs)} PCs: {[pc+1 for pc in failed_pcs]}")

# ==============================================================================
# CREATE GRID VISUALIZATION WITH VALIDITY COLORING
# ==============================================================================

print("\n4. CREATING GRID VISUALIZATION WITH VALIDITY COLORING...")
print("-"*60)

# Create a grid showing the first 9 of your specific PCs
grid_pcs = specific_pcs[:9]  # Take first 9 for a 3x3 grid
print(f"Creating validity-colored grid for PCs: {[pc+1 for pc in grid_pcs]}")

try:
    grid_fig = haam.create_top_pcs_wordcloud_grid(
        pc_indices=grid_pcs,
        k=1,  # Fewer topics for grid view
        max_words=50,  # Fewer words for clarity in grid
        output_file=os.path.join(wordcloud_dir, 'pc_wordcloud_grid_validity.png'),
        display=True,
        color_mode='validity'  # USE VALIDITY COLORING
    )
    print("✓ Validity-colored grid visualization saved successfully!")
except Exception as e:
    print(f"✗ Grid visualization failed: {str(e)}")

# ==============================================================================
# BATCH GENERATION WITH VALIDITY COLORING
# ==============================================================================

print("\n5. BATCH GENERATING ALL WORD CLOUDS WITH VALIDITY COLORING...")
print("-"*60)

# Generate all word clouds in batch mode with validity coloring
print(f"Generating validity-colored word clouds for all {len(specific_pcs)} specified PCs...")

try:
    output_paths = haam.create_all_pc_wordclouds(
        pc_indices=specific_pcs,
        k=k_topics,
        max_words=max_words,
        figsize=figsize,
        output_dir=os.path.join(wordcloud_dir, 'batch'),
        display=False,  # Don't display in batch mode
        color_mode='validity'  # USE VALIDITY COLORING
    )
    
    print(f"\n✓ Batch generation complete!")
    print(f"✓ Generated validity-colored word clouds for {len(output_paths)} PCs")
    
except Exception as e:
    print(f"✗ Batch generation failed: {str(e)}")

# ==============================================================================
# EXPLORE PC TOPICS WITH VALIDITY CONTEXT
# ==============================================================================

print("\n6. EXPLORING PC TOPICS FOR VALIDITY CONTEXT...")
print("-"*60)

# Get topic information for the specific PCs
print("Retrieving topic information to understand validity patterns...")

try:
    pc_topics_df = haam.explore_pc_topics(pc_indices=specific_pcs, n_topics=5)
    
    # Save topic exploration results
    topics_path = os.path.join(wordcloud_dir, 'pc_topics_validity_exploration.csv')
    pc_topics_df.to_csv(topics_path, index=False)
    print(f"✓ Saved topic exploration: {topics_path}")
    
    # Display summary for each PC with validity interpretation hints
    print("\nPC Topic Summary (consider validity patterns):")
    print("-"*60)
    for pc_idx in specific_pcs[:5]:  # Show first 5 for brevity
        pc_data = pc_topics_df[pc_topics_df['PC'] == pc_idx + 1]
        if not pc_data.empty:
            print(f"\nPC{pc_idx + 1}:")
            high_data = pc_data[pc_data['Direction'] == 'HIGH']
            low_data = pc_data[pc_data['Direction'] == 'LOW']
            
            if not high_data.empty:
                print(f"  HIGH: {high_data.iloc[0]['Keywords'][:80]}...")
                print(f"        (Check word cloud for validity colors)")
            if not low_data.empty:
                print(f"  LOW:  {low_data.iloc[0]['Keywords'][:80]}...")
                print(f"        (Check word cloud for validity colors)")
                
except Exception as e:
    print(f"✗ Topic exploration failed: {str(e)}")

# ==============================================================================
# CREATE RANKING-BASED VALIDITY VISUALIZATIONS
# ==============================================================================

print("\n7. CREATING RANKING-BASED VALIDITY VISUALIZATIONS...")
print("-"*60)

# Create validity-colored visualizations based on different ranking methods
ranking_methods = ['HU', 'AI', 'Y']

for ranking in ranking_methods:
    print(f"\nCreating validity-colored grid for top PCs by {ranking} ranking...")
    
    try:
        # Get top PCs by this ranking
        top_pcs_by_ranking = haam.analysis.get_top_pcs(n=9, ranking_method=ranking)
        
        # Create grid with validity coloring
        grid_path = os.path.join(wordcloud_dir, f'validity_grid_{ranking.lower()}_ranking.png')
        fig = haam.create_top_pcs_wordcloud_grid(
            pc_indices=top_pcs_by_ranking,
            k=k_topics,
            max_words=50,
            output_file=grid_path,
            display=False,  # Don't display all of them
            color_mode='validity'  # USE VALIDITY COLORING
        )
        print(f"  ✓ Saved: {grid_path}")
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")

# Display one example
print("\nDisplaying validity-colored grid for top Y-ranked PCs...")
try:
    top_y_pcs = haam.analysis.get_top_pcs(n=9, ranking_method='Y')
    fig = haam.create_top_pcs_wordcloud_grid(
        pc_indices=top_y_pcs,
        k=k_topics,
        max_words=50,
        display=True,
        color_mode='validity'
    )
except:
    pass

# ==============================================================================
# VALIDITY INTERPRETATION GUIDE
# ==============================================================================

print("\n" + "="*80)
print("INTERPRETING YOUR VALIDITY-COLORED WORD CLOUDS")
print("="*80)

print("""
Look for these patterns in your word clouds:

1. DARK RED words (Valid high SC):
   - These genuinely indicate high social class
   - Reliable markers for research
   - Example: "ivy league", "trust fund", "private equity"

2. LIGHT RED words (Perceived high SC):
   - Stereotypes of high class that aren't actually predictive
   - Reveal biases in human/AI perception
   - Example: "designer brands", "luxury cars"

3. DARK BLUE words (Valid low SC):
   - These genuinely indicate low social class
   - Reliable markers for research
   - Example: "food stamps", "public housing", "welfare"

4. LIGHT BLUE words (Perceived low SC):
   - Stereotypes of low class that aren't actually predictive
   - Reveal biases in human/AI perception
   - Example: "fast food", "walmart", "trailer"

5. GREY words (Mixed signals):
   - Complex or ambiguous markers
   - May vary by context
   - Need further investigation
""")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("VALIDITY-COLORED WORD CLOUD GENERATION COMPLETE!")
print("="*80)

print(f"\nAll validity-colored word clouds saved to: {wordcloud_dir}/")
print("\nGenerated files:")
print(f"  🎨 Individual validity word clouds for {len(successful_pcs)} PCs")
print(f"  📊 Validity grid visualization: pc_wordcloud_grid_validity.png")
print(f"  📁 Batch validity word clouds in: {wordcloud_dir}/batch/")
print(f"  📝 Topic exploration: pc_topics_validity_exploration.csv")
print(f"  🎯 Validity grids by ranking: HU, AI, and Y")

print("\nSpecific PCs processed (1-based):")
print(f"  {[pc+1 for pc in specific_pcs]}")

print("\nWord cloud parameters used:")
print(f"  - Topics per pole (k): {k_topics}")
print(f"  - Max words displayed: {max_words}")
print(f"  - Figure size: {figsize}")
print(f"  - Color mode: VALIDITY")

print("\n✅ Validity-colored word cloud generation complete!")
print("🔍 Use these to identify genuine vs perceived social class markers!")
print("📊 Compare with your pole-colored versions to see the difference!")