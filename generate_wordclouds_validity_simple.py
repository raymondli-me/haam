#!/usr/bin/env python3

"""
Simple Enhanced Word Cloud Generation with Validity Coloring
===========================================================
This script generates word clouds with validity coloring and additional info
using the existing HAAM methods.
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
import matplotlib.gridspec as gridspec
from haam import HAAM
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED WORD CLOUD GENERATION WITH VALIDITY COLORING")
print("="*80)

# ==============================================================================
# LOAD DATA AND RUN HAAM
# ==============================================================================

print("\n1. LOADING DATA AND RUNNING HAAM ANALYSIS...")
print("-"*60)

# Load the data
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
# GENERATE ENHANCED WORD CLOUDS WITH INFO
# ==============================================================================

print("\n2. GENERATING ENHANCED WORD CLOUDS...")
print("-"*60)

# Your specific PCs
specific_pcs = [2, 1, 3, 4, 5, 14, 13, 11, 12, 46, 9, 17, 16, 20, 105]

# Create output directory
output_dir = 'haam_wordclouds_enhanced_simple'
os.makedirs(output_dir, exist_ok=True)

# Function to get PC associations
def get_pc_associations(haam_instance, pc_idx):
    """Get Y/HU/AI associations for a PC."""
    associations = {}
    try:
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in haam_instance.analysis.results['debiased_lasso']:
                coef = haam_instance.analysis.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                associations[outcome] = 'High' if coef > 0 else 'Low'
    except:
        pass
    return associations

# Function to get topic stats
def get_topic_stats(pc_topics):
    """Get number of topics and sample size."""
    high_topics = pc_topics.get('high', [])
    low_topics = pc_topics.get('low', [])
    high_samples = sum(t.get('size', 0) for t in high_topics)
    low_samples = sum(t.get('size', 0) for t in low_topics)
    return len(high_topics), len(low_topics), high_samples, low_samples

# Generate word clouds with extra info
print("\nGenerating individual word clouds with enhanced info...")
for i, pc_idx in enumerate(specific_pcs[:5]):  # First 5 with display
    print(f"\n[{i+1}/5] Generating PC{pc_idx + 1}...")
    
    try:
        # Get PC associations
        associations = get_pc_associations(haam, pc_idx)
        
        # Generate word cloud
        fig, _, _ = haam.create_pc_wordclouds(
            pc_idx=pc_idx,
            k=3,
            max_words=150,
            figsize=(14, 7),
            output_dir=output_dir,
            display=False,
            color_mode='validity'
        )
        
        # Get topic statistics
        pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
            pc_idx=pc_idx, n_high=3, n_low=3, p_threshold=0.05
        )
        n_high, n_low, high_samples, low_samples = get_topic_stats(pc_topics)
        
        # Add text info to the figure
        plt.figure(fig.number)
        
        # Add association info
        assoc_text = f"Y: {associations.get('SC', '?')} | HU: {associations.get('HU', '?')} | AI: {associations.get('AI', '?')}"
        plt.figtext(0.5, 0.02, assoc_text, ha='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7))
        
        # Add sample size info
        info_text = f"High: {n_high} topics, {high_samples:,} docs | Low: {n_low} topics, {low_samples:,} docs"
        plt.figtext(0.5, 0.06, info_text, ha='center', fontsize=12)
        
        plt.show()
        
        print(f"  ✓ PC{pc_idx + 1}: {assoc_text}")
        print(f"    {info_text}")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

# ==============================================================================
# CREATE TABLE VISUALIZATION FOR ALL 15 PCs
# ==============================================================================

print("\n3. CREATING TABLE VISUALIZATION...")
print("-"*60)

# Create a comprehensive table view
fig = plt.figure(figsize=(20, 60))
n_pcs = len(specific_pcs)

# Create grid - 3 columns per PC row
for i, pc_idx in enumerate(specific_pcs):
    print(f"  Processing PC{pc_idx + 1}...")
    
    # Create subplots for this PC
    # Column 1: PC label
    ax1 = plt.subplot(n_pcs, 3, i*3 + 1)
    ax1.text(0.5, 0.5, f'PC{pc_idx + 1}', fontsize=28, fontweight='bold',
             ha='center', va='center')
    ax1.axis('off')
    
    try:
        # Get associations and topics
        associations = get_pc_associations(haam, pc_idx)
        pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
            pc_idx=pc_idx, n_high=3, n_low=3, p_threshold=0.05
        )
        n_high, n_low, high_samples, low_samples = get_topic_stats(pc_topics)
        
        # Add association text below PC label
        assoc_text = f"Y:{associations.get('SC', '?')[0]} HU:{associations.get('HU', '?')[0]} AI:{associations.get('AI', '?')[0]}"
        ax1.text(0.5, 0.2, assoc_text, fontsize=16, ha='center', va='center')
        
        # Column 2: High pole
        ax2 = plt.subplot(n_pcs, 3, i*3 + 2)
        
        # Generate and save individual word cloud for high pole
        temp_fig, _, _ = haam.create_pc_wordclouds(
            pc_idx=pc_idx, k=3, max_words=100,
            output_dir=None, display=False, color_mode='validity'
        )
        
        # Extract the high pole image from temp figure
        temp_axes = temp_fig.get_axes()
        if temp_axes and len(temp_axes) > 0:
            # Get image data from first subplot (high pole)
            for child in temp_axes[0].get_children():
                if hasattr(child, 'get_array'):
                    ax2.imshow(child.get_array())
                    break
        
        ax2.set_title(f'High ({n_high} topics, n={high_samples:,})', 
                     fontsize=14, color='darkred')
        ax2.axis('off')
        plt.close(temp_fig)
        
        # Column 3: Low pole
        ax3 = plt.subplot(n_pcs, 3, i*3 + 3)
        
        # Generate another temp figure to get low pole
        temp_fig, _, _ = haam.create_pc_wordclouds(
            pc_idx=pc_idx, k=3, max_words=100,
            output_dir=None, display=False, color_mode='validity'
        )
        
        # Extract the low pole image from temp figure
        temp_axes = temp_fig.get_axes()
        if temp_axes and len(temp_axes) > 1:
            # Get image data from second subplot (low pole)
            for child in temp_axes[1].get_children():
                if hasattr(child, 'get_array'):
                    ax3.imshow(child.get_array())
                    break
        
        ax3.set_title(f'Low ({n_low} topics, n={low_samples:,})', 
                     fontsize=14, color='darkblue')
        ax3.axis('off')
        plt.close(temp_fig)
        
    except Exception as e:
        ax2 = plt.subplot(n_pcs, 3, i*3 + 2)
        ax2.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center')
        ax2.axis('off')
        
        ax3 = plt.subplot(n_pcs, 3, i*3 + 3)
        ax3.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax3.axis('off')

plt.suptitle('All 15 Principal Components - Validity Colored Word Clouds', 
             fontsize=24, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#8B0000', label='Consensus high'),
    Patch(facecolor='#FF6B6B', label='Any high'),
    Patch(facecolor='#00008B', label='Consensus low'),
    Patch(facecolor='#6B9AFF', label='Any low'),
    Patch(facecolor='#4A4A4A', label='Opposing'),
    Patch(facecolor='#B0B0B0', label='All middle')
]
plt.figlegend(handles=legend_elements, loc='lower center', 
             bbox_to_anchor=(0.5, -0.005), ncol=6, fontsize=14)

plt.tight_layout()
table_path = os.path.join(output_dir, 'pc_table_all_15.png')
plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"✓ Saved table visualization: {table_path}")
plt.show()

# ==============================================================================
# BATCH GENERATION
# ==============================================================================

print("\n4. BATCH GENERATING ALL WORD CLOUDS...")
print("-"*60)

# Generate all remaining PCs without display
remaining_pcs = specific_pcs[5:]  # Rest of the PCs
print(f"Generating {len(remaining_pcs)} remaining PCs...")

output_paths = haam.create_all_pc_wordclouds(
    pc_indices=remaining_pcs,
    k=3,
    max_words=150,
    figsize=(12, 6),
    output_dir=os.path.join(output_dir, 'batch'),
    display=False,
    color_mode='validity'
)

print(f"✓ Generated {len(output_paths)} word clouds")

# ==============================================================================
# FIX RANKING VISUALIZATIONS
# ==============================================================================

print("\n5. CREATING RANKING-BASED VISUALIZATIONS...")
print("-"*60)

for ranking in ['SC', 'AI', 'HU']:
    print(f"Creating grid for top {ranking}-ranked PCs...")
    try:
        # Use correct parameter name
        top_pcs = haam.analysis.get_top_pcs(n_top=9, ranking_method=ranking)
        
        grid_fig = haam.create_top_pcs_wordcloud_grid(
            pc_indices=top_pcs,
            k=3,
            max_words=50,
            output_file=os.path.join(output_dir, f'grid_{ranking.lower()}_ranking.png'),
            display=False,
            color_mode='validity'
        )
        print(f"  ✓ Saved grid for {ranking}")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

print("\n6. GENERATING SUMMARY REPORT...")
print("-"*60)

report_lines = ["PC Summary Report - Validity Patterns"]
report_lines.append("="*80)
report_lines.append("PC\tY/HU/AI\tHigh Topics\tLow Topics\tSample Sizes")
report_lines.append("-"*80)

for pc_idx in specific_pcs:
    try:
        associations = get_pc_associations(haam, pc_idx)
        pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
            pc_idx=pc_idx, n_high=3, n_low=3, p_threshold=0.05
        )
        n_high, n_low, high_samples, low_samples = get_topic_stats(pc_topics)
        
        pattern = f"{associations.get('SC', '?')[0]}/{associations.get('HU', '?')[0]}/{associations.get('AI', '?')[0]}"
        
        report_lines.append(
            f"PC{pc_idx+1}\t{pattern}\t"
            f"{n_high} topics\t{n_low} topics\t"
            f"H:{high_samples:,} L:{low_samples:,}"
        )
    except:
        report_lines.append(f"PC{pc_idx+1}\tError\tError\tError\tError")

# Save and display report
report_path = os.path.join(output_dir, 'pc_summary_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

print("PC Summary (first 5):")
for line in report_lines[:8]:
    print(line)

print(f"\n✓ Full report saved to: {report_path}")

# ==============================================================================
# DONE
# ==============================================================================

print("\n" + "="*80)
print("ENHANCED VALIDITY WORD CLOUD GENERATION COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {output_dir}/")
print("\n✅ Features included:")
print("  - Y/HU/AI associations shown for each PC")
print("  - Number of topics and sample sizes displayed")
print("  - Table visualization of all 15 PCs")
print("  - Fixed ranking-based grids")
print("  - Summary report with patterns")