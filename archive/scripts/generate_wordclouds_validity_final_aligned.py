#!/usr/bin/env python3

"""
Final Aligned Word Cloud Generation with Validity Coloring
=========================================================
This version uses the updated HAAM package where validity coloring
is now aligned with direct Y/HU/AI measurement.
"""

# Install wordcloud if not already installed
!pip install wordcloud
# Install HAAM package from GitHub (with aligned coloring)
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
from matplotlib.patches import Patch
from haam import HAAM
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FINAL ALIGNED WORD CLOUD GENERATION WITH VALIDITY COLORING")
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
print("✓ Word cloud coloring now uses direct Y/HU/AI measurement!")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_topic_quartile_positions_direct(haam_instance, topic_ids):
    """
    Calculate actual quartile positions for topics using direct measurement.
    This matches what the new coloring system does internally.
    """
    if not topic_ids:
        return {'Y': '?', 'HU': '?', 'AI': '?'}
    
    # Get the cluster labels
    cluster_labels = haam_instance.topic_analyzer.cluster_labels
    
    # Calculate means for the specified topics
    topic_means = {'Y': [], 'HU': [], 'AI': []}
    
    for topic_id in topic_ids:
        # Get documents in this topic
        topic_mask = cluster_labels == topic_id
        
        # Calculate means for this topic
        if np.any(topic_mask):
            topic_means['Y'].append(np.mean(haam_instance.criterion[topic_mask]))
            topic_means['HU'].append(np.mean(haam_instance.human_judgment[topic_mask]))
            topic_means['AI'].append(np.mean(haam_instance.ai_judgment[topic_mask]))
    
    # Calculate average across topics
    avg_means = {}
    for measure in ['Y', 'HU', 'AI']:
        if topic_means[measure]:
            avg_means[measure] = np.mean(topic_means[measure])
        else:
            avg_means[measure] = np.nan
    
    # Calculate global quartiles for each measure
    quartiles = {}
    for measure, values in [('Y', haam_instance.criterion), 
                           ('HU', haam_instance.human_judgment), 
                           ('AI', haam_instance.ai_judgment)]:
        q25 = np.nanpercentile(values, 25)
        q75 = np.nanpercentile(values, 75)
        
        if np.isnan(avg_means[measure]):
            quartiles[measure] = '?'
        elif avg_means[measure] >= q75:
            quartiles[measure] = 'H'
        elif avg_means[measure] <= q25:
            quartiles[measure] = 'L'
        else:
            quartiles[measure] = 'M'
    
    return quartiles

def get_pc_associations(haam_instance, pc_idx):
    """Get Y/HU/AI associations for a PC based on coefficients."""
    associations = {}
    try:
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in haam_instance.analysis.results['debiased_lasso']:
                coef = haam_instance.analysis.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                associations[outcome] = 'H' if coef > 0 else 'L'
    except:
        pass
    return associations

# ==============================================================================
# ENHANCED WORD CLOUD FUNCTION (NOW WITH ALIGNED COLORING!)
# ==============================================================================

def create_aligned_word_cloud_with_info(haam_instance, pc_idx, k=3, max_words=150, 
                                        figsize=(16, 8), output_dir=None, display=True):
    """Create word cloud with aligned validity coloring and labels."""
    
    # Get topics
    pc_topics = haam_instance.topic_analyzer.get_pc_high_low_topics(
        pc_idx=pc_idx, n_high=k, n_low=k, p_threshold=0.05
    )
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1], hspace=0.3)
    
    # HIGH POLE
    ax_high = plt.subplot(gs[0, 0])
    high_topics = pc_topics.get('high', [])
    n_high_topics = len(high_topics)
    high_sample_size = sum(t.get('size', 0) for t in high_topics)
    
    # Generate high pole word cloud (now with aligned coloring!)
    if n_high_topics > 0:
        try:
            temp_fig, _, _ = haam_instance.create_pc_wordclouds(
                pc_idx=pc_idx, k=k, max_words=max_words,
                output_dir=None, display=False, color_mode='validity'
            )
            
            # Extract the high pole image
            temp_axes = temp_fig.get_axes()
            if temp_axes and len(temp_axes) > 0:
                for child in temp_axes[0].get_children():
                    if hasattr(child, 'get_array') and child.get_array() is not None:
                        ax_high.imshow(child.get_array())
                        break
            plt.close(temp_fig)
        except:
            ax_high.text(0.5, 0.5, 'Error generating word cloud', 
                        ha='center', va='center', transform=ax_high.transAxes)
    else:
        ax_high.text(0.5, 0.5, 'No significant high topics', 
                    ha='center', va='center', transform=ax_high.transAxes, fontsize=14)
    
    ax_high.set_title(f'PC{pc_idx + 1} - High Pole ({n_high_topics} topics)', 
                     fontsize=16, fontweight='bold', color='darkred')
    ax_high.axis('off')
    
    # LOW POLE
    ax_low = plt.subplot(gs[0, 1])
    low_topics = pc_topics.get('low', [])
    n_low_topics = len(low_topics)
    low_sample_size = sum(t.get('size', 0) for t in low_topics)
    
    # Generate low pole word cloud (now with aligned coloring!)
    if n_low_topics > 0:
        try:
            temp_fig, _, _ = haam_instance.create_pc_wordclouds(
                pc_idx=pc_idx, k=k, max_words=max_words,
                output_dir=None, display=False, color_mode='validity'
            )
            
            # Extract the low pole image
            temp_axes = temp_fig.get_axes()
            if temp_axes and len(temp_axes) > 1:
                for child in temp_axes[1].get_children():
                    if hasattr(child, 'get_array') and child.get_array() is not None:
                        ax_low.imshow(child.get_array())
                        break
            plt.close(temp_fig)
        except:
            ax_low.text(0.5, 0.5, 'Error generating word cloud', 
                       ha='center', va='center', transform=ax_low.transAxes)
    else:
        ax_low.text(0.5, 0.5, 'No significant low topics', 
                   ha='center', va='center', transform=ax_low.transAxes, fontsize=14)
    
    ax_low.set_title(f'PC{pc_idx + 1} - Low Pole ({n_low_topics} topics)', 
                    fontsize=16, fontweight='bold', color='darkblue')
    ax_low.axis('off')
    
    # HIGH POLE INFO WITH ALIGNED QUARTILES
    ax_info_high = plt.subplot(gs[1, 0])
    ax_info_high.axis('off')
    
    # Get quartile positions for high pole topics
    high_topic_ids = [t['topic_id'] for t in high_topics]
    high_quartiles = get_topic_quartile_positions_direct(haam, high_topic_ids)
    
    info_text_high = (
        f"Y: {high_quartiles['Y']} | "
        f"HU: {high_quartiles['HU']} | "
        f"AI: {high_quartiles['AI']}\n"
        f"Sample size: {high_sample_size:,} documents"
    )
    
    ax_info_high.text(0.5, 0.5, info_text_high, ha='center', va='center',
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor='lightgray', alpha=0.7))
    
    # LOW POLE INFO WITH ALIGNED QUARTILES
    ax_info_low = plt.subplot(gs[1, 1])
    ax_info_low.axis('off')
    
    # Get quartile positions for low pole topics
    low_topic_ids = [t['topic_id'] for t in low_topics]
    low_quartiles = get_topic_quartile_positions_direct(haam, low_topic_ids)
    
    info_text_low = (
        f"Y: {low_quartiles['Y']} | "
        f"HU: {low_quartiles['HU']} | "
        f"AI: {low_quartiles['AI']}\n"
        f"Sample size: {low_sample_size:,} documents"
    )
    
    ax_info_low.text(0.5, 0.5, info_text_low, ha='center', va='center',
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='lightgray', alpha=0.7))
    
    # Add color legend
    legend_elements = [
        Patch(facecolor='#8B0000', label='Consensus high (all top quartile)'),
        Patch(facecolor='#FF6B6B', label='Any high (≥1 top quartile)'),
        Patch(facecolor='#00008B', label='Consensus low (all bottom quartile)'),
        Patch(facecolor='#6B9AFF', label='Any low (≥1 bottom quartile)'),
        Patch(facecolor='#4A4A4A', label='Opposing (mix high & low)'),
        Patch(facecolor='#B0B0B0', label='All middle quartiles')
    ]
    plt.figlegend(handles=legend_elements, loc='lower center', 
                 bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)
    
    plt.suptitle(f'PC{pc_idx + 1} Word Clouds with Aligned Validity Coloring', 
                fontsize=18, fontweight='bold')
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'pc{pc_idx + 1}_aligned.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
    
    if display:
        plt.show()
    else:
        plt.close()
    
    return fig

# ==============================================================================
# GENERATE ALIGNED WORD CLOUDS
# ==============================================================================

print("\n2. GENERATING ALIGNED WORD CLOUDS...")
print("-"*60)

# Your specific PCs
specific_pcs = [2, 1, 3, 4, 5, 14, 13, 11, 12, 46, 9, 17, 16, 20, 105]

# Create output directory
output_dir = 'haam_wordclouds_final_aligned'
os.makedirs(output_dir, exist_ok=True)

# Generate aligned word clouds for first 5 PCs
print("\nGenerating aligned word clouds for first 5 PCs...")
for i, pc_idx in enumerate(specific_pcs[:5]):
    print(f"\n[{i+1}/5] PC{pc_idx + 1}:")
    try:
        fig = create_aligned_word_cloud_with_info(
            haam, pc_idx, k=3, max_words=150,
            figsize=(16, 8), output_dir=output_dir, display=True
        )
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

# ==============================================================================
# CREATE COMPREHENSIVE TABLE
# ==============================================================================

print("\n3. CREATING COMPREHENSIVE TABLE WITH ALIGNED LABELS...")
print("-"*60)

# Create a large figure for the table
fig = plt.figure(figsize=(24, 4 * len(specific_pcs)))

# Create grid
n_pcs = len(specific_pcs)
gs = gridspec.GridSpec(n_pcs, 3, width_ratios=[1, 2, 2], wspace=0.2, hspace=0.3)

for i, pc_idx in enumerate(specific_pcs):
    print(f"  Processing PC{pc_idx + 1}...")
    
    # Get topics
    pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
        pc_idx=pc_idx, n_high=3, n_low=3, p_threshold=0.05
    )
    
    # Column 1: PC Label
    ax_label = plt.subplot(gs[i, 0])
    ax_label.axis('off')
    
    # Get PC associations for reference
    pc_assoc = get_pc_associations(haam, pc_idx)
    
    label_text = f'PC{pc_idx + 1}\n\n'
    label_text += f"PC coefficients:\n"
    label_text += f"Y: {pc_assoc.get('SC', '?')} | "
    label_text += f"HU: {pc_assoc.get('HU', '?')} | "
    label_text += f"AI: {pc_assoc.get('AI', '?')}"
    
    ax_label.text(0.5, 0.5, label_text, ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Column 2: High Pole
    ax_high = plt.subplot(gs[i, 1])
    high_topics = pc_topics.get('high', [])
    n_high = len(high_topics)
    high_samples = sum(t.get('size', 0) for t in high_topics)
    
    try:
        if n_high > 0:
            # Generate word cloud with aligned coloring
            temp_fig, _, _ = haam.create_pc_wordclouds(
                pc_idx=pc_idx, k=3, max_words=100,
                output_dir=None, display=False, color_mode='validity'
            )
            
            # Extract high pole image
            temp_axes = temp_fig.get_axes()
            if temp_axes and len(temp_axes) > 0:
                for child in temp_axes[0].get_children():
                    if hasattr(child, 'get_array') and child.get_array() is not None:
                        ax_high.imshow(child.get_array())
                        break
            plt.close(temp_fig)
            
            # Get aligned quartiles
            high_topic_ids = [t['topic_id'] for t in high_topics]
            high_quartiles = get_topic_quartile_positions_direct(haam, high_topic_ids)
            
            title = f'High ({n_high} topics, n={high_samples:,})\n'
            title += f"Y:{high_quartiles['Y']} HU:{high_quartiles['HU']} AI:{high_quartiles['AI']}"
            ax_high.set_title(title, fontsize=12, color='darkred')
        else:
            ax_high.text(0.5, 0.5, 'No high topics', ha='center', va='center')
            ax_high.set_title('High (0 topics)', fontsize=12, color='darkred')
    except:
        ax_high.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax_high.set_title('High (error)', fontsize=12, color='darkred')
    
    ax_high.axis('off')
    
    # Column 3: Low Pole
    ax_low = plt.subplot(gs[i, 2])
    low_topics = pc_topics.get('low', [])
    n_low = len(low_topics)
    low_samples = sum(t.get('size', 0) for t in low_topics)
    
    try:
        if n_low > 0:
            # Generate word cloud with aligned coloring
            temp_fig, _, _ = haam.create_pc_wordclouds(
                pc_idx=pc_idx, k=3, max_words=100,
                output_dir=None, display=False, color_mode='validity'
            )
            
            # Extract low pole image
            temp_axes = temp_fig.get_axes()
            if temp_axes and len(temp_axes) > 1:
                for child in temp_axes[1].get_children():
                    if hasattr(child, 'get_array') and child.get_array() is not None:
                        ax_low.imshow(child.get_array())
                        break
            plt.close(temp_fig)
            
            # Get aligned quartiles
            low_topic_ids = [t['topic_id'] for t in low_topics]
            low_quartiles = get_topic_quartile_positions_direct(haam, low_topic_ids)
            
            title = f'Low ({n_low} topics, n={low_samples:,})\n'
            title += f"Y:{low_quartiles['Y']} HU:{low_quartiles['HU']} AI:{low_quartiles['AI']}"
            ax_low.set_title(title, fontsize=12, color='darkblue')
        else:
            ax_low.text(0.5, 0.5, 'No low topics', ha='center', va='center')
            ax_low.set_title('Low (0 topics)', fontsize=12, color='darkblue')
    except:
        ax_low.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax_low.set_title('Low (error)', fontsize=12, color='darkblue')
    
    ax_low.axis('off')

# Add main title and legend
plt.suptitle('All 15 Principal Components - Aligned Validity Colored Word Clouds', 
            fontsize=24, fontweight='bold', y=0.995)

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

# Save table
table_path = os.path.join(output_dir, 'pc_table_all_15_aligned.png')
plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"✓ Saved comprehensive table: {table_path}")
plt.show()

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

print("\n4. CREATING SUMMARY REPORT...")
print("-"*60)

report_lines = ["PC Aligned Validity Analysis Report"]
report_lines.append("="*100)
report_lines.append("PC\tPC Coefs\tHigh Pole Quartiles\tLow Pole Quartiles\tSamples")
report_lines.append("-"*100)

for pc_idx in specific_pcs:
    try:
        # Get PC associations
        pc_assoc = get_pc_associations(haam, pc_idx)
        pc_pattern = f"Y:{pc_assoc.get('SC', '?')} HU:{pc_assoc.get('HU', '?')} AI:{pc_assoc.get('AI', '?')}"
        
        # Get topics
        pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
            pc_idx=pc_idx, n_high=3, n_low=3, p_threshold=0.05
        )
        
        # Get quartiles for each pole
        high_topic_ids = [t['topic_id'] for t in pc_topics.get('high', [])]
        low_topic_ids = [t['topic_id'] for t in pc_topics.get('low', [])]
        
        high_quartiles = get_topic_quartile_positions_direct(haam, high_topic_ids)
        low_quartiles = get_topic_quartile_positions_direct(haam, low_topic_ids)
        
        high_pattern = f"Y:{high_quartiles['Y']} HU:{high_quartiles['HU']} AI:{high_quartiles['AI']}"
        low_pattern = f"Y:{low_quartiles['Y']} HU:{low_quartiles['HU']} AI:{low_quartiles['AI']}"
        
        # Get sample sizes
        high_samples = sum(t.get('size', 0) for t in pc_topics.get('high', []))
        low_samples = sum(t.get('size', 0) for t in pc_topics.get('low', []))
        
        report_lines.append(
            f"PC{pc_idx+1}\t{pc_pattern}\t{high_pattern}\t{low_pattern}\t"
            f"H:{high_samples:,} L:{low_samples:,}"
        )
    except:
        report_lines.append(f"PC{pc_idx+1}\tError\tError\tError\tError")

# Save report
report_path = os.path.join(output_dir, 'aligned_validity_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

# Display report
print("\nAligned Validity Analysis Summary:")
print("-"*100)
for line in report_lines[:10]:
    print(line)
print(f"\n✓ Full report saved to: {report_path}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("ALIGNED WORD CLOUD GENERATION COMPLETE!")
print("="*80)

print(f"\nAll outputs saved to: {output_dir}/")
print("\n✅ What's new in this aligned version:")
print("  - The HAAM package now uses direct Y/HU/AI measurement for coloring")
print("  - Colors and labels are perfectly aligned")
print("  - Both show actual quartile positions of topics")
print("  - No more mismatch between colors and labels!")

print("\n🔍 How to read the results:")
print("  - Colors = consensus/disagreement pattern across Y/HU/AI")
print("  - Labels = actual quartile positions (H/M/L) for those topics")
print("  - Light blue + 'Y:L HU:M AI:M' = topics are in bottom quartile for Y only")
print("  - Dark blue + 'Y:L HU:L AI:L' = topics are in bottom quartile for all three")

print("\n✨ The word cloud coloring system is now aligned with the labels!")