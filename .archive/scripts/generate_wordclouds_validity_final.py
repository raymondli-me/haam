#!/usr/bin/env python3

"""
Final Enhanced Word Cloud Generation with Validity Coloring
==========================================================
This version fixes empty plots and shows proper Y/HU/AI values for each pole.
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
from matplotlib.patches import Patch
from haam import HAAM
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FINAL ENHANCED WORD CLOUD GENERATION WITH VALIDITY COLORING")
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
# HELPER FUNCTIONS
# ==============================================================================

def get_pc_associations(haam_instance, pc_idx):
    """Get Y/HU/AI associations for a PC."""
    associations = {}
    try:
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in haam_instance.analysis.results['debiased_lasso']:
                coef = haam_instance.analysis.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                associations[outcome] = 'H' if coef > 0 else 'L'
    except:
        pass
    return associations

def get_pole_specific_associations(associations, is_high_pole):
    """Get pole-specific Y/HU/AI values."""
    if not associations:
        return {'SC': '?', 'HU': '?', 'AI': '?'}
    
    if is_high_pole:
        # For high pole, positive coefficient means this pole is high
        return associations
    else:
        # For low pole, flip the associations
        flipped = {}
        for k, v in associations.items():
            flipped[k] = 'L' if v == 'H' else 'H'
        return flipped

# ==============================================================================
# ENHANCED WORD CLOUD FUNCTION WITH PROPER Y/HU/AI VALUES
# ==============================================================================

def create_enhanced_word_cloud_with_info(haam_instance, pc_idx, k=3, max_words=150, 
                                        figsize=(16, 8), output_dir=None, display=True):
    """Create enhanced word cloud with Y/HU/AI info for each pole."""
    
    # Get PC associations
    associations = get_pc_associations(haam_instance, pc_idx)
    
    # Get topics
    pc_topics = haam_instance.topic_analyzer.get_pc_high_low_topics(
        pc_idx=pc_idx, n_high=k, n_low=k, p_threshold=0.05
    )
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    
    # Create grid: 2 rows, 2 columns
    # Top row for word clouds, bottom row for info
    gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1], hspace=0.3)
    
    # HIGH POLE
    ax_high = plt.subplot(gs[0, 0])
    high_topics = pc_topics.get('high', [])
    n_high_topics = len(high_topics)
    high_sample_size = sum(t.get('size', 0) for t in high_topics)
    
    # Generate high pole word cloud
    if n_high_topics > 0:
        try:
            # Create a temporary figure for the word cloud
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
    
    # Generate low pole word cloud
    if n_low_topics > 0:
        try:
            # Create a temporary figure for the word cloud
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
    
    # HIGH POLE INFO
    ax_info_high = plt.subplot(gs[1, 0])
    ax_info_high.axis('off')
    
    high_assoc = get_pole_specific_associations(associations, is_high_pole=True)
    info_text_high = (
        f"Y: {high_assoc.get('SC', '?')} | "
        f"HU: {high_assoc.get('HU', '?')} | "
        f"AI: {high_assoc.get('AI', '?')}\n"
        f"Sample size: {high_sample_size:,} documents"
    )
    
    ax_info_high.text(0.5, 0.5, info_text_high, ha='center', va='center',
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor='lightgray', alpha=0.7))
    
    # LOW POLE INFO
    ax_info_low = plt.subplot(gs[1, 1])
    ax_info_low.axis('off')
    
    low_assoc = get_pole_specific_associations(associations, is_high_pole=False)
    info_text_low = (
        f"Y: {low_assoc.get('SC', '?')} | "
        f"HU: {low_assoc.get('HU', '?')} | "
        f"AI: {low_assoc.get('AI', '?')}\n"
        f"Sample size: {low_sample_size:,} documents"
    )
    
    ax_info_low.text(0.5, 0.5, info_text_low, ha='center', va='center',
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='lightgray', alpha=0.7))
    
    # Add color legend
    legend_elements = [
        Patch(facecolor='#8B0000', label='Consensus high'),
        Patch(facecolor='#FF6B6B', label='Any high'),
        Patch(facecolor='#00008B', label='Consensus low'),
        Patch(facecolor='#6B9AFF', label='Any low'),
        Patch(facecolor='#4A4A4A', label='Opposing'),
        Patch(facecolor='#B0B0B0', label='All middle')
    ]
    plt.figlegend(handles=legend_elements, loc='lower center', 
                 bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=10)
    
    plt.suptitle(f'PC{pc_idx + 1} Word Clouds with Validity Coloring', 
                fontsize=18, fontweight='bold')
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'pc{pc_idx + 1}_enhanced.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {save_path}")
    
    if display:
        plt.show()
    else:
        plt.close()
    
    return fig

# ==============================================================================
# GENERATE ENHANCED WORD CLOUDS
# ==============================================================================

print("\n2. GENERATING ENHANCED WORD CLOUDS WITH PROPER INFO...")
print("-"*60)

# Your specific PCs
specific_pcs = [2, 1, 3, 4, 5, 14, 13, 11, 12, 46, 9, 17, 16, 20, 105]

# Create output directory
output_dir = 'haam_wordclouds_final'
os.makedirs(output_dir, exist_ok=True)

# Generate enhanced word clouds for first 5 PCs
print("\nGenerating enhanced word clouds for first 5 PCs...")
for i, pc_idx in enumerate(specific_pcs[:5]):
    print(f"\n[{i+1}/5] PC{pc_idx + 1}:")
    try:
        fig = create_enhanced_word_cloud_with_info(
            haam, pc_idx, k=3, max_words=150,
            figsize=(16, 8), output_dir=output_dir, display=True
        )
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

# ==============================================================================
# CREATE COMPREHENSIVE TABLE FOR ALL 15 PCs
# ==============================================================================

print("\n3. CREATING COMPREHENSIVE TABLE FOR ALL 15 PCs...")
print("-"*60)

# Create a large figure for the table
fig = plt.figure(figsize=(24, 4 * len(specific_pcs)))

# Create grid
n_pcs = len(specific_pcs)
gs = gridspec.GridSpec(n_pcs, 3, width_ratios=[1, 2, 2], wspace=0.2, hspace=0.3)

for i, pc_idx in enumerate(specific_pcs):
    print(f"  Processing PC{pc_idx + 1}...")
    
    # Get associations and topics
    associations = get_pc_associations(haam, pc_idx)
    pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
        pc_idx=pc_idx, n_high=3, n_low=3, p_threshold=0.05
    )
    
    # Column 1: PC Label with overall pattern
    ax_label = plt.subplot(gs[i, 0])
    ax_label.axis('off')
    
    # Show PC number and overall Y/HU/AI pattern
    label_text = f'PC{pc_idx + 1}\n\n'
    label_text += f"Overall PC pattern:\n"
    label_text += f"Y: {associations.get('SC', '?')} | "
    label_text += f"HU: {associations.get('HU', '?')} | "
    label_text += f"AI: {associations.get('AI', '?')}"
    
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
            # Generate word cloud
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
            
            # Add pole-specific info
            high_assoc = get_pole_specific_associations(associations, is_high_pole=True)
            title = f'High ({n_high} topics, n={high_samples:,})\n'
            title += f"Y:{high_assoc['SC']} HU:{high_assoc['HU']} AI:{high_assoc['AI']}"
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
            # Generate word cloud
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
            
            # Add pole-specific info
            low_assoc = get_pole_specific_associations(associations, is_high_pole=False)
            title = f'Low ({n_low} topics, n={low_samples:,})\n'
            title += f"Y:{low_assoc['SC']} HU:{low_assoc['HU']} AI:{low_assoc['AI']}"
            ax_low.set_title(title, fontsize=12, color='darkblue')
        else:
            ax_low.text(0.5, 0.5, 'No low topics', ha='center', va='center')
            ax_low.set_title('Low (0 topics)', fontsize=12, color='darkblue')
    except:
        ax_low.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax_low.set_title('Low (error)', fontsize=12, color='darkblue')
    
    ax_low.axis('off')

# Add main title and legend
plt.suptitle('All 15 Principal Components - Validity Colored Word Clouds', 
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
table_path = os.path.join(output_dir, 'pc_table_all_15_final.png')
plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"✓ Saved comprehensive table: {table_path}")
plt.show()

# ==============================================================================
# GENERATE REMAINING WORD CLOUDS
# ==============================================================================

print("\n4. BATCH GENERATING REMAINING WORD CLOUDS...")
print("-"*60)

remaining_pcs = specific_pcs[5:]
print(f"Generating {len(remaining_pcs)} remaining PCs...")

for i, pc_idx in enumerate(remaining_pcs):
    print(f"  [{i+1}/{len(remaining_pcs)}] PC{pc_idx + 1}")
    try:
        fig = create_enhanced_word_cloud_with_info(
            haam, pc_idx, k=3, max_words=150,
            figsize=(16, 8), output_dir=output_dir, display=False
        )
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")

# ==============================================================================
# CREATE DETAILED SUMMARY REPORT
# ==============================================================================

print("\n5. CREATING DETAILED SUMMARY REPORT...")
print("-"*60)

report_lines = ["PC Validity Analysis Summary Report"]
report_lines.append("="*100)
report_lines.append("PC\tOverall\tHigh Pole\tLow Pole\tTopics\tSamples")
report_lines.append("-"*100)

for pc_idx in specific_pcs:
    try:
        # Get associations and topics
        associations = get_pc_associations(haam, pc_idx)
        pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
            pc_idx=pc_idx, n_high=3, n_low=3, p_threshold=0.05
        )
        
        # Overall pattern
        overall = f"Y:{associations.get('SC', '?')} HU:{associations.get('HU', '?')} AI:{associations.get('AI', '?')}"
        
        # High pole info
        high_assoc = get_pole_specific_associations(associations, is_high_pole=True)
        high_pattern = f"Y:{high_assoc['SC']} HU:{high_assoc['HU']} AI:{high_assoc['AI']}"
        n_high = len(pc_topics.get('high', []))
        high_samples = sum(t.get('size', 0) for t in pc_topics.get('high', []))
        
        # Low pole info
        low_assoc = get_pole_specific_associations(associations, is_high_pole=False)
        low_pattern = f"Y:{low_assoc['SC']} HU:{low_assoc['HU']} AI:{low_assoc['AI']}"
        n_low = len(pc_topics.get('low', []))
        low_samples = sum(t.get('size', 0) for t in pc_topics.get('low', []))
        
        report_lines.append(
            f"PC{pc_idx+1}\t{overall}\t{high_pattern}\t{low_pattern}\t"
            f"H:{n_high} L:{n_low}\tH:{high_samples:,} L:{low_samples:,}"
        )
    except:
        report_lines.append(f"PC{pc_idx+1}\tError\tError\tError\tError\tError")

# Save report
report_path = os.path.join(output_dir, 'validity_analysis_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

# Display report
print("\nValidity Analysis Summary:")
print("-"*100)
for line in report_lines[:10]:  # Show first few lines
    print(line)
print(f"\n✓ Full report saved to: {report_path}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("FINAL ENHANCED WORD CLOUD GENERATION COMPLETE!")
print("="*80)

print(f"\nAll outputs saved to: {output_dir}/")
print("\n✅ Features included:")
print("  - Proper Y/HU/AI values for EACH pole (not just overall)")
print("  - Number of topics and sample sizes for each pole")
print("  - No empty plots - all word clouds properly displayed")
print("  - Comprehensive table showing all 15 PCs")
print("  - Detailed validity analysis report")

print("\n🔍 Key insights:")
print("  - High pole Y/HU/AI shows what high values of that PC indicate")
print("  - Low pole Y/HU/AI shows what low values of that PC indicate")
print("  - Colors show consensus vs disagreement between measures")
print("  - Sample sizes show how many documents support each pattern")