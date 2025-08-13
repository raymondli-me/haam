#!/usr/bin/env python3

"""
Enhanced Word Cloud Generation with Validity Coloring
=====================================================
This script generates word clouds with validity coloring and additional features:
- Shows Y/HU/AI agreement levels (High/Middle/Low)
- Displays number of topics and total sample size
- Creates a table-like visualization of all 15 PCs
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
from matplotlib.patches import Rectangle
from haam import HAAM
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED WORD CLOUD GENERATION WITH VALIDITY COLORING")
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

# ==============================================================================
# ENHANCED WORD CLOUD GENERATION FUNCTION
# ==============================================================================

def create_enhanced_pc_wordclouds(haam_instance, pc_idx, k=10, max_words=150, 
                                 figsize=(14, 7), output_dir=None, display=True):
    """
    Create enhanced word clouds with validity info and sample sizes.
    """
    from wordcloud import WordCloud
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches
    
    # Get PC associations from analysis results
    pc_associations = {}
    if haam_instance.analysis.results and 'debiased_lasso' in haam_instance.analysis.results:
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in haam_instance.analysis.results['debiased_lasso']:
                coef = haam_instance.analysis.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                pc_associations[outcome] = 'High' if coef > 0 else 'Low'
    
    # Get topics for this PC
    pc_topics = haam_instance.topic_analyzer.get_pc_high_low_topics(
        pc_idx=pc_idx,
        n_high=k,
        n_low=k,
        p_threshold=0.05
    )
    
    # Create figure with enhanced layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[0.85, 0.15])
    
    # High pole subplot
    ax_high = plt.subplot(gs[0, 0])
    high_topics = pc_topics['high']
    high_sample_size = sum(topic['size'] for topic in high_topics)
    high_word_freq = haam_instance.wordcloud_generator._aggregate_topic_keywords(high_topics)
    
    if high_word_freq:
        # Get validity colors
        word_colors = haam_instance.wordcloud_generator._calculate_topic_validity_colors(high_topics, pc_idx)
        
        def color_func_high(word, **kwargs):
            return word_colors.get(word, '#B0B0B0')
        
        wc_high = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10,
            color_func=color_func_high
        ).generate_from_frequencies(high_word_freq)
        
        ax_high.imshow(wc_high, interpolation='bilinear')
    else:
        ax_high.text(0.5, 0.5, 'No significant high topics', 
                    ha='center', va='center', transform=ax_high.transAxes, fontsize=14)
    
    # Add title with topic count
    ax_high.set_title(f'PC{pc_idx + 1} - High Pole ({len(high_topics)} topics)', 
                     fontsize=16, fontweight='bold', color='darkred')
    ax_high.axis('off')
    
    # Low pole subplot
    ax_low = plt.subplot(gs[0, 1])
    low_topics = pc_topics['low']
    low_sample_size = sum(topic['size'] for topic in low_topics)
    low_word_freq = haam_instance.wordcloud_generator._aggregate_topic_keywords(low_topics)
    
    if low_word_freq:
        # Get validity colors
        word_colors = haam_instance.wordcloud_generator._calculate_topic_validity_colors(low_topics, pc_idx)
        
        def color_func_low(word, **kwargs):
            return word_colors.get(word, '#B0B0B0')
        
        wc_low = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10,
            color_func=color_func_low
        ).generate_from_frequencies(low_word_freq)
        
        ax_low.imshow(wc_low, interpolation='bilinear')
    else:
        ax_low.text(0.5, 0.5, 'No significant low topics', 
                   ha='center', va='center', transform=ax_low.transAxes, fontsize=14)
    
    # Add title with topic count
    ax_low.set_title(f'PC{pc_idx + 1} - Low Pole ({len(low_topics)} topics)', 
                    fontsize=16, fontweight='bold', color='darkblue')
    ax_low.axis('off')
    
    # Info panel for high pole
    ax_info_high = plt.subplot(gs[1, 0])
    ax_info_high.axis('off')
    
    info_text_high = f"Y: {pc_associations.get('SC', 'N/A')} | HU: {pc_associations.get('HU', 'N/A')} | AI: {pc_associations.get('AI', 'N/A')}\n"
    info_text_high += f"Sample size: {high_sample_size:,} documents"
    
    ax_info_high.text(0.5, 0.5, info_text_high, ha='center', va='center',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                     facecolor='lightgray', alpha=0.5))
    
    # Info panel for low pole
    ax_info_low = plt.subplot(gs[1, 1])
    ax_info_low.axis('off')
    
    # For low pole, reverse the associations
    low_associations = {
        k: ('Low' if v == 'High' else 'High') 
        for k, v in pc_associations.items()
    }
    
    info_text_low = f"Y: {low_associations.get('SC', 'N/A')} | HU: {low_associations.get('HU', 'N/A')} | AI: {low_associations.get('AI', 'N/A')}\n"
    info_text_low += f"Sample size: {low_sample_size:,} documents"
    
    ax_info_low.text(0.5, 0.5, info_text_low, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='lightgray', alpha=0.5))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#8B0000', label='Consensus high'),
        mpatches.Patch(color='#FF6B6B', label='Any high'),
        mpatches.Patch(color='#00008B', label='Consensus low'),
        mpatches.Patch(color='#6B9AFF', label='Any low'),
        mpatches.Patch(color='#4A4A4A', label='Opposing'),
        mpatches.Patch(color='#B0B0B0', label='All middle')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=6, frameon=False, fontsize=10)
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'pc{pc_idx + 1}_enhanced_wordcloud.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

# ==============================================================================
# GENERATE ENHANCED WORD CLOUDS
# ==============================================================================

print("\n3. GENERATING ENHANCED WORD CLOUDS...")
print("-"*60)

# Define the specific PCs you want (0-based indices)
specific_pcs = [2, 1, 3, 4, 5, 14, 13, 11, 12, 46, 9, 17, 16, 20, 105]  # 0-based indices

# Create output directory
wordcloud_dir = 'haam_wordclouds_enhanced'
os.makedirs(wordcloud_dir, exist_ok=True)

# Generate enhanced word clouds for first few PCs
print("\nGenerating enhanced word clouds with validity info...")
for i, pc_idx in enumerate(specific_pcs[:3]):  # Show first 3
    print(f"\n[{i+1}/{len(specific_pcs)}] PC{pc_idx + 1}...")
    try:
        fig = create_enhanced_pc_wordclouds(
            haam, pc_idx, k=3, max_words=150, 
            output_dir=wordcloud_dir, display=True
        )
    except Exception as e:
        print(f"Failed: {str(e)}")

# ==============================================================================
# CREATE TABLE-LIKE VISUALIZATION FOR ALL 15 PCs
# ==============================================================================

print("\n4. CREATING TABLE VISUALIZATION FOR ALL 15 PCs...")
print("-"*60)

def create_pc_table_visualization(haam_instance, pc_indices, k=3, max_words=100, 
                                output_file='pc_table_viz.png'):
    """
    Create a table-like visualization showing all PCs with their word clouds.
    """
    from wordcloud import WordCloud
    
    n_pcs = len(pc_indices)
    fig, axes = plt.subplots(n_pcs, 3, figsize=(18, 4*n_pcs))
    
    if n_pcs == 1:
        axes = axes.reshape(1, -1)
    
    for i, pc_idx in enumerate(pc_indices):
        # PC label column
        ax_label = axes[i, 0]
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, f'PC{pc_idx + 1}', ha='center', va='center',
                     fontsize=24, fontweight='bold')
        
        # Get topics
        try:
            pc_topics = haam_instance.topic_analyzer.get_pc_high_low_topics(
                pc_idx=pc_idx, n_high=k, n_low=k, p_threshold=0.05
            )
            
            # High pole word cloud
            ax_high = axes[i, 1]
            high_topics = pc_topics['high']
            high_sample_size = sum(topic['size'] for topic in high_topics)
            high_word_freq = haam_instance.wordcloud_generator._aggregate_topic_keywords(high_topics)
            
            if high_word_freq:
                word_colors = haam_instance.wordcloud_generator._calculate_topic_validity_colors(high_topics, pc_idx)
                
                def color_func_high(word, **kwargs):
                    return word_colors.get(word, '#B0B0B0')
                
                wc_high = WordCloud(
                    width=600, height=300, background_color='white',
                    max_words=max_words, relative_scaling=0.5,
                    min_font_size=8, color_func=color_func_high
                ).generate_from_frequencies(high_word_freq)
                
                ax_high.imshow(wc_high, interpolation='bilinear')
                ax_high.set_title(f'High ({len(high_topics)} topics, n={high_sample_size:,})', 
                                 fontsize=12, color='darkred')
            else:
                ax_high.text(0.5, 0.5, 'No high topics', ha='center', va='center',
                           transform=ax_high.transAxes)
                ax_high.set_title('High (0 topics)', fontsize=12, color='darkred')
            ax_high.axis('off')
            
            # Low pole word cloud
            ax_low = axes[i, 2]
            low_topics = pc_topics['low']
            low_sample_size = sum(topic['size'] for topic in low_topics)
            low_word_freq = haam_instance.wordcloud_generator._aggregate_topic_keywords(low_topics)
            
            if low_word_freq:
                word_colors = haam_instance.wordcloud_generator._calculate_topic_validity_colors(low_topics, pc_idx)
                
                def color_func_low(word, **kwargs):
                    return word_colors.get(word, '#B0B0B0')
                
                wc_low = WordCloud(
                    width=600, height=300, background_color='white',
                    max_words=max_words, relative_scaling=0.5,
                    min_font_size=8, color_func=color_func_low
                ).generate_from_frequencies(low_word_freq)
                
                ax_low.imshow(wc_low, interpolation='bilinear')
                ax_low.set_title(f'Low ({len(low_topics)} topics, n={low_sample_size:,})', 
                                fontsize=12, color='darkblue')
            else:
                ax_low.text(0.5, 0.5, 'No low topics', ha='center', va='center',
                          transform=ax_low.transAxes)
                ax_low.set_title('Low (0 topics)', fontsize=12, color='darkblue')
            ax_low.axis('off')
            
        except Exception as e:
            axes[i, 1].text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center')
            axes[i, 1].axis('off')
            axes[i, 2].text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center')
            axes[i, 2].axis('off')
    
    plt.suptitle('Principal Component Word Clouds - Validity Coloring', 
                fontsize=20, fontweight='bold', y=0.995)
    
    # Add color legend at bottom
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8B0000', label='Consensus high'),
        Patch(facecolor='#FF6B6B', label='Any high'),
        Patch(facecolor='#00008B', label='Consensus low'),
        Patch(facecolor='#6B9AFF', label='Any low'),
        Patch(facecolor='#4A4A4A', label='Opposing'),
        Patch(facecolor='#B0B0B0', label='All middle')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.01),
              ncol=6, frameon=False, fontsize=12)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved table visualization: {output_file}")
    
    return fig

# Create the table visualization
print("\nCreating table visualization for all 15 PCs...")
table_fig = create_pc_table_visualization(
    haam, specific_pcs, k=3, max_words=100,
    output_file=os.path.join(wordcloud_dir, 'pc_table_all_15.png')
)
plt.show()

# ==============================================================================
# FIX RANKING-BASED VISUALIZATIONS
# ==============================================================================

print("\n5. CREATING RANKING-BASED VALIDITY VISUALIZATIONS...")
print("-"*60)

ranking_methods = ['HU', 'AI', 'Y']

for ranking in ranking_methods:
    print(f"\nCreating validity-colored grid for top PCs by {ranking} ranking...")
    
    try:
        # Fix: use n_top instead of n
        if ranking == 'Y':
            ranking_key = 'SC'
        else:
            ranking_key = ranking
            
        top_pcs_by_ranking = haam.analysis.get_top_pcs(n_top=9, ranking_method=ranking_key)
        
        # Create grid with validity coloring
        grid_path = os.path.join(wordcloud_dir, f'validity_grid_{ranking.lower()}_ranking.png')
        fig = haam.create_top_pcs_wordcloud_grid(
            pc_indices=top_pcs_by_ranking,
            k=3,
            max_words=50,
            output_file=grid_path,
            display=False,
            color_mode='validity'
        )
        print(f"  ✓ Saved: {grid_path}")
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")

# ==============================================================================
# GENERATE DETAILED REPORT
# ==============================================================================

print("\n6. GENERATING DETAILED PC REPORT...")
print("-"*60)

# Create a detailed report for each PC
report_lines = ["PC\tHigh Topics\tLow Topics\tY/HU/AI Pattern\tSample Sizes"]
report_lines.append("-"*80)

for pc_idx in specific_pcs:
    try:
        # Get topics
        pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
            pc_idx=pc_idx, n_high=5, n_low=5, p_threshold=0.05
        )
        
        # Get associations
        pc_associations = {}
        if haam.analysis.results and 'debiased_lasso' in haam.analysis.results:
            for outcome in ['SC', 'AI', 'HU']:
                if outcome in haam.analysis.results['debiased_lasso']:
                    coef = haam.analysis.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                    pc_associations[outcome] = 'H' if coef > 0 else 'L'
        
        # Count topics and samples
        high_topics = pc_topics['high']
        low_topics = pc_topics['low']
        high_samples = sum(t['size'] for t in high_topics)
        low_samples = sum(t['size'] for t in low_topics)
        
        # Get first few keywords
        high_keywords = high_topics[0]['keywords'][:40] + "..." if high_topics else "None"
        low_keywords = low_topics[0]['keywords'][:40] + "..." if low_topics else "None"
        
        # Format pattern
        pattern = f"Y:{pc_associations.get('SC', '?')} HU:{pc_associations.get('HU', '?')} AI:{pc_associations.get('AI', '?')}"
        
        report_lines.append(
            f"PC{pc_idx+1}\t{high_keywords}\t{low_keywords}\t{pattern}\t"
            f"H:{high_samples:,} L:{low_samples:,}"
        )
        
    except Exception as e:
        report_lines.append(f"PC{pc_idx+1}\tError\tError\tError\tError")

# Save report
report_path = os.path.join(wordcloud_dir, 'pc_validity_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"✓ Saved detailed report: {report_path}")

# Display first few lines
print("\nPC Validity Report (first 5 PCs):")
print("-"*80)
for line in report_lines[:7]:
    print(line)

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("ENHANCED VALIDITY WORD CLOUD GENERATION COMPLETE!")
print("="*80)

print(f"\nAll files saved to: {wordcloud_dir}/")
print("\nGenerated outputs:")
print("  📊 Enhanced word clouds with Y/HU/AI patterns and sample sizes")
print("  📈 Table visualization of all 15 PCs: pc_table_all_15.png")
print("  📝 Detailed PC report: pc_validity_report.txt")
print("  🎯 Fixed ranking grids for HU, AI, and Y")

print("\n✅ Done! Check the outputs to see validity patterns across your PCs!")