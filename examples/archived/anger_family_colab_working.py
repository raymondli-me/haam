#!/usr/bin/env python3
"""
Anger Family Analysis - Working Colab Version
============================================
End-to-end analysis with word cloud generation for anger_family.csv
Using 5% sample for faster tutorial execution
Uses X (angry word count) as criterion, HU (human ratings), AI (GPT ratings)

This version fixes the display() issue and provides complete working analysis.
"""

# Install required packages
print("Installing packages...")
import subprocess
import sys

# Install packages quietly
subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/raymondli-me/haam.git", "-q"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os
import re
from haam import HAAM
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANGER FAMILY ANALYSIS WITH VALIDITY COLORING (TUTORIAL - 5% SAMPLE)")
print("="*80)

# ==============================================================================
# LOAD DATA FROM GITHUB
# ==============================================================================

print("\n1. LOADING DATA FROM GITHUB...")
print("-"*60)

# Load anger_family.csv directly from GitHub
github_url = 'https://raw.githubusercontent.com/raymondli-me/haam/main/data/anger_family.csv'
df = pd.read_csv(github_url)
print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Filter to texts with exactly 3 raters
df_3_raters = df[df['num_raters'] == 3].copy()
print(f"✓ Filtered to 3 raters: {len(df_3_raters)} texts ({len(df_3_raters)/len(df)*100:.1f}%)")

# Take only 5% sample for tutorial (set random_state for reproducibility)
sample_size = int(len(df_3_raters) * 0.05)
df_filtered = df_3_raters.sample(n=sample_size, random_state=42)
print(f"✓ Using 5% sample for tutorial: {len(df_filtered)} texts")

# ==============================================================================
# CREATE GROUND TRUTH: ANGRY WORD COUNT
# ==============================================================================

print("\n2. CREATING GROUND TRUTH (ANGRY WORD COUNT)...")
print("-"*60)

# Define angry words
ANGRY_WORDS = [
    # Direct anger words
    'angry', 'anger', 'mad', 'furious', 'rage', 'raging', 'pissed', 'annoyed', 
    'irritated', 'frustrated', 'infuriated', 'livid', 'irate', 'outraged',
    'enraged', 'seething', 'fuming', 'incensed', 'wrathful', 'hostile',
    
    # Aggressive words
    'hate', 'hatred', 'despise', 'loathe', 'detest', 'abhor', 'disgust',
    'aggressive', 'attack', 'fight', 'violence', 'violent', 'assault',
    'threat', 'threaten', 'kill', 'murder', 'hurt', 'harm', 'destroy',
    
    # Expletives and intensifiers
    'damn', 'dammit', 'hell', 'fuck', 'fucking', 'shit', 'crap', 'bastard',
    'bitch', 'asshole', 'idiot', 'stupid', 'moron', 'jerk', 'screw',
    
    # Conflict words
    'argue', 'argument', 'conflict', 'confrontation', 'quarrel', 'dispute',
    'clash', 'feud', 'antagonize', 'provoke', 'insult', 'offend',
    
    # Negative emotional states
    'upset', 'bothered', 'disturbed', 'agitated', 'worked up', 'ticked off',
    'fed up', 'sick of', 'had enough', 'lose it', 'blow up', 'snap'
]

def count_angry_words(text):
    """Count the number of angry words in a text."""
    if pd.isna(text):
        return 0
    
    text_lower = text.lower()
    count = 0
    
    for word in ANGRY_WORDS:
        # Use word boundaries to match whole words
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text_lower)
        count += len(matches)
    
    return count

# Create ground truth
df_filtered['angry_word_count'] = df_filtered['text'].apply(count_angry_words)

print(f"Angry word count statistics:")
print(f"  Mean: {df_filtered['angry_word_count'].mean():.2f}")
print(f"  Std: {df_filtered['angry_word_count'].std():.2f}")
print(f"  Range: {df_filtered['angry_word_count'].min()}-{df_filtered['angry_word_count'].max()}")
print(f"  % with 0 angry words: {(df_filtered['angry_word_count']==0).sum()/len(df_filtered)*100:.1f}%")

# Prepare data for HAAM
texts = df_filtered['text'].values.tolist()
criterion = df_filtered['angry_word_count'].values.astype(float)  # X: angry word count
human_judgment = df_filtered['human_sum_score'].values.astype(float)  # HU: human ratings
ai_judgment = df_filtered['gpt_sum_score'].values.astype(float)  # AI: GPT ratings

# Print data availability
print(f"\nData availability:")
print(f"  X (angry words): {(~np.isnan(criterion)).sum():,} ({(~np.isnan(criterion)).sum()/len(criterion)*100:.1f}%)")
print(f"  HU (human sum): {(~np.isnan(human_judgment)).sum():,} ({(~np.isnan(human_judgment)).sum()/len(human_judgment)*100:.1f}%)")
print(f"  AI (GPT score): {(~np.isnan(ai_judgment)).sum():,} ({(~np.isnan(ai_judgment)).sum()/len(ai_judgment)*100:.1f}%)")

# ==============================================================================
# RUN HAAM ANALYSIS
# ==============================================================================

print("\n3. RUNNING HAAM ANALYSIS...")
print("-"*60)

# Initialize HAAM (will automatically extract embeddings from texts)
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
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
print("✓ Word cloud coloring now uses direct X/HU/AI measurement!")

# Display results using the analysis object directly
if hasattr(haam.analysis, 'display_global_statistics'):
    haam.analysis.display_global_statistics()
else:
    # Fallback: print basic results
    print("\nBasic Results:")
    if 'debiased_lasso' in haam.analysis.results:
        for outcome in ['X', 'AI', 'HU']:
            if outcome in haam.analysis.results['debiased_lasso']:
                res = haam.analysis.results['debiased_lasso'][outcome]
                print(f"{outcome} model: {res.get('n_selected', 0)} PCs selected, R²(CV) = {res.get('r2_cv', 0):.3f}")

# ==============================================================================
# HELPER FUNCTIONS (UPDATED TO USE AVAILABLE DATA)
# ==============================================================================

def get_topic_quartile_positions_sparse(haam_instance, topic_ids):
    """
    Calculate actual quartile positions for topics using available data.
    Handles sparse data by using nanmean and showing actual values when available.
    """
    if not topic_ids:
        return {'X': '?', 'HU': '?', 'AI': '?'}

    # Get the cluster labels
    cluster_labels = haam_instance.topic_analyzer.cluster_labels

    # Calculate means for the specified topics using available data
    topic_means = {'X': [], 'HU': [], 'AI': []}
    topic_counts = {'X': 0, 'HU': 0, 'AI': 0}

    for topic_id in topic_ids:
        # Get documents in this topic
        topic_mask = cluster_labels == topic_id

        if np.any(topic_mask):
            # For X (criterion)
            x_values = haam_instance.criterion[topic_mask]
            x_valid = x_values[~np.isnan(x_values)]
            if len(x_valid) > 0:
                topic_means['X'].append(np.mean(x_valid))
                topic_counts['X'] += len(x_valid)

            # For HU (human judgment) - use whatever data is available
            hu_values = haam_instance.human_judgment[topic_mask]
            hu_valid = hu_values[~np.isnan(hu_values)]
            if len(hu_valid) > 0:
                topic_means['HU'].append(np.mean(hu_valid))
                topic_counts['HU'] += len(hu_valid)

            # For AI (ai judgment)
            ai_values = haam_instance.ai_judgment[topic_mask]
            ai_valid = ai_values[~np.isnan(ai_values)]
            if len(ai_valid) > 0:
                topic_means['AI'].append(np.mean(ai_valid))
                topic_counts['AI'] += len(ai_valid)

    # Calculate average across topics (only for topics with data)
    avg_means = {}
    for measure in ['X', 'HU', 'AI']:
        if topic_means[measure]:
            avg_means[measure] = np.mean(topic_means[measure])
        else:
            avg_means[measure] = np.nan

    # Calculate global quartiles using only non-NaN values
    quartiles = {}
    for measure, values in [('X', haam_instance.criterion),
                           ('HU', haam_instance.human_judgment),
                           ('AI', haam_instance.ai_judgment)]:
        # Get non-NaN values for quartile calculation
        valid_values = values[~np.isnan(values)]

        if len(valid_values) > 0:
            q25 = np.percentile(valid_values, 25)
            q75 = np.percentile(valid_values, 75)

            if np.isnan(avg_means[measure]):
                # No data for this measure in these topics
                quartiles[measure] = '?'
            elif avg_means[measure] >= q75:
                quartiles[measure] = 'H'
            elif avg_means[measure] <= q25:
                quartiles[measure] = 'L'
            else:
                quartiles[measure] = 'M'
        else:
            # No valid data for this measure at all
            quartiles[measure] = '?'

    # Add data counts for transparency
    quartiles['_counts'] = topic_counts

    return quartiles

def get_pc_associations(haam_instance, pc_idx):
    """Get X/HU/AI associations for a PC based on coefficients."""
    associations = {}
    try:
        for outcome in ['X', 'AI', 'HU']:
            if outcome in haam_instance.analysis.results['debiased_lasso']:
                coef = haam_instance.analysis.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                associations[outcome] = 'H' if coef > 0 else 'L'
    except:
        pass
    return associations

# ==============================================================================
# ENHANCED WORD CLOUD FUNCTION (WITH SPARSE DATA HANDLING)
# ==============================================================================

def create_aligned_word_cloud_with_info(haam_instance, pc_idx, k=3, max_words=150,
                                        figsize=(16, 8), output_dir=None, display=True,
                                        show_data_counts=True):
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

    # HIGH POLE INFO WITH SPARSE DATA HANDLING
    ax_info_high = plt.subplot(gs[1, 0])
    ax_info_high.axis('off')

    # Get quartile positions for high pole topics
    high_topic_ids = [t['topic_id'] for t in high_topics]
    high_quartiles = get_topic_quartile_positions_sparse(haam, high_topic_ids)

    info_text_high = (
        f"X: {high_quartiles['X']} | "
        f"HU: {high_quartiles['HU']} | "
        f"AI: {high_quartiles['AI']}\n"
        f"Sample size: {high_sample_size:,} documents"
    )

    # Add data counts if requested
    if show_data_counts and '_counts' in high_quartiles:
        counts = high_quartiles['_counts']
        if counts['HU'] < 10:  # Show warning for sparse HU data
            info_text_high += f"\n(HU: n={counts['HU']})"

    ax_info_high.text(0.5, 0.5, info_text_high, ha='center', va='center',
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.5",
                     facecolor='lightgray', alpha=0.7))

    # LOW POLE INFO WITH SPARSE DATA HANDLING
    ax_info_low = plt.subplot(gs[1, 1])
    ax_info_low.axis('off')

    # Get quartile positions for low pole topics
    low_topic_ids = [t['topic_id'] for t in low_topics]
    low_quartiles = get_topic_quartile_positions_sparse(haam, low_topic_ids)

    info_text_low = (
        f"X: {low_quartiles['X']} | "
        f"HU: {low_quartiles['HU']} | "
        f"AI: {low_quartiles['AI']}\n"
        f"Sample size: {low_sample_size:,} documents"
    )

    # Add data counts if requested
    if show_data_counts and '_counts' in low_quartiles:
        counts = low_quartiles['_counts']
        if counts['HU'] < 10:  # Show warning for sparse HU data
            info_text_low += f"\n(HU: n={counts['HU']})"

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

print("\n4. GENERATING ALIGNED WORD CLOUDS...")
print("-"*60)

# Create output directory
output_dir = 'anger_wordclouds_aligned'
os.makedirs(output_dir, exist_ok=True)

# Generate aligned word clouds for first 5 PCs
print("\nGenerating aligned word clouds for first 5 PCs...")
for i in range(min(5, haam.n_components)):
    print(f"\n[{i+1}/5] PC{i + 1}:")
    try:
        fig = create_aligned_word_cloud_with_info(
            haam, i, k=3, max_words=150,
            figsize=(16, 8), output_dir=output_dir, display=True,
            show_data_counts=True
        )
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

# ==============================================================================
# CREATE COMPREHENSIVE TABLE FOR TOP 10 PCs
# ==============================================================================

print("\n5. CREATING COMPREHENSIVE TABLE WITH ALIGNED LABELS...")
print("-"*60)

# Use top 10 PCs for the anger dataset
n_pcs_to_show = min(10, haam.n_components)

# Create a large figure for the table
fig = plt.figure(figsize=(24, 4 * n_pcs_to_show))

# Create grid
gs = gridspec.GridSpec(n_pcs_to_show, 3, width_ratios=[1, 2, 2], wspace=0.2, hspace=0.3)

for i in range(n_pcs_to_show):
    print(f"  Processing PC{i + 1}...")

    # Get topics
    pc_topics = haam.topic_analyzer.get_pc_high_low_topics(
        pc_idx=i, n_high=3, n_low=3, p_threshold=0.05
    )

    # Column 1: PC Label
    ax_label = plt.subplot(gs[i, 0])
    ax_label.axis('off')

    # Get PC associations for reference
    pc_assoc = get_pc_associations(haam, i)

    label_text = f'PC{i + 1}\n\n'
    label_text += f"PC coefficients:\n"
    label_text += f"X: {pc_assoc.get('X', '?')} | "
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
                pc_idx=i, k=3, max_words=100,
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
            high_quartiles = get_topic_quartile_positions_sparse(haam, high_topic_ids)

            title = f'High ({n_high} topics, n={high_samples:,})\n'
            title += f"X:{high_quartiles['X']} HU:{high_quartiles['HU']} AI:{high_quartiles['AI']}"

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
                pc_idx=i, k=3, max_words=100,
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
            low_quartiles = get_topic_quartile_positions_sparse(haam, low_topic_ids)

            title = f'Low ({n_low} topics, n={low_samples:,})\n'
            title += f"X:{low_quartiles['X']} HU:{low_quartiles['HU']} AI:{low_quartiles['AI']}"

            ax_low.set_title(title, fontsize=12, color='darkblue')
        else:
            ax_low.text(0.5, 0.5, 'No low topics', ha='center', va='center')
            ax_low.set_title('Low (0 topics)', fontsize=12, color='darkblue')
    except:
        ax_low.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax_low.set_title('Low (error)', fontsize=12, color='darkblue')

    ax_low.axis('off')

# Add main title and legend
plt.suptitle(f'Top {n_pcs_to_show} Principal Components - Anger Analysis Word Clouds',
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
table_path = os.path.join(output_dir, f'pc_table_top_{n_pcs_to_show}_aligned.png')
plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"✓ Saved comprehensive table: {table_path}")
plt.show()

# ==============================================================================
# ANALYZE ANGRY WORD DETECTION ACCURACY
# ==============================================================================

print("\n6. ANALYZING ANGRY WORD DETECTION ACCURACY...")
print("-"*60)

# Compare human and AI performance
from scipy.stats import pearsonr

# Correlations with ground truth (angry word count)
# Only use non-NaN values for correlation
mask = ~(np.isnan(criterion) | np.isnan(human_judgment))
if mask.sum() > 10:
    human_corr, human_p = pearsonr(criterion[mask], human_judgment[mask])
else:
    human_corr, human_p = 0, 1

mask = ~(np.isnan(criterion) | np.isnan(ai_judgment))
if mask.sum() > 10:
    ai_corr, ai_p = pearsonr(criterion[mask], ai_judgment[mask])
else:
    ai_corr, ai_p = 0, 1

print(f"Correlation with angry word count:")
print(f"  Human ratings: r = {human_corr:.3f} (p = {human_p:.4f})")
print(f"  AI ratings:    r = {ai_corr:.3f} (p = {ai_p:.4f})")

# Create scatter plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Human vs angry words
ax1.scatter(criterion, human_judgment, alpha=0.5)
ax1.set_xlabel('Angry Word Count')
ax1.set_ylabel('Human Sum Score')
ax1.set_title(f'Human Ratings vs Angry Words\nr = {human_corr:.3f}')

# AI vs angry words
ax2.scatter(criterion, ai_judgment, alpha=0.5)
ax2.set_xlabel('Angry Word Count')
ax2.set_ylabel('GPT Score')
ax2.set_title(f'AI Ratings vs Angry Words\nr = {ai_corr:.3f}')

plt.tight_layout()
plt.show()

# ==============================================================================
# CREATE 3D VISUALIZATION
# ==============================================================================

print("\n7. CREATING 3D UMAP VISUALIZATION...")
print("-"*60)

try:
    output_path = os.path.join(output_dir, 'umap_3d_pc_arrows.html')
    haam.create_3d_umap_with_pc_arrows(
        top_pcs=5,
        arrow_scale=2.0,
        point_size=50,
        output_path=output_path
    )
    print(f"✓ Created 3D UMAP visualization: {output_path}")
except Exception as e:
    print(f"Error creating 3D visualization: {e}")

# ==============================================================================
# EXTRACT KEY METRICS FROM RESULTS
# ==============================================================================

# Get PoMA values if available
human_poma = 0
ai_poma = 0

try:
    if 'mediation_analysis' in haam.analysis.results:
        if 'HU' in haam.analysis.results['mediation_analysis']:
            med = haam.analysis.results['mediation_analysis']['HU']
            if med.get('total_effect', 0) != 0:
                human_poma = (med.get('indirect_effect', 0) / med['total_effect'])
        
        if 'AI' in haam.analysis.results['mediation_analysis']:
            med = haam.analysis.results['mediation_analysis']['AI']
            if med.get('total_effect', 0) != 0:
                ai_poma = (med.get('indirect_effect', 0) / med['total_effect'])
except:
    pass

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("ANGER ANALYSIS COMPLETE!")
print("="*80)

print(f"\nKEY INSIGHTS:")
print(f"1. Dataset: {len(df_filtered):,} texts with 3 raters each")
print(f"2. {(criterion>0).sum()/len(criterion)*100:.1f}% of texts contain angry words")
print(f"3. Human accuracy: r = {human_corr:.3f} with angry word count")
print(f"4. AI accuracy: r = {ai_corr:.3f} with angry word count")
print(f"5. Human PoMA: {human_poma:.1%} of accuracy through measured cues")
print(f"6. AI PoMA: {ai_poma:.1%} of accuracy through measured cues")

print(f"\nAll outputs saved to: {output_dir}/")

# Display some example texts
print("\n" + "="*80)
print("EXAMPLE TEXTS WITH HIGH ANGRY WORD COUNT:")
print("="*80)

# Get texts with high angry word count
high_angry_idx = np.argsort(criterion)[-3:]
for idx in high_angry_idx:
    print(f"\nAngry words: {int(criterion[idx])}, Human: {human_judgment[idx]}, AI: {ai_judgment[idx]}")
    print(f"Text: {texts[idx][:150]}...")

print("\n" + "="*80)
print("EXAMPLE TEXTS WITH NO ANGRY WORDS BUT HIGH RATINGS:")
print("="*80)

# Get texts with zero angry words but high human ratings
zero_angry = criterion == 0
if np.any(zero_angry):
    high_human = human_judgment > np.percentile(human_judgment[zero_angry], 90)
    interesting_idx = np.where(zero_angry & high_human)[0][:3]
    
    for idx in interesting_idx:
        print(f"\nAngry words: 0, Human: {human_judgment[idx]}, AI: {ai_judgment[idx]}")
        print(f"Text: {texts[idx][:150]}...")

print("\n✓ Script completed successfully!")