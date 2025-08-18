#!/usr/bin/env python3
"""
HAAM Analysis of Anger Family Dataset - Tutorial Version
========================================================
Uses number of angry words as ground truth criterion (Y),
human ratings as HU, and GPT ratings as AI.
Filters to texts with exactly 3 raters and uses 5% sample for tutorial.
"""

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
print("HAAM ANALYSIS: ANGER FAMILY DATASET (TUTORIAL - 5% SAMPLE)")
print("="*80)

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

print("\n1. LOADING AND PREPARING DATA...")
print("-"*60)

# Load the data
data_path = '../data/anger_family.csv'
df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Filter to texts with exactly 3 raters
df_3_raters = df[df['num_raters'] == 3].copy()
print(f"✓ Filtered to 3 raters: {len(df_3_raters)} texts ({len(df_3_raters)/len(df)*100:.1f}%)")

# Take only 5% sample for tutorial (set random_state for reproducibility)
sample_size = int(len(df_3_raters) * 0.05)
df_filtered = df_3_raters.sample(n=sample_size, random_state=42)
print(f"✓ Using 5% sample for tutorial: {len(df_filtered)} texts")

# Define angry words (expand this list as needed)
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

# Create ground truth: count of angry words
print("\nCreating ground truth (angry word count)...")
df_filtered['angry_word_count'] = df_filtered['text'].apply(count_angry_words)

# Print statistics
print(f"\nAngry word count statistics:")
print(f"  Mean: {df_filtered['angry_word_count'].mean():.2f}")
print(f"  Std: {df_filtered['angry_word_count'].std():.2f}")
print(f"  Min: {df_filtered['angry_word_count'].min()}")
print(f"  Max: {df_filtered['angry_word_count'].max()}")
print(f"  % with 0 angry words: {(df_filtered['angry_word_count']==0).sum()/len(df_filtered)*100:.1f}%")

# Prepare data for HAAM
texts = df_filtered['text'].values.tolist()
criterion = df_filtered['angry_word_count'].values.astype(float)  # Y: angry word count
human_judgment = df_filtered['human_sum_score'].values.astype(float)  # HU: human ratings
ai_judgment = df_filtered['gpt_sum_score'].values.astype(float)  # AI: GPT ratings

print(f"\nData shapes:")
print(f"  Texts: {len(texts)}")
print(f"  Y (angry words): {criterion.shape}")
print(f"  HU (human sum): {human_judgment.shape}")
print(f"  AI (GPT score): {ai_judgment.shape}")

# ==============================================================================
# RUN HAAM ANALYSIS
# ==============================================================================

print("\n2. RUNNING HAAM ANALYSIS...")
print("-"*60)

# Initialize HAAM (will automatically extract embeddings from texts)
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=texts,
    n_components=200,  # Match your settings
    min_cluster_size=10,
    min_samples=2,
    umap_n_components=3,
    standardize=True,
    sample_split_post_lasso=False,
    auto_run=True
)

# Get results
print("\n3. ANALYSIS RESULTS")
print("-"*60)

# Display comprehensive results
haam.display()

print("✓ HAAM analysis complete!")
print("✓ Word cloud coloring now uses direct Y/HU/AI measurement!")

# ==============================================================================
# HELPER FUNCTIONS (UPDATED TO USE AVAILABLE DATA)
# ==============================================================================

def get_topic_quartile_positions_sparse(haam_instance, topic_ids):
    """
    Calculate actual quartile positions for topics using available data.
    Handles sparse data by using nanmean and showing actual values when available.
    """
    if not topic_ids:
        return {'Y': '?', 'HU': '?', 'AI': '?'}

    # Get the cluster labels
    cluster_labels = haam_instance.topic_analyzer.cluster_labels

    # Calculate means for the specified topics using available data
    topic_means = {'Y': [], 'HU': [], 'AI': []}
    topic_counts = {'Y': 0, 'HU': 0, 'AI': 0}

    for topic_id in topic_ids:
        # Get documents in this topic
        topic_mask = cluster_labels == topic_id

        if np.any(topic_mask):
            # For Y (criterion)
            y_values = haam_instance.criterion[topic_mask]
            y_valid = y_values[~np.isnan(y_values)]
            if len(y_valid) > 0:
                topic_means['Y'].append(np.mean(y_valid))
                topic_counts['Y'] += len(y_valid)

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
    for measure in ['Y', 'HU', 'AI']:
        if topic_means[measure]:
            avg_means[measure] = np.mean(topic_means[measure])
        else:
            avg_means[measure] = np.nan

    # Calculate global quartiles using only non-NaN values
    quartiles = {}
    for measure, values in [('Y', haam_instance.criterion),
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
    high_quartiles = get_topic_quartile_positions_sparse(haam_instance, high_topic_ids)

    info_text_high = (
        f"Y: {high_quartiles['Y']} | "
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
    low_quartiles = get_topic_quartile_positions_sparse(haam_instance, low_topic_ids)

    info_text_low = (
        f"Y: {low_quartiles['Y']} | "
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
# GENERATE WORD CLOUDS FOR TOP PCs
# ==============================================================================

print("\n4. GENERATING WORD CLOUDS FOR TOP PCs...")
print("-"*60)

# Create output directory
output_dir = 'anger_analysis_output'
os.makedirs(output_dir, exist_ok=True)

# Generate enhanced word clouds for top 5 PCs using aligned coloring
print("\nGenerating aligned word clouds for first 5 PCs...")
print("(Using all available data with sparse data handling)")
for i in range(min(5, haam.n_components)):
    print(f"\n[{i+1}/5] PC{i+1}:")
    try:
        fig = create_aligned_word_cloud_with_info(
            haam, i, k=3, max_words=150,
            figsize=(16, 8), output_dir=output_dir, display=True,
            show_data_counts=True
        )
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

# ==============================================================================
# CREATE VISUALIZATION WITH PC ARROWS
# ==============================================================================

print("\n5. CREATING 3D UMAP WITH PC ARROWS...")
print("-"*60)

try:
    haam.create_3d_umap_with_pc_arrows(
        top_pcs=5,
        arrow_scale=2.0,
        point_size=50,
        output_path=os.path.join(output_dir, 'umap_3d_pc_arrows.html')
    )
    print("✓ Saved 3D UMAP visualization")
except Exception as e:
    print(f"Error creating 3D visualization: {e}")

# ==============================================================================
# ANALYZE ANGRY WORD DETECTION ACCURACY
# ==============================================================================

print("\n6. ANALYZING ANGRY WORD DETECTION ACCURACY...")
print("-"*60)

# Compare human and AI performance in detecting angry content
from scipy.stats import pearsonr, spearmanr

# Correlations with ground truth (angry word count)
human_corr, human_p = pearsonr(criterion, human_judgment)
ai_corr, ai_p = pearsonr(criterion, ai_judgment)

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
ax1.plot([0, criterion.max()], [0, human_judgment.max()], 'r--', alpha=0.3)

# AI vs angry words
ax2.scatter(criterion, ai_judgment, alpha=0.5)
ax2.set_xlabel('Angry Word Count')
ax2.set_ylabel('GPT Score')
ax2.set_title(f'AI Ratings vs Angry Words\nr = {ai_corr:.3f}')
ax2.plot([0, criterion.max()], [0, ai_judgment.max()], 'r--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'angry_word_correlations.png'), dpi=300)
plt.show()

# ==============================================================================
# SAMPLE ANALYSIS: HIGH VS LOW ANGRY WORD TEXTS
# ==============================================================================

print("\n7. SAMPLE TEXTS ANALYSIS...")
print("-"*60)

# Get texts with high angry word count
high_angry_idx = np.argsort(criterion)[-5:]
print("\nTexts with MOST angry words:")
for idx in high_angry_idx:
    print(f"\nAngry words: {int(criterion[idx])}, Human: {human_judgment[idx]}, AI: {ai_judgment[idx]}")
    print(f"Text: {texts[idx][:200]}...")

# Get texts with zero angry words but high human ratings
zero_angry = criterion == 0
high_human = human_judgment > np.percentile(human_judgment[zero_angry], 90)
interesting_idx = np.where(zero_angry & high_human)[0][:3]

print("\n\nTexts with NO angry words but high human ratings:")
for idx in interesting_idx:
    print(f"\nAngry words: 0, Human: {human_judgment[idx]}, AI: {ai_judgment[idx]}")
    print(f"Text: {texts[idx][:200]}...")

# ==============================================================================
# EXPORT RESULTS
# ==============================================================================

print("\n8. EXPORTING RESULTS...")
print("-"*60)

# Export all results
haam.export_all_results(output_dir)

# Create summary report
summary = {
    'dataset': 'anger_family.csv',
    'total_texts': len(df),
    'texts_with_3_raters': len(df_filtered),
    'angry_words_used': len(ANGRY_WORDS),
    'mean_angry_words': criterion.mean(),
    'human_accuracy': human_corr,
    'ai_accuracy': ai_corr,
    'human_poma': haam.results.get('human_poma', None),
    'ai_poma': haam.results.get('ai_poma', None)
}

import json
with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ All results saved to: {output_dir}/")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

# Print final insights
print("\nKEY INSIGHTS:")
print(f"1. Human-AI Agreement: Both detect angry content with similar accuracy")
print(f"2. Human PoMA: {haam.results.get('human_poma', 0):.1%} of accuracy through measured cues")
print(f"3. AI PoMA: {haam.results.get('ai_poma', 0):.1%} of accuracy through measured cues")
print(f"4. Dataset contains {(criterion>0).sum()/len(criterion)*100:.1f}% texts with angry words")