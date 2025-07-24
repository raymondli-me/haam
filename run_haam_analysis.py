#!/usr/bin/env python3
"""
Clean HAAM Analysis Script (v1.1.0)
===================================
Run complete HAAM analysis with the new enhanced parameters and features.
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from haam import HAAM

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 9
})

print("="*80)
print("HAAM ANALYSIS v1.1.0 - WITH ENHANCED CLUSTERING AND VISUALIZATIONS")
print("="*80)

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n1. LOADING DATA...")
print("-"*60)

# Load the complete dataset with text
filename = 'essay_embeddings_minilm_with_text.csv'
data_path = f'/content/drive/MyDrive/2025_06_30_anonymized_data_dmllme_2025_06_30/{filename}'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)
print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Extract data
texts = df['text'].values.tolist()
embedding_cols = [f'embed_dim_{i}' for i in range(384)]
embeddings = df[embedding_cols].values
social_class = df['social_class'].astype(float).values
ai_rating = df['ai_rating'].astype(float).values
human_rating = df['human_rating'].astype(float).values

print(f"\nData summary:")
print(f"  Texts: {len(texts)} essays")
print(f"  Embeddings: {embeddings.shape}")
print(f"  Valid criterion (Y): {(~np.isnan(social_class)).sum()}")
print(f"  Valid AI ratings: {(~np.isnan(ai_rating)).sum()}")
print(f"  Valid human ratings: {(~np.isnan(human_rating)).sum()}")

# Create output directory
output_dir = 'haam_results'
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# RUN HAAM ANALYSIS
# ==============================================================================

print("\n2. RUNNING HAAM ANALYSIS...")
print("-"*60)

# Initialize and run HAAM with new default parameters
# Note: New defaults match my_colab.py exactly
haam = HAAM(
    criterion=social_class,
    ai_judgment=ai_rating,
    human_judgment=human_rating,
    embeddings=embeddings,
    texts=texts,  # Include texts for topic analysis
    n_components=200,
    # New defaults: min_cluster_size=10, min_samples=2
    # UMAP: n_neighbors=5, min_dist=0.0, metric='cosine'
    auto_run=True
)

results = haam.results
print("\n✓ Analysis complete with enhanced c-TF-IDF topic modeling!")

# ==============================================================================
# SAVE AND DISPLAY RESULTS
# ==============================================================================

print("\n3. SAVING RESULTS...")
print("-"*60)

# Save coefficients (now uses Y instead of SC)
if 'coefficients' in results:
    coef_df = results['coefficients']
    coef_path = os.path.join(output_dir, 'coefficients.csv')
    coef_df.to_csv(coef_path, index=False)
    print(f"✓ Saved coefficients: {coef_path}")
    
    # Display sample
    print("\nSample coefficients (first 5 PCs):")
    print(coef_df.head().to_string())

# Save model summary
if 'model_summary' in results:
    summary_df = results['model_summary']
    summary_path = os.path.join(output_dir, 'model_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved model summary: {summary_path}")
    
    print("\nModel Performance:")
    print(summary_df.to_string(index=False))

# Save comprehensive metrics summary (NEW in v1.1.0)
print("\nSaving comprehensive metrics...")
metrics_path = haam.create_metrics_summary(output_dir=output_dir)
print(f"✓ Saved metrics summary: {metrics_path}")

# Save topic summaries for ALL 200 PCs (NEW: top 30 / bottom 30)
if hasattr(haam, 'topic_summaries') and haam.topic_summaries:
    topic_path = os.path.join(output_dir, 'topic_summaries_all_pcs.txt')
    with open(topic_path, 'w') as f:
        f.write("COMPREHENSIVE TOPIC SUMMARIES FOR ALL 200 PRINCIPAL COMPONENTS\n")
        f.write("="*80 + "\n")
        f.write("Showing top 30 and bottom 30 topics for each PC\n")
        f.write("="*80 + "\n\n")
        
        # Write summaries for ALL PCs
        for pc_idx in sorted(haam.topic_summaries.keys()):
            topics = haam.topic_summaries[pc_idx]
            f.write(f"\nPC{pc_idx + 1}:\n")
            f.write("="*60 + "\n")
            
            # High topics
            if 'high_topics' in topics and topics['high_topics']:
                high_topics = [t for t in topics['high_topics'] if t != 'No significant high topics']
                if high_topics:
                    f.write(f"\nHIGH on PC{pc_idx + 1} (Top {len(high_topics)} topics):\n")
                    f.write("-"*40 + "\n")
                    for i, topic in enumerate(high_topics, 1):
                        f.write(f"{i:3d}. {topic}\n")
                else:
                    f.write("\nHIGH: No significant high topics\n")
            
            # Low topics
            if 'low_topics' in topics and topics['low_topics']:
                low_topics = [t for t in topics['low_topics'] if t != 'No significant low topics']
                if low_topics:
                    f.write(f"\nLOW on PC{pc_idx + 1} (Bottom {len(low_topics)} topics):\n")
                    f.write("-"*40 + "\n")
                    for i, topic in enumerate(low_topics, 1):
                        f.write(f"{i:3d}. {topic}\n")
                else:
                    f.write("\nLOW: No significant low topics\n")
            
            f.write("\n")
    
    print(f"\n✓ Saved comprehensive topic summaries: {topic_path}")

# ==============================================================================
# CREATE VISUALIZATIONS
# ==============================================================================

print("\n4. CREATING VISUALIZATIONS...")
print("-"*60)

# Create main visualization (default: shows "-" for PC names)
try:
    print("Creating main framework visualization...")
    main_viz = haam.create_main_visualization(output_dir=output_dir)
    print(f"✓ Saved: {main_viz}")
except Exception as e:
    print(f"⚠️  Main visualization error: {str(e)[:100]}...")

# Create mini coefficient grid
try:
    print("Creating mini coefficient grid...")
    mini_grid = haam.create_mini_grid(output_dir=output_dir)
    print(f"✓ Saved: {mini_grid}")
except Exception as e:
    print(f"⚠️  Mini grid error: {str(e)[:100]}...")

# Create UMAP visualization
try:
    print("Creating UMAP visualization...")
    umap_viz = haam.create_umap_visualization(output_dir=output_dir)
    print(f"✓ Saved: {umap_viz}")
except Exception as e:
    print(f"⚠️  UMAP error: {str(e)[:100]}...")

# Create PC effects visualization
try:
    print("Creating PC effects visualization...")
    pc_effects = haam.create_pc_effects_visualization(output_dir=output_dir, n_top=20)
    print(f"✓ Saved: {pc_effects}")
except Exception as e:
    print(f"⚠️  PC effects error: {str(e)[:100]}...")

# ==============================================================================
# OPTIONAL: RE-CREATE VISUALIZATION WITH MANUAL PC NAMES
# ==============================================================================

print("\n5. EXAMPLE: RE-CREATE VISUALIZATION WITH INTERPRETIVE NAMES...")
print("-"*60)

# After analyzing the results, you can add meaningful names
# This is just an example - replace with your own interpretations
pc_names_example = {
    # Map PC indices (0-based) to descriptive names
    # e.g., if PC5 is about lifestyle: 4: "Lifestyle & Work"
    # Leave this empty for now - fill after interpreting results
}

if pc_names_example:
    print("Re-creating main visualization with custom PC names...")
    main_viz_named = haam.create_main_visualization(
        output_dir=output_dir,
        pc_names=pc_names_example
    )
    # Save with different filename
    os.rename(main_viz_named, os.path.join(output_dir, 'haam_main_visualization_named.html'))
    print(f"✓ Saved named version: haam_main_visualization_named.html")
else:
    print("ℹ️  No custom PC names provided. Edit pc_names_example after interpreting results.")

# ==============================================================================
# CREATE CUSTOM VISUALIZATIONS
# ==============================================================================

print("\n6. CREATING CUSTOM VISUALIZATIONS...")
print("-"*60)

# 1. Coefficient Heatmap (Updated to use Y instead of SC)
if 'coefficients' in results:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Prepare data
    coef_matrix = np.zeros((3, 50))  # First 50 PCs
    for i, outcome in enumerate(['Y', 'AI', 'HU']):
        coef_col = f'{outcome}_coef'
        if coef_col in coef_df.columns:
            coef_matrix[i, :] = coef_df[coef_col].values[:50]
    
    # Heatmap
    im = ax1.imshow(coef_matrix, aspect='auto', cmap='RdBu_r',
                    vmin=-np.max(np.abs(coef_matrix)), vmax=np.max(np.abs(coef_matrix)))
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Criterion (Y)', 'AI Rating', 'Human Rating'])
    ax1.set_xlabel('Principal Component')
    ax1.set_title('Coefficient Heatmap (First 50 PCs)')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    # Top PCs bar chart
    if 'top_pcs' in results:
        top_10 = results['top_pcs'][:10]
        avg_abs_coef = []
        for pc in top_10:
            coefs = []
            for outcome in ['Y', 'AI', 'HU']:
                coef_col = f'{outcome}_coef'
                if coef_col in coef_df.columns:
                    coefs.append(abs(coef_df.iloc[pc][coef_col]))
            avg_abs_coef.append(np.mean(coefs))
        
        ax2.bar([f'PC{pc+1}' for pc in top_10], avg_abs_coef, color='#9467bd')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Average |Coefficient|')
        ax2.set_title('Top 10 PCs by Average Absolute Effect (Triple Method)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coefficient_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 2. Model Performance Dashboard
if 'model_summary' in results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    summary_df = results['model_summary']
    outcomes = ['Criterion (Y)', 'AI Rating', 'Human Rating']
    outcome_codes = summary_df['Outcome'].values  # Should be ['Y', 'AI', 'HU']
    r2_cv = summary_df['R2_cv'].values
    r2_in = summary_df['R2_insample'].values
    n_selected = summary_df['N_selected'].values
    
    # R² comparison
    ax = axes[0, 0]
    x = np.arange(len(outcomes))
    width = 0.35
    ax.bar(x - width/2, r2_cv, width, label='Cross-validated', color='#1f77b4')
    ax.bar(x + width/2, r2_in, width, label='In-sample', color='#ff7f0e')
    ax.set_ylabel('R²')
    ax.set_title('Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Feature selection
    ax = axes[0, 1]
    ax.bar(outcomes, n_selected, color=['#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('Number of Selected PCs')
    ax.set_title('Feature Selection (out of 200)')
    ax.axhline(y=200, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Overfitting assessment
    ax = axes[1, 0]
    overfitting = r2_in - r2_cv
    colors = ['red' if o > 0.1 else 'orange' if o > 0.05 else 'green' for o in overfitting]
    ax.bar(outcomes, overfitting, color=colors, alpha=0.7)
    ax.set_ylabel('R²(in) - R²(CV)')
    ax.set_title('Overfitting Assessment')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"SUMMARY\n{'='*20}\n\n"
    summary_text += f"Avg R²(CV): {r2_cv.mean():.3f}\n"
    summary_text += f"Avg PCs: {n_selected.mean():.0f}\n\n"
    
    if r2_cv.mean() > 0.3:
        summary_text += "✓ Strong predictive power\n"
    elif r2_cv.mean() > 0.1:
        summary_text += "✓ Moderate predictive power\n"
    else:
        summary_text += "⚠️  Limited predictive power\n"
    
    best_idx = np.argmax(r2_cv)
    best_model = outcomes[best_idx]
    summary_text += f"\nBest model: {best_model}"
    summary_text += f"\nR²(CV) = {r2_cv[best_idx]:.3f}"
    
    ax.text(0.1, 0.9, summary_text, ha='left', va='top', fontsize=11,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.suptitle('Model Performance Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"\nAll results saved to: {output_dir}/")
print("\nGenerated files:")
print("  📊 coefficients.csv - Full coefficient matrix")
print("  📈 model_summary.csv - Model performance metrics")
print("  📝 topic_summaries_all_pcs.txt - Top 30 & bottom 30 topics for ALL 200 PCs")
print("  📋 haam_metrics_summary.json - Comprehensive metrics (R², PoMA, etc.)")
print("  🎨 HTML visualizations (main, mini grid, UMAP, PC effects)")
print("  📊 coefficient_analysis.png - Coefficient heatmap and top PCs")
print("  📈 performance_dashboard.png - Model performance summary")

# Display key metrics
print("\n📋 KEY METRICS:")
print("-"*60)

# Load and display metrics summary
import json
with open(os.path.join(output_dir, 'haam_metrics_summary.json'), 'r') as f:
    metrics = json.load(f)

if 'model_performance' in metrics:
    print("\nModel Performance (R² cross-validated):")
    for outcome, perf in metrics['model_performance'].items():
        print(f"  {outcome}: {perf['r2_cv']:.3f}")

if 'mediation_analysis' in metrics and metrics['mediation_analysis']:
    print("\nProportion Mediated (PoMA):")
    for outcome, med in metrics['mediation_analysis'].items():
        print(f"  {outcome}: {med['proportion_mediated']:.1f}%")

if 'feature_selection' in metrics:
    print("\nFeatures Selected:")
    for outcome, feat in metrics['feature_selection'].items():
        print(f"  {outcome}: {feat['n_selected']} PCs")

print("\n✅ Analysis complete! Review topic_summaries_all_pcs.txt to interpret PCs.")
print("💡 Then use pc_names parameter to add meaningful labels to visualization.")