#!/usr/bin/env python3
"""
Run HAAM with Three Estimation Modes
=====================================

This script generates HAAM visualizations using three different estimation approaches:
1. Post-LASSO: LASSO selection + OLS on selected
2. LASSO: LASSO coefficients only
3. Multiple Regression: OLS on all PCs

Settings:
- k_topics = 5 (word cloud topics)
- n_components = 50 (PCs)
- sample_split_post_lasso = False (use all data)
- Timestamped output folders
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add the dev modifications to path
sys.path.insert(0, os.path.join(os.getcwd(), 'haam_dev_modifications'))

from haam_three_modes import fit_all_three_modes, create_visualization_for_mode

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = f"haam_outputs_{timestamp}"
os.makedirs(base_output_dir, exist_ok=True)

print("="*80)
print(f"HAAM THREE MODES ANALYSIS - {timestamp}")
print("="*80)
print("Configuration:")
print("  • n_components: 50")
print("  • k_topics: 5")
print("  • Modes: post-lasso, lasso, multiple-regression")
print("  • sample_split_post_lasso: False")
print("="*80)

# Load Study 1 data
print("\n1. Loading Study 1 data...")
df1 = pd.read_excel('Study 1/Data Study 1.xlsx', sheet_name='Data Study 1')
ratings_s1 = pd.read_csv('study1_gemini_ratings.csv')

df1_data = {
    'ID': df1['ID'].values,
    'text': df1['SourceB'].values,
    'self_power': df1['Power_mean'].values,
    'self_dominance': df1['Dom_mean'].values,
    'self_prestige': df1['Pres_mean'].values,
    'judge_power': df1['Power_F'].values,
    'judge_dominance': df1['Dom_F'].values,
    'judge_prestige': df1['Pres_F'].values,
}
df_study1 = pd.DataFrame(df1_data).dropna(subset=['text', 'self_power', 'judge_power'])
df_study1 = df_study1.merge(ratings_s1, on='ID', how='left')

# Load MiniLM embeddings
embeddings_s1 = np.load('study1_embeddings_minilm.npy')
print(f"✓ Loaded {len(df_study1)} participants with {embeddings_s1.shape[1]}-dim embeddings")

from haam import HAAM

constructs = [
    ('power', 'self_power', 'judge_power', 'AI_Power_mean'),
    ('dominance', 'self_dominance', 'judge_dominance', 'AI_Dom_mean'),
    ('prestige', 'self_prestige', 'judge_prestige', 'AI_Pres_mean'),
]

summary_data = {
    'post_lasso': [],
    'lasso': [],
    'multiple_regression': []
}

for construct_name, self_col, judge_col, ai_col in constructs:
    print(f"\n{'='*80}")
    print(f"CONSTRUCT: {construct_name.upper()}")
    print('='*80)

    output_dir = f"{base_output_dir}/study1_{construct_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize HAAM (no auto_run)
    print(f"\nInitializing HAAM for {construct_name}...")
    haam = HAAM(
        criterion=df_study1[self_col].values,
        human_judgment=df_study1[judge_col].values,
        ai_judgment=df_study1[ai_col].values,
        embeddings=embeddings_s1,
        texts=df_study1['text'].tolist(),
        n_components=50,
        standardize=True,
        sample_split_post_lasso=False,  # Will be handled by fit_all_three_modes
        min_cluster_size=3,
        min_samples=1,
        auto_run=False  # Don't auto-run, we'll use custom fitting
    )

    # Initialize analysis manually (since auto_run=False)
    print("Performing PCA...")
    from haam import HAAMAnalysis
    haam.analysis = HAAMAnalysis(
        criterion=haam.criterion,
        ai_judgment=haam.ai_judgment,
        human_judgment=haam.human_judgment,
        embeddings=haam.embeddings,
        texts=haam.texts,
        n_components=haam.n_components,
        standardize=haam.standardize
    )

    print("Performing topic analysis...")
    from haam import TopicAnalyzer
    haam.topic_analyzer = TopicAnalyzer(
        texts=haam.texts,
        embeddings=haam.analysis.embeddings,
        pca_features=haam.analysis.results['pca_features'],
        min_cluster_size=haam.min_cluster_size,
        min_samples=haam.min_samples,
        umap_n_components=haam.umap_n_components
    )

    # Get topic summaries
    all_pcs = list(range(haam.n_components))
    haam.topic_summaries = haam.topic_analyzer.create_topic_summary_for_pcs(
        all_pcs,
        n_keywords=10,
        n_topics_per_side=30
    )

    # Initialize visualizer
    from haam import HAAMVisualizer
    haam.visualizer = HAAMVisualizer(
        haam_results=haam.analysis.results,
        topic_summaries=haam.topic_summaries
    )

    # Fit all three estimation modes
    print(f"\nFitting all three estimation modes...")
    print("-"*60)
    results_all_modes = fit_all_three_modes(
        haam.analysis,
        use_sample_splitting=False
    )

    # Generate visualizations for all three modes
    print(f"\nGenerating HTML visualizations...")
    print("-"*60)

    for mode in ['post_lasso', 'lasso', 'multiple_regression']:
        try:
            mode_label = {
                'post_lasso': 'Post-LASSO',
                'lasso': 'LASSO Only',
                'multiple_regression': 'Multiple Regression'
            }[mode]

            print(f"  [{mode_label}]...")
            viz_path = create_visualization_for_mode(haam, mode, output_dir)

        except Exception as e:
            print(f"  ⚠️  Error creating {mode} visualization: {str(e)[:100]}")

    # Generate comprehensive analysis with k_topics=5
    print(f"\nGenerating comprehensive PC analysis...")
    try:
        # Restore results to post-lasso for comprehensive analysis
        haam.analysis.results['debiased_lasso'] = haam.analysis.results['post_lasso']
        haam.visualizer.results = haam.analysis.results

        results = haam.create_comprehensive_pc_analysis(
            k_topics=5,
            max_words=100,
            generate_wordclouds=True,
            generate_3d_umap=True,
            umap_arrow_k=1,
            output_dir=output_dir,
            display=False
        )
        print(f"  ✓ Comprehensive analysis complete")
    except Exception as e:
        print(f"  ⚠️  Error in comprehensive analysis: {str(e)[:150]}")

    # Collect summary statistics for all three modes
    for mode in ['post_lasso', 'lasso', 'multiple_regression']:
        mode_results = haam.analysis.results[mode]

        # Get PoMA values safely
        poma_hu = np.nan
        poma_ai = np.nan
        mediation_key = f'{mode}_mediation_analysis'
        if mediation_key in haam.analysis.results:
            med = haam.analysis.results[mediation_key]
            if 'HU' in med:
                med_hu = med['HU']
                if med_hu.get('total_effect', 0) != 0:
                    poma_hu = (med_hu['indirect_effect'] / med_hu['total_effect']) * 100
            if 'AI' in med:
                med_ai = med['AI']
                if med_ai.get('total_effect', 0) != 0:
                    poma_ai = (med_ai['indirect_effect'] / med_ai['total_effect']) * 100

        summary_data[mode].append({
            'Construct': construct_name.title(),
            'N': len(haam.criterion),
            'R2_X_CV': mode_results['X']['r2_cv'] if 'X' in mode_results else np.nan,
            'R2_HU_CV': mode_results['HU']['r2_cv'] if 'HU' in mode_results else np.nan,
            'R2_AI_CV': mode_results['AI']['r2_cv'] if 'AI' in mode_results else np.nan,
            'N_PCs_X': mode_results['X']['n_selected'] if 'X' in mode_results else 0,
            'N_PCs_HU': mode_results['HU']['n_selected'] if 'HU' in mode_results else 0,
            'N_PCs_AI': mode_results['AI']['n_selected'] if 'AI' in mode_results else 0,
            'PoMA_HU (%)': poma_hu,
            'PoMA_AI (%)': poma_ai,
        })

    print(f"\n✓ {construct_name.title()} complete!")

# Save summaries for all three modes
print("\n" + "="*80)
print("SAVING SUMMARY STATISTICS")
print("="*80)

for mode in ['post_lasso', 'lasso', 'multiple_regression']:
    summary_df = pd.DataFrame(summary_data[mode])
    summary_path = f"{base_output_dir}/summary_{mode}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{mode.upper().replace('_', ' ')}:")
    print(summary_df.to_string(index=False))
    print(f"Saved to: {summary_path}")

print("\n" + "="*80)
print("ALL ANALYSES COMPLETE!")
print("="*80)
print(f"\nOutput directory: {base_output_dir}/")

print("\n" + "="*80)
print("GENERATED FILES BY CONSTRUCT:")
print("="*80)
for construct_name, _, _, _ in constructs:
    output_dir = f"{base_output_dir}/study1_{construct_name}"
    print(f"\n{construct_name.upper()}:")
    print(f"  Directory: {output_dir}/")

    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        html_files = [f for f in files if f.endswith('.html')]

        # Group by mode
        post_lasso_files = [f for f in html_files if 'post_lasso' in f]
        lasso_files = [f for f in html_files if 'lasso' in f and 'post' not in f]
        mr_files = [f for f in html_files if 'multiple_regression' in f]
        other_files = [f for f in html_files if f not in post_lasso_files + lasso_files + mr_files]

        if post_lasso_files:
            print("  Post-LASSO visualizations:")
            for f in post_lasso_files:
                print(f"    - {f}")

        if lasso_files:
            print("  LASSO visualizations:")
            for f in lasso_files:
                print(f"    - {f}")

        if mr_files:
            print("  Multiple Regression visualizations:")
            for f in mr_files:
                print(f"    - {f}")

        if other_files:
            print("  Other visualizations:")
            for f in other_files:
                print(f"    - {f}")

        # Check for other file types
        csv_files = [f for f in files if f.endswith('.csv')]
        png_files = [f for f in files if f.endswith('.png')]

        if csv_files:
            print(f"  CSV files: {len(csv_files)} files")
        if png_files:
            print(f"  PNG files: {len(png_files)} files")
        if 'wordclouds' in files:
            print("  + wordclouds/ directory")
        if 'haam_comprehensive_analysis' in files:
            print("  + haam_comprehensive_analysis/ directory")

print(f"\n✓ All results saved to: {base_output_dir}/")
print("\n" + "="*80)
print("KEY FEATURES:")
print("="*80)
print("  • Three estimation modes compared:")
print("    1. Post-LASSO: LASSO selection + OLS on selected")
print("    2. LASSO: LASSO coefficients only")
print("    3. Multiple Regression: OLS on all 50 PCs")
print("  • k_topics = 5 for word clouds")
print("  • Actual n_components shown in visualizations")
print("  • Separate HTML files for each mode")
print("="*80)
