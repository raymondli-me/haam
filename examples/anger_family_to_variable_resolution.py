#!/usr/bin/env python3
"""
Convert anger_family.csv to Variable Resolution Data Standard using HAAM
=======================================================================

This script demonstrates converting the anger family Reddit dataset to the
Variable Resolution format, treating angry word count as the criterion
and including principal components as additional variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import HAAM
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from haam import HAAM, haam_to_variable_resolution

def main():
    print("="*80)
    print("ANGER FAMILY DATASET TO VARIABLE RESOLUTION CONVERSION")
    print("="*80)
    
    # Load the anger family dataset
    data_path = Path(__file__).parent.parent / "data" / "anger_family_with_angry_word_count.csv"
    print(f"\n1. Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} Reddit comments")
        print(f"  Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Data exploration
    print("\n2. Data Overview:")
    print(f"  - Total comments: {len(df)}")
    print(f"  - Human ratings range: [{df['human_sum_score'].min()}, {df['human_sum_score'].max()}]")
    print(f"  - GPT ratings range: [{df['gpt_sum_score'].min()}, {df['gpt_sum_score'].max()}]")
    print(f"  - Angry word count range: [{df['angry_word_count'].min()}, {df['angry_word_count'].max()}]")
    print(f"  - Comments with angry words: {(df['angry_word_count'] > 0).sum()} ({(df['angry_word_count'] > 0).mean()*100:.1f}%)")
    
    # Prepare data for HAAM
    # Using angry_word_count as the criterion (ground truth)
    # This makes sense as it's an objective measure of angry language
    texts = df['text'].values.tolist()
    text_ids = df['text_id'].values.tolist()
    
    # Convert to float arrays
    criterion = df['angry_word_count'].values.astype(float)  # Objective angry word count
    human_judgment = df['human_sum_score'].values.astype(float)  # Human anger ratings
    ai_judgment = df['gpt_sum_score'].values.astype(float)  # GPT anger ratings
    
    # Additional variable: number of raters
    num_raters = df['num_raters'].values.astype(float)
    
    print("\n3. Running HAAM Analysis...")
    print("  - Criterion: Angry word count (objective measure)")
    print("  - Human judgment: Sum of human anger ratings")
    print("  - AI judgment: GPT anger ratings")
    
    # Run HAAM analysis
    haam = HAAM(
        criterion=criterion,
        ai_judgment=ai_judgment,
        human_judgment=human_judgment,
        texts=texts,
        n_components=50,  # Extract 50 principal components
        min_cluster_size=20,  # Reasonable cluster size for this dataset
        min_samples=5,
        umap_n_components=3,
        standardize=True,  # Standardize for better comparison
        sample_split_post_lasso=False,  # Use full sample for maximum power
        auto_run=True
    )
    
    print("\n4. Analysis Results:")
    # Calculate correlations directly
    from scipy import stats
    hu_ai_corr = stats.pearsonr(human_judgment, ai_judgment)[0]
    x_hu_corr = stats.pearsonr(criterion, human_judgment)[0]
    x_ai_corr = stats.pearsonr(criterion, ai_judgment)[0]
    
    print(f"  - Human-AI correlation: {hu_ai_corr:.3f}")
    print(f"  - Human-Criterion correlation: {x_hu_corr:.3f}")
    print(f"  - AI-Criterion correlation: {x_ai_corr:.3f}")
    
    # Show DML results if available
    if hasattr(haam.analysis, 'results') and 'debiased_lasso' in haam.analysis.results:
        dml_results = haam.analysis.results['debiased_lasso']
        if 'X' in dml_results:
            print(f"  - Criterion model R² (CV): {dml_results['X'].get('r2_cv', 0):.3f}")
        if 'AI' in dml_results:
            print(f"  - AI model R² (CV): {dml_results['AI'].get('r2_cv', 0):.3f}")
        if 'HU' in dml_results:
            print(f"  - Human model R² (CV): {dml_results['HU'].get('r2_cv', 0):.3f}")
    
    # Show top PCs if available
    if 'top_pcs' in haam.results and isinstance(haam.results['top_pcs'], list):
        print("\n  Top Principal Components by importance:")
        for i, pc_info in enumerate(haam.results['top_pcs'][:5]):
            if isinstance(pc_info, dict) and 'pc' in pc_info:
                print(f"    PC{pc_info['pc']+1}: {pc_info.get('importance', 0):.3f}")
            else:
                print(f"    PC{i+1} (index {i})")
    
    print("\n5. Converting to Variable Resolution format...")
    
    # Method 1: Basic conversion with HAAM instance
    output_path = Path(__file__).parent / "output" / "anger_family_basic.json"
    output_path.parent.mkdir(exist_ok=True)
    
    vr_data_basic = haam_to_variable_resolution(
        haam,
        output_file=str(output_path),
        title="Reddit Anger Analysis: Human vs AI Ratings",
        description="Analysis of anger in Reddit comments comparing human and GPT ratings against objective angry word count",
        author="HAAM Anger Family Analysis",
        include_pcs=20  # Include first 20 principal components
    )
    
    print(f"\n✓ Basic export saved to: {output_path}")
    
    # Method 2: Advanced conversion with additional variables and custom IDs
    from haam import HAAMToVariableResolution
    
    converter = HAAMToVariableResolution()
    
    # Get topic/cluster information if available
    clusters = None
    positions = None
    
    if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer:
        if hasattr(haam.topic_analyzer, 'clusters'):
            # Create cluster dictionary
            cluster_ids = haam.topic_analyzer.clusters
            unique_clusters = np.unique(cluster_ids)
            
            # Get topic labels from topic summaries
            cluster_labels = {}
            if hasattr(haam, 'topic_summaries'):
                for cluster_id in unique_clusters:
                    if cluster_id in haam.topic_summaries:
                        # Use first few keywords as label
                        keywords = haam.topic_summaries[cluster_id].get('keywords', [])[:3]
                        cluster_labels[cluster_id] = ' '.join(keywords) if keywords else f"Topic {cluster_id}"
                    else:
                        cluster_labels[cluster_id] = f"Topic {cluster_id}" if cluster_id != -1 else "Outlier"
            
            clusters = {
                "ids": cluster_ids,
                "labels": cluster_labels
            }
        
        if hasattr(haam.topic_analyzer, 'umap_embeddings'):
            positions = haam.topic_analyzer.umap_embeddings
    
    # Calculate derived metrics
    # Agreement between human and AI
    agreement_score = np.where(
        (human_judgment > 0) == (ai_judgment > 0),  # Same sign
        1.0 - np.abs(human_judgment - ai_judgment) / (np.abs(human_judgment) + np.abs(ai_judgment) + 1e-6),
        0.0
    )
    
    # Confidence based on number of raters and agreement
    confidence_score = num_raters / num_raters.max() * 100 * (0.5 + 0.5 * agreement_score)
    
    vr_data_advanced = converter.convert_from_data(
        criterion=criterion,
        human_judgment=human_judgment,
        ai_judgment=ai_judgment,
        texts=texts,
        ids=text_ids,  # Use Reddit comment IDs
        pca_features=haam.analysis.results['pca_features'],
        clusters=clusters,
        positions=positions,
        additional_variables={
            "num_raters": num_raters,
            "human_ai_agreement": agreement_score,
            "confidence": confidence_score
        },
        title="Reddit Anger Analysis: Enhanced",
        description="Comprehensive analysis of anger in Reddit comments with human-AI alignment metrics",
        include_pcs=30  # Include more PCs for detailed analysis
    )
    
    # Validate the data
    is_valid, errors = converter.validate()
    if is_valid:
        output_advanced = Path(__file__).parent / "output" / "anger_family_enhanced.json"
        converter.save_to_file(output_advanced)
        print(f"✓ Enhanced export saved to: {output_advanced}")
    else:
        print("\n✗ Validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Print summary statistics
    print("\n6. Export Summary:")
    print(f"  Variables exported: {len(vr_data_advanced['schema']['variables'])}")
    print(f"  - Core HAAM variables: 3 (criterion, human_judgment, ai_judgment)")
    print(f"  - Principal components: {sum(1 for k in vr_data_advanced['schema']['variables'] if k.startswith('PC'))}")
    print(f"  - Additional variables: 3 (num_raters, human_ai_agreement, confidence)")
    
    if 'clusters' in vr_data_advanced['schema']:
        print(f"  Clustering: Yes ({len(set(clusters['ids']))} clusters)")
    if 'positioning' in vr_data_advanced['schema']:
        print(f"  3D positioning: Yes (UMAP embeddings)")
    
    # Sample of interesting comments
    print("\n7. Sample Comments by Pattern:")
    
    # High human, low AI
    mask = (human_judgment > 2) & (ai_judgment < 1)
    if mask.sum() > 0:
        idx = np.where(mask)[0][0]
        print(f"\n  Human detected anger, AI didn't:")
        print(f"    ID: {text_ids[idx]}")
        print(f"    Text: {texts[idx][:100]}...")
        print(f"    Human: {human_judgment[idx]}, AI: {ai_judgment[idx]}, Angry words: {criterion[idx]}")
    
    # Low human, high AI
    mask = (human_judgment < 1) & (ai_judgment > 2)
    if mask.sum() > 0:
        idx = np.where(mask)[0][0]
        print(f"\n  AI detected anger, humans didn't:")
        print(f"    ID: {text_ids[idx]}")
        print(f"    Text: {texts[idx][:100]}...")
        print(f"    Human: {human_judgment[idx]}, AI: {ai_judgment[idx]}, Angry words: {criterion[idx]}")
    
    # High agreement on anger
    mask = (human_judgment > 2) & (ai_judgment > 2) & (criterion > 0)
    if mask.sum() > 0:
        idx = np.where(mask)[0][0]
        print(f"\n  Both detected anger (with angry words):")
        print(f"    ID: {text_ids[idx]}")
        print(f"    Text: {texts[idx][:100]}...")
        print(f"    Human: {human_judgment[idx]}, AI: {ai_judgment[idx]}, Angry words: {criterion[idx]}")
    
    print("\n✓ Conversion complete!")
    print("\nThe exported JSON files can be uploaded to Variable Resolution for visualization.")
    print("Try coloring by different variables to see patterns in human-AI alignment on anger detection.")

if __name__ == "__main__":
    main()