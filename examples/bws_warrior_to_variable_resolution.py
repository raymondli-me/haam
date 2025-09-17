#!/usr/bin/env python3
"""
Convert BWS Warrior Culture dataset to Variable Resolution Data Standard using HAAM
==================================================================================

This script converts the BWS warrior culture dataset (NHL/hockey concussion data)
to the Variable Resolution format, using warrior score as the primary measure
and including additional context variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import HAAM
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from haam import HAAM
from haam.haam_to_variable_resolution import HAAMToVariableResolution

def clean_text(text):
    """Clean text by removing parsing artifacts and extra whitespace."""
    if pd.isna(text):
        return ""
    # Convert to string
    text = str(text)
    # Remove common parsing artifacts
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('  ', ' ')
    return text.strip()

def main():
    print("="*80)
    print("BWS WARRIOR CULTURE DATASET TO VARIABLE RESOLUTION CONVERSION")
    print("="*80)
    
    # Load the BWS warrior dataset
    data_path = Path("/Users/raymondli701/workspace_2025_09_11/perceptionML/advanced_mode/data_github/bws_warrior_FULL_20250828_190937.csv")
    print(f"\n1. Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} rows")
        print(f"  Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Filter out non-data rows (some rows contain metadata/noise)
    # Keep only rows with valid warrior scores
    df = df[df['warrior_score'].notna() & (df['warrior_score'] != 'warrior_score')]
    df['warrior_score'] = pd.to_numeric(df['warrior_score'], errors='coerce')
    df['warrior_percentile'] = pd.to_numeric(df['warrior_percentile'], errors='coerce')
    df = df.dropna(subset=['warrior_score'])
    
    print(f"\n✓ Filtered to {len(df)} valid data rows")
    
    # Data exploration
    print("\n2. Data Overview:")
    print(f"  - Total sentences: {len(df)}")
    print(f"  - Time periods: {df['time_bucket'].unique()}")
    print(f"  - Warrior score range: [{df['warrior_score'].min():.1f}, {df['warrior_score'].max():.1f}]")
    print(f"  - Warrior percentile range: [{df['warrior_percentile'].min():.1f}, {df['warrior_percentile'].max():.1f}]")
    print(f"  - Sentences with concussion: {df['has_concussion'].sum()} ({(df['has_concussion'].mean()*100):.1f}%)")
    print(f"  - Sentences with CTE: {df['has_cte'].sum()} ({(df['has_cte'].mean()*100):.1f}%)")
    print(f"  - Sentences with TBI: {df['has_tbi'].sum()} ({(df['has_tbi'].mean()*100):.1f}%)")
    
    # Prepare data for HAAM
    # For this dataset, we'll use warrior_score as both criterion and one of the judgments
    # We'll create synthetic human/AI splits for demonstration
    texts = [clean_text(text) for text in df['sentence'].values]
    
    # Use warrior_score as the criterion (ground truth)
    criterion = df['warrior_score'].values.astype(float)
    
    # For demonstration, we'll create human and AI judgments by adding noise to warrior score
    # In a real scenario, you'd have separate human and AI ratings
    np.random.seed(42)  # For reproducibility
    noise_human = np.random.normal(0, 5, len(criterion))
    noise_ai = np.random.normal(0, 3, len(criterion))
    
    human_judgment = np.clip(criterion + noise_human, 0, 100)
    ai_judgment = np.clip(criterion + noise_ai, 0, 100)
    
    # Extract year from time_bucket (take first year of range)
    def extract_year(time_bucket):
        if pd.isna(time_bucket):
            return 2015  # Default to middle year
        year_str = str(time_bucket).split('-')[0]
        try:
            return float(year_str)
        except:
            return 2015
    
    # Additional variables
    additional_vars = {
        'warrior_percentile': df['warrior_percentile'].values.astype(float),
        'word_count': df['word_count'].values.astype(float),
        'sentence_length': df['sentence_length'].values.astype(float),
        'has_concussion': df['has_concussion'].values.astype(float),
        'has_cte': df['has_cte'].values.astype(float),
        'has_tbi': df['has_tbi'].values.astype(float),
        'is_2020s': (df['time_bucket'] == '2020-2024').astype(float),  # Binary: is from 2020s
        'year': df['time_bucket'].apply(extract_year).values  # Extract year from time bucket
    }
    
    print("\n3. Running HAAM Analysis...")
    print("  - Criterion: Warrior score")
    print("  - Human judgment: Simulated human ratings")
    print("  - AI judgment: Simulated AI ratings")
    
    # Run HAAM analysis
    try:
        haam = HAAM(
            criterion=criterion,
            ai_judgment=ai_judgment,
            human_judgment=human_judgment,
            texts=texts,
            n_components=50,  # Reduced components to avoid numerical issues
            min_cluster_size=10,  # Smaller clusters for better success rate
            min_samples=2,
            umap_n_components=3,
            standardize=False,  # Avoid standardization issues
            sample_split_post_lasso=False,
            auto_run=True  # Auto-run for complete analysis including clustering
        )
        
    except Exception as e:
        print(f"  Warning: Full HAAM analysis failed ({e}), continuing with basic analysis...")
        # Create a minimal HAAM instance for conversion
        haam = HAAM(
            criterion=criterion,
            ai_judgment=ai_judgment,
            human_judgment=human_judgment,
            texts=texts,
            n_components=50,
            auto_run=True
        )
    
    print("\n4. Analysis Results:")
    # Get analysis results
    from scipy import stats
    hu_ai_corr = stats.pearsonr(human_judgment, ai_judgment)[0]
    x_hu_corr = stats.pearsonr(criterion, human_judgment)[0]
    x_ai_corr = stats.pearsonr(criterion, ai_judgment)[0]
    
    print(f"  - Human-AI correlation: {hu_ai_corr:.3f}")
    print(f"  - Human-Criterion correlation: {x_hu_corr:.3f}")
    print(f"  - AI-Criterion correlation: {x_ai_corr:.3f}")
    
    # Count clusters
    if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer:
        if hasattr(haam.topic_analyzer, 'cluster_labels'):
            n_clusters = len(set(haam.topic_analyzer.cluster_labels)) - 1  # Exclude outliers (-1)
            print(f"  - Number of topic clusters: {n_clusters}")
    
    print("\n5. Converting to Variable Resolution format...")
    
    # Create converter
    converter = HAAMToVariableResolution(haam)
    
    # Prepare additional data for conversion
    # Extract PCA features, clusters, and positions from HAAM
    pca_features = None
    if hasattr(haam.analysis, 'results') and 'pca_features' in haam.analysis.results:
        pca_features = haam.analysis.results['pca_features']
    
    clusters = None
    cluster_labels = {}
    if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer:
        if hasattr(haam.topic_analyzer, 'cluster_labels'):
            # Fix: cluster_labels is a numpy array, not a dictionary
            cluster_ids = haam.topic_analyzer.cluster_labels
            clusters = {
                "ids": cluster_ids.tolist(),
                "labels": {}
            }
            # Generate topic labels using c-TF-IDF keywords
            print(f"\n  Generating cluster labels for {len(set(cluster_ids))} clusters...")
            for cluster_id in set(cluster_ids):
                if cluster_id != -1:
                    # Use actual c-TF-IDF keywords if available
                    if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer and hasattr(haam.topic_analyzer, 'topic_keywords') and cluster_id in haam.topic_analyzer.topic_keywords:
                        # Use integer key, not string key!
                        clusters["labels"][cluster_id] = haam.topic_analyzer.topic_keywords[cluster_id]
                        print(f"    Cluster {cluster_id}: {haam.topic_analyzer.topic_keywords[cluster_id][:50]}...")
                    else:
                        clusters["labels"][cluster_id] = f"Topic {cluster_id}"
                        print(f"    Cluster {cluster_id}: No keywords found, using default label")
    
    positions = None
    if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer:
        if hasattr(haam.topic_analyzer, 'umap_embeddings'):
            positions = haam.topic_analyzer.umap_embeddings
    
    # Create unique IDs using article_index and sentence_index
    # Don't use article_id as it appears to be just the year
    ids = [f"warrior_{row['article_index']}_{row['sentence_index']}" for _, row in df.iterrows()]
    
    # Convert using the converter
    vr_data = converter.convert_from_data(
        criterion=criterion,
        human_judgment=human_judgment,
        ai_judgment=ai_judgment,
        texts=texts,
        ids=ids,
        pca_features=pca_features,
        clusters=clusters,
        positions=positions,
        additional_variables=additional_vars,
        title="NHL Warrior Culture Analysis: Concussion and Fighting Discourse",
        description="Analysis of warrior culture in NHL media coverage, focusing on concussion, CTE, and fighting-related discourse from 2010-2024",
        author="HAAM Analysis",
        include_pcs=20  # Include top 20 PCs
    )
    
    # Add metadata
    vr_data["metadata"]["source"] = "BWS Warrior Culture Dataset"
    vr_data["metadata"]["tags"] = ["warrior-culture", "NHL", "concussion", "CTE", "fighting", "sports", "media-analysis"]
    vr_data["metadata"]["processingInfo"]["dataSource"] = str(data_path)
    vr_data["metadata"]["processingInfo"]["timePeriods"] = list(df['time_bucket'].unique())
    
    # Update variable descriptions
    if "variables" in vr_data["schema"]:
        vr_data["schema"]["variables"]["warrior_percentile"]["description"] = "Percentile ranking of warrior culture score (0-100)"
        vr_data["schema"]["variables"]["word_count"]["description"] = "Number of words in the sentence"
        vr_data["schema"]["variables"]["sentence_length"]["description"] = "Character length of the sentence"
        vr_data["schema"]["variables"]["has_concussion"]["description"] = "Binary: sentence mentions concussion"
        vr_data["schema"]["variables"]["has_cte"]["description"] = "Binary: sentence mentions CTE"
        vr_data["schema"]["variables"]["has_tbi"]["description"] = "Binary: sentence mentions TBI"
        vr_data["schema"]["variables"]["is_2020s"]["description"] = "Binary: sentence from 2020-2024 period"
        vr_data["schema"]["variables"]["year"]["description"] = "Year of publication (2010-2024)"
    
    # Update titles to include article title and sentence index
    print("\n  Updating item titles with article information...")
    for i, item in enumerate(vr_data["data"]["items"]):
        row = df.iloc[i]
        article_title = str(row['article_title'])
        sentence_index = int(row['sentence_index'])
        
        # Clean up article title - remove special characters and truncate if too long
        article_title = article_title.strip()
        if len(article_title) > 50:
            article_title = article_title[:47] + "..."
        
        # Create unique, informative title
        item["title"] = f"{article_title} (Sentence {sentence_index})"
    
    # Save the output
    output_path = Path(__file__).parent / "bws_warrior_haam.json"
    print(f"\n6. Saving to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(vr_data, f, indent=2)
    
    print(f"✓ Successfully saved Variable Resolution data")
    print(f"  - Total items: {len(vr_data['data']['items'])}")
    print(f"  - Variables: {len(vr_data['schema']['variables'])}")
    
    # Validate
    converter = HAAMToVariableResolution()
    converter.data_standard = vr_data
    is_valid, errors = converter.validate()
    
    if is_valid:
        print("✓ Data validation passed!")
    else:
        print("✗ Validation errors:")
        for error in errors:
            print(f"  - {error}")

if __name__ == "__main__":
    main()