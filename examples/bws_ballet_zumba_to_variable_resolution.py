#!/usr/bin/env python3
"""
Convert BWS Ballet/Zumba dataset to Variable Resolution Data Standard using HAAM
===============================================================================

This script converts the BWS ballet/zumba dataset to the Variable Resolution format,
using accessibility and artistic scores as the primary measures.
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
    print("BWS BALLET/ZUMBA DATASET TO VARIABLE RESOLUTION CONVERSION")
    print("="*80)
    
    # Load the BWS ballet/zumba dataset
    data_path = Path("/Users/raymondli701/workspace_2025_09_11/perceptionML/advanced_mode/data_github/bws_ballet_zumba_results_20250910_175036.csv")
    print(f"\n1. Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} rows")
        print(f"  Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Filter out rows with invalid genre values or missing scores
    valid_genres = ['Ballet', 'Zumba']
    df = df[df['genre'].isin(valid_genres)]
    df['accessibility'] = pd.to_numeric(df['accessibility'], errors='coerce')
    df['artistic'] = pd.to_numeric(df['artistic'], errors='coerce')
    df = df.dropna(subset=['accessibility', 'artistic'])
    
    # Only keep relevant sentences
    df = df[df['is_relevant'] == 'RELEVANT']
    
    print(f"\n✓ Filtered to {len(df)} valid relevant sentences")
    
    # Data exploration
    print("\n2. Data Overview:")
    print(f"  - Total sentences: {len(df)}")
    print(f"  - Genres: {df['genre'].value_counts().to_dict()}")
    print(f"  - Accessibility score range: [{df['accessibility'].min():.1f}, {df['accessibility'].max():.1f}]")
    print(f"  - Artistic score range: [{df['artistic'].min():.1f}, {df['artistic'].max():.1f}]")
    print(f"  - Confidence levels: {df['confidence'].value_counts().to_dict()}")
    print(f"  - Publications: {df['publication'].nunique()} unique sources")
    
    # Prepare data for HAAM
    texts = [clean_text(text) for text in df['sentence'].values]
    
    # For this dataset, we have two dimensions: accessibility and artistic
    # We'll use accessibility as the criterion and compare human vs AI judgments
    # In this case, we'll treat the scores as human judgments and create synthetic AI judgments
    
    criterion = df['accessibility'].values.astype(float)  # Use accessibility as criterion
    human_judgment = df['artistic'].values.astype(float)  # Use artistic as human judgment
    
    # Create synthetic AI judgment by combining accessibility and artistic with noise
    np.random.seed(42)
    ai_judgment = 0.6 * df['accessibility'].values + 0.4 * df['artistic'].values
    ai_judgment += np.random.normal(0, 5, len(ai_judgment))
    ai_judgment = np.clip(ai_judgment, 0, 100).astype(float)
    
    # Additional variables
    additional_vars = {
        'accessibility_score': df['accessibility'].values.astype(float),
        'artistic_score': df['artistic'].values.astype(float),
        'is_ballet': (df['genre'] == 'Ballet').astype(float),
        'is_zumba': (df['genre'] == 'Zumba').astype(float),
        'is_high_confidence': (df['confidence'] == 'HIGH').astype(float),
        'word_count': df['word_count'].fillna(0).astype(float),
    }
    
    # Calculate accessibility-artistic correlation
    acc_art_corr = np.corrcoef(df['accessibility'].values, df['artistic'].values)[0, 1]
    print(f"  - Accessibility-Artistic correlation: {acc_art_corr:.3f}")
    
    print("\n3. Running HAAM Analysis...")
    print("  - Criterion: Accessibility score")
    print("  - Human judgment: Artistic score")
    print("  - AI judgment: Synthetic combined score")
    
    # Run HAAM analysis
    try:
        haam = HAAM(
            criterion=criterion,
            ai_judgment=ai_judgment,
            human_judgment=human_judgment,
            texts=texts,
            n_components=50,  # Extract 50 principal components
            min_cluster_size=30,  # Reasonable cluster size
            min_samples=5,
            umap_n_components=3,
            standardize=True,
            sample_split_post_lasso=False,
            auto_run=False  # Don't auto-run to control the process
        )
        
        # Run the full analysis manually
        print("  Running full HAAM analysis...")
        results = haam.run_full_analysis()
        
        print(f"  Analysis complete. Found {len(results.get('top_pcs', []))} top PCs")
        
    except Exception as e:
        import traceback
        print(f"  Warning: Full HAAM analysis failed ({e}), continuing with basic analysis...")
        print("  Full traceback:")
        traceback.print_exc()
        # Create a minimal HAAM instance for conversion
        haam = HAAM(
            criterion=criterion,
            ai_judgment=ai_judgment,
            human_judgment=human_judgment,
            texts=texts,
            n_components=30,
            auto_run=False
        )
    
    print("\n4. Analysis Results:")
    # Get analysis results
    from scipy import stats
    hu_ai_corr = stats.pearsonr(human_judgment, ai_judgment)[0]
    x_hu_corr = stats.pearsonr(criterion, human_judgment)[0]
    x_ai_corr = stats.pearsonr(criterion, ai_judgment)[0]
    
    print(f"  - Artistic-AI correlation: {hu_ai_corr:.3f}")
    print(f"  - Accessibility-Artistic correlation: {x_hu_corr:.3f}")
    print(f"  - Accessibility-AI correlation: {x_ai_corr:.3f}")
    
    # Count clusters
    if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer:
        if hasattr(haam.topic_analyzer, 'cluster_labels'):
            n_clusters = len(set(haam.topic_analyzer.cluster_labels)) - 1  # Exclude outliers
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
            # cluster_labels is a numpy array, not a dict
            clusters = {
                "ids": haam.topic_analyzer.cluster_labels.tolist(),
                "labels": {}
            }
            # Generate topic labels using c-TF-IDF keywords
            for cluster_id in set(clusters["ids"]):
                if cluster_id != -1:
                    # Use actual c-TF-IDF keywords if available
                    if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer and hasattr(haam.topic_analyzer, 'topic_keywords') and cluster_id in haam.topic_analyzer.topic_keywords:
                        clusters["labels"][cluster_id] = haam.topic_analyzer.topic_keywords[cluster_id]
                    else:
                        clusters["labels"][cluster_id] = f"Topic {cluster_id}"
    
    positions = None
    if hasattr(haam, 'topic_analyzer') and haam.topic_analyzer:
        if hasattr(haam.topic_analyzer, 'umap_embeddings'):
            positions = haam.topic_analyzer.umap_embeddings
    
    # Create IDs from bws_id
    ids = df['bws_id'].values.tolist()
    
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
        title="Ballet vs Zumba: Accessibility and Artistic Analysis",
        description="Comparative analysis of ballet and zumba discourse in media, evaluating accessibility and artistic dimensions",
        author="HAAM Analysis",
        include_pcs=15  # Include top 15 PCs
    )
    
    # Add metadata
    vr_data["metadata"]["source"] = "BWS Ballet/Zumba Dataset"
    vr_data["metadata"]["tags"] = ["dance", "ballet", "zumba", "accessibility", "artistic", "cultural-analysis", "media"]
    vr_data["metadata"]["processingInfo"]["dataSource"] = str(data_path)
    vr_data["metadata"]["processingInfo"]["genres"] = list(df['genre'].unique())
    vr_data["metadata"]["processingInfo"]["dateRange"] = "2025-06 to 2025-07"
    
    # Update variable descriptions
    if "variables" in vr_data["schema"]:
        # Update the criterion/human/AI descriptions for clarity
        vr_data["schema"]["variables"]["criterion"]["displayName"] = "Accessibility"
        vr_data["schema"]["variables"]["criterion"]["description"] = "How accessible/approachable the dance content is"
        vr_data["schema"]["variables"]["human_judgment"]["displayName"] = "Artistic Merit"
        vr_data["schema"]["variables"]["human_judgment"]["description"] = "Artistic quality and sophistication"
        vr_data["schema"]["variables"]["ai_judgment"]["displayName"] = "Combined Score"
        vr_data["schema"]["variables"]["ai_judgment"]["description"] = "Synthetic combined accessibility-artistic score"
        
        # Additional variables
        vr_data["schema"]["variables"]["accessibility_score"]["description"] = "Original accessibility rating (0-100)"
        vr_data["schema"]["variables"]["artistic_score"]["description"] = "Original artistic merit rating (0-100)"
        vr_data["schema"]["variables"]["is_ballet"]["description"] = "Binary: sentence is about ballet"
        vr_data["schema"]["variables"]["is_zumba"]["description"] = "Binary: sentence is about zumba"
        vr_data["schema"]["variables"]["is_high_confidence"]["description"] = "Binary: annotator had high confidence"
        vr_data["schema"]["variables"]["word_count"]["description"] = "Word count of the article"
    
    # Add article titles to items for better context
    for i, item in enumerate(vr_data["data"]["items"]):
        row_idx = df.index[i]
        title = df.loc[row_idx, 'title'] if pd.notna(df.loc[row_idx, 'title']) else f"Article {i+1}"
        sentence_num = df.loc[row_idx, 'sentence_num'] if pd.notna(df.loc[row_idx, 'sentence_num']) else i+1
        
        # Clean up title and truncate if too long
        title = str(title).strip()
        if len(title) > 50:
            title = title[:47] + "..."
        
        # Create unique, informative title
        item["title"] = f"{title} (Sentence {int(sentence_num)})"
        
        # Add metadata with NaN handling
        item["metadata"] = {
            "genre": df.loc[row_idx, 'genre'],
            "publication": df.loc[row_idx, 'publication'] if pd.notna(df.loc[row_idx, 'publication']) else None,
            "date": df.loc[row_idx, 'date'] if pd.notna(df.loc[row_idx, 'date']) else None,
            "annotator": df.loc[row_idx, 'ra'],
            "confidence": df.loc[row_idx, 'confidence'],
            "article_num": int(df.loc[row_idx, 'article_num']) if pd.notna(df.loc[row_idx, 'article_num']) else None
        }
    
    # Save the output
    output_path = Path(__file__).parent / "bws_ballet_zumba_haam.json"
    print(f"\n6. Saving to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(vr_data, f, indent=2)
    
    print(f"✓ Successfully saved Variable Resolution data")
    print(f"  - Total items: {len(vr_data['data']['items'])}")
    print(f"  - Variables: {len(vr_data['schema']['variables'])}")
    print(f"  - Ballet sentences: {sum(1 for item in vr_data['data']['items'] if item.get('metadata', {}).get('genre') == 'Ballet')}")
    print(f"  - Zumba sentences: {sum(1 for item in vr_data['data']['items'] if item.get('metadata', {}).get('genre') == 'Zumba')}")
    
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