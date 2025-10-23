#!/usr/bin/env python3
"""
HAAM with Visualizations
========================
Full analysis with comprehensive visualizations: wordclouds, UMAP, topic clustering.
Same settings as the Brunswik Newsletter analysis.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from haam import HAAM
import os
from datetime import datetime

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("HAAM WITH VISUALIZATIONS")
print("="*80 + "\n")

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"haam_results_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# 1. Load data
print("Loading data...")
df = pd.read_csv('data.csv')
print(f"✓ Loaded {len(df)} participants\n")

# 2. Generate embeddings
print("Generating embeddings...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
print(f"✓ Generated embeddings: {embeddings.shape}\n")

# 3. Define constructs
constructs = [
    ('prestige', 'self_prestige', 'judge_prestige', 'AI_Pres_mean'),
    ('power', 'self_power', 'judge_power', 'AI_Power_mean'),
    ('dominance', 'self_dominance', 'judge_dominance', 'AI_Dom_mean'),
]

# 4. Run HAAM with comprehensive visualizations
for construct_name, self_col, judge_col, ai_col in constructs:
    print("="*80)
    print(f"ANALYZING: {construct_name.upper()}")
    print("="*80)

    # Extract variables
    X = df[self_col].values
    Y_HU = df[judge_col].values
    Y_AI = df[ai_col].values

    # Remove NaN values
    valid_mask = ~(np.isnan(X) | np.isnan(Y_HU) | np.isnan(Y_AI))
    X_clean = X[valid_mask]
    Y_HU_clean = Y_HU[valid_mask]
    Y_AI_clean = Y_AI[valid_mask]
    embeddings_clean = embeddings[valid_mask]
    texts_clean = [df['text'].tolist()[i] for i in range(len(df)) if valid_mask[i]]

    print(f"Sample size: {len(X_clean)}")

    # Run HAAM
    print("\nRunning HAAM analysis...")
    haam = HAAM(
        criterion=X_clean,
        human_judgment=Y_HU_clean,
        ai_judgment=Y_AI_clean,
        embeddings=embeddings_clean,
        texts=texts_clean,
        n_components=50,
        standardize=True,
        sample_split_post_lasso=False,
        min_cluster_size=3,
        min_samples=1,
        auto_run=True
    )

    # Generate comprehensive visualizations
    print("\nGenerating comprehensive PC analysis with visualizations...")
    comprehensive_output_dir = os.path.join(output_folder, f"{construct_name}_comprehensive")

    try:
        comprehensive_results = haam.create_comprehensive_pc_analysis(
            n_pcs=50,                # Analyze all 50 PCs
            k_topics=5,              # 5 topics at each end for word clouds
            max_words=100,           # 100 words per word cloud
            generate_wordclouds=True,
            generate_3d_umap=True,
            umap_arrow_k=1,          # PCA arrow topic cluster = 1
            output_dir=comprehensive_output_dir,
            display=False
        )
        print(f"✓ Visualizations saved to: {comprehensive_output_dir}/")
        print(f"  - Word cloud strips: wordclouds/")
        print(f"  - 3D UMAP: 3d_umap_pc1_2_3_arrows_validity.html")
        print(f"  - PC table: pc_table_comprehensive.png")
        print(f"  - Report: validity_analysis_report.txt")
    except Exception as e:
        print(f"⚠️  Warning: Visualization generation failed: {str(e)[:150]}")

    # Generate mediation path diagram
    print("\nGenerating mediation path diagram...")
    try:
        # Get top 9 PCs for the main visualization
        res = haam.analysis.results
        top_pcs = list(range(9))  # Top 9 PCs (0-8)

        path_diagram_file = os.path.join(output_folder, f"{construct_name}_mediation_diagram.html")
        haam.analysis.visualizations.create_main_visualization(
            pc_indices=top_pcs,
            output_file=path_diagram_file,
            ranking_method='HU'
        )
        print(f"✓ Mediation diagram saved to: {construct_name}_mediation_diagram.html")
    except Exception as e:
        print(f"⚠️  Warning: Mediation diagram generation failed: {str(e)[:150]}")

    print(f"\n✓ {construct_name.title()} complete\n")

print("\n" + "="*80)
print("✓ ALL ANALYSES COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {output_folder}/")
print("\nOutput structure:")
print(f"{output_folder}/")
print("├── prestige_mediation_diagram.html")
print("├── prestige_comprehensive/")
print("│   ├── 3d_umap_pc1_2_3_arrows_validity.html")
print("│   ├── pc_table_comprehensive.png")
print("│   ├── validity_analysis_report.txt")
print("│   └── wordclouds/ (100 PNG files for all 50 PCs)")
print("├── power_mediation_diagram.html")
print("├── power_comprehensive/")
print("│   └── (same structure)")
print("├── dominance_mediation_diagram.html")
print("└── dominance_comprehensive/")
print("    └── (same structure)")
print("\n" + "="*80 + "\n")
