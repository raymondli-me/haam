#!/usr/bin/env python3
"""
Example: Using HAAM's Comprehensive PC Analysis
==============================================
This example shows how to use the new create_comprehensive_pc_analysis() method
that generates word cloud tables and 3D UMAP visualizations.
"""

from haam import HAAM
import numpy as np

# Generate example data
np.random.seed(42)
n_samples = 500

# Create some example data
criterion = np.random.normal(3, 1, n_samples)  # X variable
ai_judgment = criterion + np.random.normal(0, 0.5, n_samples)  # AI predictions
human_judgment = criterion + np.random.normal(0, 0.7, n_samples)  # Human ratings

# Generate example texts
texts = [f"Example text {i} discussing various topics." for i in range(n_samples)]

print("="*80)
print("HAAM COMPREHENSIVE PC ANALYSIS EXAMPLE")
print("="*80)

# Initialize HAAM
print("\n1. Initializing HAAM...")
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=texts,
    n_components=50,  # Use 50 PCs for this example
    auto_run=True
)

# Example 1: Use all defaults (first 15 PCs)
print("\n2. Running comprehensive analysis with defaults...")
results = haam.create_comprehensive_pc_analysis()

print("\nResults generated:")
print(f"  - Word cloud paths: {len(results['wordcloud_paths'])} PC pairs")
print(f"  - Table path: {results['table_path']}")
print(f"  - 3D UMAP path: {results['umap_path']}")
print(f"  - Report path: {results['report_path']}")

# Example 2: Analyze specific PCs
print("\n3. Running analysis for specific PCs...")
specific_pcs = [0, 1, 2, 4, 7]  # PC1, PC2, PC3, PC5, PC8
results2 = haam.create_comprehensive_pc_analysis(
    pc_indices=specific_pcs,
    k_topics=5,  # Use 5 topics per pole
    max_words=150,  # More words per cloud
    output_dir='specific_pcs_analysis'
)

# Example 3: Generate only word clouds (skip 3D UMAP)
print("\n4. Generating only word clouds...")
results3 = haam.create_comprehensive_pc_analysis(
    n_pcs=10,  # First 10 PCs
    generate_3d_umap=False,  # Skip 3D UMAP
    output_dir='wordclouds_only'
)

# Example 4: Customize all parameters
print("\n5. Full customization example...")
results4 = haam.create_comprehensive_pc_analysis(
    pc_indices=[2, 1, 3, 4, 5, 14, 13, 11, 12],  # Custom PC selection
    k_topics=3,  # 3 topics per pole
    max_words=100,  # 100 words per cloud
    generate_wordclouds=True,
    generate_3d_umap=True,
    umap_arrow_k=1,  # Single topic endpoints for arrows
    show_data_counts=True,  # Show sparse data warnings
    output_dir='custom_analysis',
    display=False  # Don't display plots (useful for batch processing)
)

print("\n✓ Examples complete!")
print("\nKey parameters you can control:")
print("  - pc_indices: Which PCs to analyze (default: first n_pcs)")
print("  - n_pcs: Number of PCs if indices not specified (default: 15)")
print("  - k_topics: Topics per pole in word clouds (default: 3)")
print("  - max_words: Words per cloud (default: 100)")
print("  - generate_wordclouds: Whether to generate word clouds (default: True)")
print("  - generate_3d_umap: Whether to generate 3D UMAP (default: True)")
print("  - umap_arrow_k: Topics for arrow endpoints (default: 1)")
print("  - show_data_counts: Show sparse data counts (default: True)")
print("  - output_dir: Where to save outputs")
print("  - display: Whether to display plots")