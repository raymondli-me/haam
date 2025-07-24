#!/usr/bin/env python3
"""
HAAM Ranking Methods Example
============================
Shows how to select top PCs using different ranking methods.
"""

from haam import HAAM

# After running your HAAM analysis
haam = HAAM(
    criterion=your_criterion,
    ai_judgment=your_ai_ratings,
    human_judgment=your_human_ratings,
    embeddings=your_embeddings,
    texts=your_texts
)

# Method 1: Get top 9 by criterion (Y)
top_pcs_criterion = haam.analysis.get_top_pcs(n_top=9, ranking_method='SC')

# Method 2: Get top 9 by AI
top_pcs_ai = haam.analysis.get_top_pcs(n_top=9, ranking_method='AI')

# Method 3: Get top 9 by Human
top_pcs_human = haam.analysis.get_top_pcs(n_top=9, ranking_method='HU')

# Method 4: Default triple method (top 3 from each)
top_pcs_triple = haam.analysis.get_top_pcs(n_top=9, ranking_method='triple')

# Create visualizations with different rankings
# For Human-focused analysis:
haam.visualizer.create_main_visualization(
    pc_indices=top_pcs_human,
    output_file='./output/haam_viz_human.html'
)

# For Criterion-focused analysis:
haam.visualizer.create_main_visualization(
    pc_indices=top_pcs_criterion,
    output_file='./output/haam_viz_criterion.html'
)

# With custom names after interpretation
pc_names = {
    # Based on human ranking, these might be the top PCs
    172: "Writing Style",
    18: "Professional Topics",
    5: "Personal Narratives",
    # etc...
}

haam.visualizer.create_main_visualization(
    pc_indices=top_pcs_human,
    output_file='./output/haam_viz_human_named.html',
    pc_names=pc_names
)

# Print which PCs were selected by each method
print(f"Top 9 by Criterion: PC{[pc+1 for pc in top_pcs_criterion]}")
print(f"Top 9 by AI:        PC{[pc+1 for pc in top_pcs_ai]}")
print(f"Top 9 by Human:     PC{[pc+1 for pc in top_pcs_human]}")
print(f"Triple method:      PC{[pc+1 for pc in top_pcs_triple]}")