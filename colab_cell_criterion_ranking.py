# Create visualization with top 9 PCs ranked by Criterion (Y)
# This cell assumes you've already run HAAM analysis with variable name 'haam'

# Get top 9 PCs ranked by their effect on the criterion (Y/social class)
top_pcs_by_criterion = haam.analysis.get_top_pcs(n_top=9, ranking_method='SC')

print(f"Top 9 PCs by Criterion effect: PC{[pc+1 for pc in top_pcs_by_criterion]}")

# Create the visualization with these PCs
output_path = os.path.join(output_dir, 'haam_main_visualization_criterion.html')
haam.visualizer.create_main_visualization(
    pc_indices=top_pcs_by_criterion,
    output_file=output_path
)

print(f"\n✓ Created visualization with top 9 PCs by criterion: {output_path}")

# Optional: After reviewing results, add meaningful names for these specific PCs
# pc_names_criterion = {
#     # Example (replace with your actual PC indices and interpretations):
#     # 7: "Social Indicators",
#     # 14: "Economic Factors", 
#     # 2: "Educational Background",
#     # etc...
# }

# Then recreate with names:
# haam.visualizer.create_main_visualization(
#     pc_indices=top_pcs_by_criterion,
#     output_file=os.path.join(output_dir, 'haam_main_visualization_criterion_named.html'),
#     pc_names=pc_names_criterion
# )