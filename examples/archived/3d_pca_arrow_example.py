"""
Example: 3D UMAP Visualization with PC Directional Arrows
========================================================

This example demonstrates how to create 3D UMAP visualizations with arrows
showing principal component directions in topic space, with color coding 
based on HU/AI usage patterns.
"""

import pandas as pd
import numpy as np
from haam import HAAM

# Example 1: Basic usage with all features
def example_full_features():
    """Create 3D UMAP plot with PC arrows and usage-based coloring."""
    print("Example 1: 3D UMAP with PC arrows - all features")
    print("-" * 50)
    
    # Load your data (replace with actual data)
    # For demonstration, using synthetic data
    n_samples = 1000
    data = {
        'criterion': np.random.randn(n_samples),
        'ai_judgment': np.random.randn(n_samples),
        'human_judgment': np.random.randn(n_samples),
        'text': [f"Sample text {i}" for i in range(n_samples)]
    }
    
    # Initialize and run HAAM
    haam = HAAM(
        criterion=data['criterion'],
        ai_judgment=data['ai_judgment'],
        human_judgment=data['human_judgment'],
        texts=data['text'],
        n_components=200
    )
    
    # Create 3D UMAP with arrows for first 3 PCs
    output_file = haam.create_3d_umap_with_pc_arrows(
        arrow_mode='all',  # Show arrows for PC1, PC2, PC3
        color_by_usage=True,  # Color by HU/AI usage patterns
        top_k=1,  # Single top/bottom topic for cleaner arrows (default)
        percentile_threshold=90.0,  # Use 90th percentile
        show_topic_labels=10,  # Show only 10 closest topics to camera (default)
        output_dir='./visualizations',
        display=True
    )
    print(f"Saved to: {output_file}")
    

# Example 2: Single PC arrow
def example_single_pc_arrow():
    """Show arrow for a specific PC of interest in UMAP space."""
    print("\nExample 2: 3D UMAP with single PC arrow")
    print("-" * 50)
    
    # Assume haam is already initialized
    # Create visualization showing only PC5's direction
    output_file = haam.create_3d_umap_with_pc_arrows(
        pc_indices=4,  # PC5 (0-based indexing)
        arrow_mode='single',
        top_k=3,  # Use top/bottom 3 topics
        color_by_usage=True,
        output_dir='./visualizations'
    )
    print(f"Saved PC5 arrow visualization to: {output_file}")


# Example 3: Multiple specific PC arrows
def example_selected_pc_arrows():
    """Show arrows for specific PCs based on their importance."""
    print("\nExample 3: 3D UMAP with selected PC arrows")
    print("-" * 50)
    
    # First, identify important PCs
    top_pcs = haam.analysis.get_top_pcs(n=9, ranking_method='triple')
    print(f"Top PCs by triple ranking: {[pc+1 for pc in top_pcs[:5]]}")
    
    # Create visualization with arrows for top 5 PCs
    output_file = haam.create_3d_umap_with_pc_arrows(
        pc_indices=top_pcs[:5],
        arrow_mode='list',
        top_k=5,
        percentile_threshold=85.0,  # Slightly looser threshold
        color_by_usage=True,
        output_dir='./visualizations'
    )
    print(f"Saved visualization with top 5 PC arrows to: {output_file}")


# Example 4: Without usage-based coloring
def example_size_based_coloring():
    """Create 3D UMAP with default topic size coloring."""
    print("\nExample 4: 3D UMAP with topic size coloring")
    print("-" * 50)
    
    output_file = haam.create_3d_umap_with_pc_arrows(
        arrow_mode='all',
        color_by_usage=False,  # Color by topic size instead
        output_dir='./visualizations'
    )
    print(f"Saved size-colored visualization to: {output_file}")


# Example 5: Exploring PC meanings before visualization
def example_explore_then_visualize():
    """First explore PC topics, then create targeted visualizations."""
    print("\nExample 5: Explore PCs then visualize")
    print("-" * 50)
    
    # Explore what each PC represents
    pc_topics = haam.explore_pc_topics(
        pc_indices=list(range(10)),  # First 10 PCs
        n_topics=5
    )
    
    # Find PCs with interesting patterns
    print("\nPC Topic Summary:")
    for i in range(10):
        pc_data = pc_topics[pc_topics['PC'] == i+1]
        if not pc_data.empty:
            high_topics = pc_data[pc_data['Direction'] == 'HIGH']['Keywords'].iloc[0] if len(pc_data[pc_data['Direction'] == 'HIGH']) > 0 else "None"
            low_topics = pc_data[pc_data['Direction'] == 'LOW']['Keywords'].iloc[0] if len(pc_data[pc_data['Direction'] == 'LOW']) > 0 else "None"
            print(f"PC{i+1}:")
            print(f"  HIGH: {high_topics[:50]}...")
            print(f"  LOW:  {low_topics[:50]}...")
    
    # Create visualization for interesting PCs
    # For example, if PC2, PC5, and PC8 show interesting patterns
    interesting_pcs = [1, 4, 7]  # 0-based indices
    
    output_file = haam.create_3d_umap_with_pc_arrows(
        pc_indices=interesting_pcs,
        arrow_mode='list',
        top_k=5,
        percentile_threshold=90.0,
        color_by_usage=True,
        output_dir='./visualizations'
    )
    print(f"\nCreated visualization for interesting PCs: {output_file}")


# Example 6: Different label display options
def example_label_display_options():
    """Demonstrate different topic label display modes."""
    print("\nExample 6: Topic label display options")
    print("-" * 50)
    
    # Version 1: Hide all labels for cleaner view
    print("\nCreating with no topic labels (hover still works)...")
    output_file = haam.create_3d_umap_with_pc_arrows(
        pc_indices=[0, 1, 2],
        arrow_mode='list',
        show_topic_labels=False,  # No labels, clean view
        output_dir='./visualizations'
    )
    print(f"Saved clean view: {output_file}")
    
    # Version 2: Show all labels
    print("\nCreating with all topic labels...")
    output_file = haam.create_3d_umap_with_pc_arrows(
        pc_indices=[0, 1, 2],
        arrow_mode='list',
        show_topic_labels=True,  # All labels visible
        output_dir='./visualizations'
    )
    print(f"Saved full labels view: {output_file}")
    
    # Version 3: Show only 5 closest topics
    print("\nCreating with only 5 closest topic labels...")
    output_file = haam.create_3d_umap_with_pc_arrows(
        pc_indices=[0, 1, 2],
        arrow_mode='list',
        show_topic_labels=5,  # Only 5 closest to camera
        output_dir='./visualizations'
    )
    print(f"Saved sparse labels view: {output_file}")
    

# Example 7: Batch creation for different configurations
def example_batch_creation():
    """Create multiple visualizations with different settings."""
    print("\nExample 6: Batch creation of visualizations")
    print("-" * 50)
    
    configurations = [
        # (pc_indices, arrow_mode, color_by_usage, description)
        (None, 'all', True, 'all_arrows_usage_colored'),
        ([0, 1, 2], 'list', False, 'first_3_pcs_size_colored'),
        (0, 'single', True, 'pc1_only_usage_colored'),
        ([4, 7, 11], 'list', True, 'selected_pcs_usage_colored'),
    ]
    
    for pc_indices, arrow_mode, color_by_usage, desc in configurations:
        output_file = haam.create_3d_umap_with_pc_arrows(
            pc_indices=pc_indices,
            arrow_mode=arrow_mode,
            color_by_usage=color_by_usage,
            top_k=1,  # Clean single-topic arrows
            percentile_threshold=90.0,
            show_topic_labels=10,  # Default label display
            output_dir=f'./visualizations/{desc}',
            display=False  # Don't display in batch mode
        )
        print(f"Created {desc}: {output_file}")


# Example 7: Integration with full HAAM workflow
def example_full_workflow():
    """Complete workflow from data to 3D visualization."""
    print("\nExample 7: Full HAAM workflow with 3D PCA")
    print("-" * 50)
    
    # Load real data
    df = pd.read_csv('your_data.csv')  # Replace with your file
    
    # Run HAAM analysis
    haam = HAAM(
        criterion=df['target_variable'],
        ai_judgment=df['ai_scores'],
        human_judgment=df['human_scores'],
        texts=df['text_content'].tolist(),
        n_components=200
    )
    
    # Export standard results
    haam.export_all_results('./haam_output')
    
    # Create 3D UMAP visualization as additional output
    output_file = haam.create_3d_umap_with_pc_arrows(
        arrow_mode='all',
        color_by_usage=True,
        top_k=5,
        output_dir='./haam_output/3d_visualizations'
    )
    
    print(f"Full analysis complete. 3D UMAP saved to: {output_file}")
    
    # Also create focused visualizations for top PCs
    top_pcs = haam.analysis.get_top_pcs(n=3, ranking_method='HU')
    focused_output = haam.create_3d_umap_with_pc_arrows(
        pc_indices=top_pcs,
        arrow_mode='list',
        top_k=7,  # Use more topics for stability
        percentile_threshold=85.0,
        color_by_usage=True,
        output_dir='./haam_output/3d_visualizations',
        display=True
    )
    
    print(f"Top HU PCs visualization: {focused_output}")


if __name__ == "__main__":
    # Note: You'll need to initialize a global 'haam' object or modify
    # these examples to work with your specific data setup
    
    try:
        # Run basic example (others assume haam is already initialized)
        example_full_features()
        
        # Uncomment to run other examples after initializing haam:
        # example_single_pc_arrow()
        # example_selected_pc_arrows()
        # example_size_based_coloring()
        # example_explore_then_visualize()
        # example_batch_creation()
        # example_full_workflow()
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure to provide real data or adjust the examples for your use case.")