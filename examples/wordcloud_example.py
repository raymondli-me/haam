"""
Example: PC Word Cloud Visualizations
====================================

This example demonstrates how to create word cloud visualizations
for principal component poles (high vs low) in HAAM.
"""

import numpy as np
from haam import HAAM

# Example 1: Create word clouds for a single PC
def example_single_pc_wordcloud():
    """Create word clouds for high and low poles of a specific PC."""
    print("Example 1: Word clouds for a single PC")
    print("-" * 50)
    
    # Load your data (replace with actual data)
    n_samples = 1000
    data = {
        'criterion': np.random.randn(n_samples),
        'ai_judgment': np.random.randn(n_samples),
        'human_judgment': np.random.randn(n_samples),
        'text': [f"Sample text {i} with various topics and content." for i in range(n_samples)]
    }
    
    # Initialize and run HAAM
    haam = HAAM(
        criterion=data['criterion'],
        ai_judgment=data['ai_judgment'],
        human_judgment=data['human_judgment'],
        texts=data['text'],
        n_components=200
    )
    
    # Create word clouds for PC1 (index 0)
    fig, high_path, low_path = haam.create_pc_wordclouds(
        pc_idx=0,  # PC1
        k=10,  # Use top/bottom 10 topics
        max_words=100,  # Show up to 100 words
        figsize=(10, 5),  # Size of each word cloud
        output_dir='./wordcloud_outputs',
        display=True
    )
    
    print(f"High pole word cloud saved to: {high_path}")
    print(f"Low pole word cloud saved to: {low_path}")


# Example 2: Create word clouds for multiple specific PCs
def example_multiple_pcs_wordclouds():
    """Create word clouds for selected PCs of interest."""
    print("\nExample 2: Word clouds for multiple PCs")
    print("-" * 50)
    
    # Assume haam is already initialized
    # Select specific PCs of interest
    pc_indices = [0, 3, 7]  # PC1, PC4, PC8
    
    output_paths = haam.create_all_pc_wordclouds(
        pc_indices=pc_indices,
        k=15,  # Use more topics for richer word clouds
        max_words=150,
        figsize=(12, 6),
        output_dir='./wordcloud_outputs/selected_pcs',
        display=False  # Don't display each one individually
    )
    
    print(f"Created word clouds for {len(output_paths)} PCs")
    for pc_idx, (high_path, low_path) in output_paths.items():
        print(f"  PC{pc_idx + 1}: {high_path}, {low_path}")


# Example 3: Auto-generate word clouds for top PCs
def example_top_pcs_wordclouds():
    """Automatically create word clouds for the most important PCs."""
    print("\nExample 3: Word clouds for top PCs")
    print("-" * 50)
    
    # Create word clouds for top 9 PCs (auto-selected by 'triple' ranking)
    output_paths = haam.create_all_pc_wordclouds(
        pc_indices=None,  # Will auto-select top 9 by 'triple' ranking
        k=10,
        max_words=100,
        output_dir='./wordcloud_outputs/top_pcs',
        display=False
    )
    
    print(f"Created word clouds for top {len(output_paths)} PCs")


# Example 4: Create a grid visualization
def example_wordcloud_grid():
    """Create a grid showing all top PC word clouds in one figure."""
    print("\nExample 4: Word cloud grid for top PCs")
    print("-" * 50)
    
    # Create grid visualization
    fig = haam.create_top_pcs_wordcloud_grid(
        pc_indices=None,  # Auto-select top PCs
        ranking_method='triple',  # Use triple selection method
        n_pcs=9,  # Show top 9 PCs
        k=10,  # Topics per pole
        max_words=50,  # Fewer words for grid view
        output_file='./wordcloud_outputs/pc_wordcloud_grid.png',
        display=True
    )
    
    print("Grid visualization created and saved")


# Example 5: Customize parameters for different analyses
def example_custom_parameters():
    """Demonstrate different parameter settings for various use cases."""
    print("\nExample 5: Custom parameter examples")
    print("-" * 50)
    
    # Use case 1: Detailed analysis with many topics
    print("\nDetailed analysis with k=20 topics:")
    haam.create_pc_wordclouds(
        pc_idx=0,
        k=20,  # Include top/bottom 20 topics
        max_words=200,  # Show more words
        figsize=(14, 7),  # Larger figure
        output_dir='./wordcloud_outputs/detailed',
        display=False
    )
    
    # Use case 2: Quick overview with fewer topics
    print("\nQuick overview with k=5 topics:")
    haam.create_pc_wordclouds(
        pc_idx=0,
        k=5,  # Only top/bottom 5 topics
        max_words=50,  # Fewer words for clarity
        figsize=(8, 4),  # Smaller figure
        output_dir='./wordcloud_outputs/overview',
        display=False
    )
    
    # Use case 3: Focus on human-selected PCs
    print("\nHuman-judgment focused PCs:")
    human_top_pcs = haam.analysis.get_top_pcs(n=6, ranking_method='HU')
    output_paths = haam.create_all_pc_wordclouds(
        pc_indices=human_top_pcs,
        k=12,
        output_dir='./wordcloud_outputs/human_pcs',
        display=False
    )
    print(f"Created word clouds for {len(output_paths)} human-focused PCs")


# Example 6: Integration with full HAAM workflow
def example_full_workflow():
    """Complete workflow from data to word cloud visualizations."""
    print("\nExample 6: Full HAAM workflow with word clouds")
    print("-" * 50)
    
    # Load real data
    import pandas as pd
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
    
    # Add word cloud visualizations
    # 1. Grid overview
    haam.create_top_pcs_wordcloud_grid(
        output_file='./haam_output/visualizations/wordcloud_grid.png'
    )
    
    # 2. Individual word clouds for all top PCs
    output_paths = haam.create_all_pc_wordclouds(
        output_dir='./haam_output/wordclouds',
        k=15,
        max_words=150
    )
    
    print(f"Full analysis complete with {len(output_paths)} PC word clouds")
    
    # 3. Explore specific interesting PCs
    # First check what each PC represents
    pc_topics = haam.explore_pc_topics(pc_indices=list(range(10)))
    print("\nPC meanings from topic analysis:")
    for i in range(10):
        pc_data = pc_topics[pc_topics['PC'] == i+1]
        if not pc_data.empty:
            print(f"PC{i+1}: {pc_data.iloc[0]['Direction']} - {pc_data.iloc[0]['Keywords'][:50]}...")


if __name__ == "__main__":
    # Note: You'll need to initialize a global 'haam' object or modify
    # these examples to work with your specific data setup
    
    try:
        # Run basic example
        example_single_pc_wordcloud()
        
        # Uncomment to run other examples after initializing haam:
        # example_multiple_pcs_wordclouds()
        # example_top_pcs_wordclouds()
        # example_wordcloud_grid()
        # example_custom_parameters()
        # example_full_workflow()
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure to provide real data or adjust the examples for your use case.")