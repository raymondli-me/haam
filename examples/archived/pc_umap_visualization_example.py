"""
Example: PC UMAP Visualizations with Topic Labels
=================================================

This example demonstrates how to create UMAP visualizations colored by PC scores
with topic labels overlaid at cluster centroids.
"""

import pandas as pd
from haam import HAAM

# Example 1: Basic usage with sample data
def example_basic():
    """Basic example with synthetic data."""
    print("Example 1: Basic PC UMAP visualization")
    print("-" * 50)
    
    # Create sample data (replace with your actual data)
    n_samples = 1000
    data = {
        'criterion': np.random.randn(n_samples),
        'ai_judgment': np.random.randn(n_samples),
        'human_judgment': np.random.randn(n_samples),
        'text': [f"Sample text {i}" for i in range(n_samples)]  # Replace with actual texts
    }
    
    # Run HAAM analysis
    haam = HAAM(
        criterion=data['criterion'],
        ai_judgment=data['ai_judgment'],
        human_judgment=data['human_judgment'],
        texts=data['text'],
        n_components=200
    )
    
    # Visualize a specific PC (e.g., PC5 = index 4)
    output_file = haam.visualize_pc_umap_with_topics(
        pc_idx=4,  # PC5 (0-based indexing)
        output_dir='./visualizations',
        show_top_n=5,     # Show 5 high-scoring topics
        show_bottom_n=5,  # Show 5 low-scoring topics
        display=True      # Display in notebook/colab
    )
    print(f"Saved PC5 visualization to: {output_file}")


# Example 2: Real data from CSV
def example_from_csv():
    """Example using data from CSV file."""
    print("\nExample 2: PC UMAP from CSV data")
    print("-" * 50)
    
    # Load your data
    df = pd.read_csv('your_data.csv')  # Replace with your file
    
    # Run HAAM analysis
    haam = HAAM(
        criterion=df['target_variable'],
        ai_judgment=df['ai_rating'],
        human_judgment=df['human_rating'],
        texts=df['text_content'].tolist(),
        n_components=200
    )
    
    # Get top PCs based on triple ranking
    top_pcs = haam.analysis.get_top_pcs(n=5, ranking_method='triple')
    print(f"Top 5 PCs by triple ranking: {[pc+1 for pc in top_pcs]}")
    
    # Visualize each top PC
    for pc_idx in top_pcs:
        output_file = haam.visualize_pc_umap_with_topics(
            pc_idx=pc_idx,
            output_dir='./pc_visualizations',
            show_top_n=5,
            show_bottom_n=5,
            display=False  # Don't display each one in batch mode
        )
        print(f"Created visualization for PC{pc_idx + 1}")


# Example 3: Batch visualization of multiple PCs
def example_batch_visualization():
    """Example of creating visualizations for multiple PCs at once."""
    print("\nExample 3: Batch PC UMAP visualizations")
    print("-" * 50)
    
    # Assume we already have a HAAM instance
    # (In practice, load your data and run HAAM first)
    
    # Option 1: Visualize top 10 PCs (default)
    output_files = haam.create_all_pc_umap_visualizations(
        output_dir='./batch_pc_visualizations'
    )
    print(f"Created {len(output_files)} visualizations")
    
    # Option 2: Visualize specific PCs
    pc_indices = [0, 3, 4, 6, 7, 171]  # PC1, PC4, PC5, PC7, PC8, PC172
    output_files = haam.create_all_pc_umap_visualizations(
        pc_indices=pc_indices,
        output_dir='./selected_pc_visualizations',
        show_top_n=7,    # Show more topics
        show_bottom_n=7,
        display=False    # Don't display in batch mode
    )
    
    for pc_idx, filepath in output_files.items():
        print(f"PC{pc_idx + 1}: {filepath}")


# Example 4: Google Colab usage
def example_colab():
    """Example for Google Colab with drive mounting."""
    print("\nExample 4: Google Colab usage")
    print("-" * 50)
    
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Load data from Drive
    df = pd.read_csv('/content/drive/MyDrive/your_data.csv')
    
    # Run HAAM analysis
    haam = HAAM(
        criterion=df['target'],
        ai_judgment=df['ai_score'],
        human_judgment=df['human_score'],
        texts=df['text'].tolist()
    )
    
    # Create visualization for PC with highest AI coefficient
    top_ai_pcs = haam.analysis.get_top_pcs(n=1, ranking_method='AI')
    pc_idx = top_ai_pcs[0]
    
    output_file = haam.visualize_pc_umap_with_topics(
        pc_idx=pc_idx,
        output_dir='/content/drive/MyDrive/haam_output',
        show_top_n=6,
        show_bottom_n=6,
        display=True  # Will display in Colab
    )
    
    print(f"Visualization for PC{pc_idx + 1} (top AI PC) saved and displayed")
    
    # Create batch visualizations for all top PCs
    print("\nCreating visualizations for top 15 PCs...")
    top_15_pcs = haam.analysis.get_top_pcs(n=15, ranking_method='triple')
    
    output_files = haam.create_all_pc_umap_visualizations(
        pc_indices=top_15_pcs,
        output_dir='/content/drive/MyDrive/haam_output/pc_umaps',
        show_top_n=5,
        show_bottom_n=5,
        display=False  # Don't display all 15
    )
    
    print(f"Created {len(output_files)} PC UMAP visualizations")


# Example 5: Exploring specific PCs of interest
def example_explore_pcs():
    """Example of exploring PCs based on their topic associations."""
    print("\nExample 5: Exploring PCs by topic content")
    print("-" * 50)
    
    # After running HAAM, explore which PCs to visualize
    # by looking at their topic associations
    
    # First, look at PC topics to identify interesting ones
    pc_topics = haam.explore_pc_topics(
        pc_indices=list(range(20)),  # First 20 PCs
        n_topics=3  # Top 3 topics per PC
    )
    
    # Find PCs with specific keywords
    pcs_with_formal = []
    pcs_with_emotional = []
    
    for pc_idx, topics in pc_topics.items():
        high_topics = topics.get('high_topics', [])
        for topic in high_topics:
            if 'formal' in topic['keywords'].lower():
                pcs_with_formal.append(pc_idx)
            if 'emotion' in topic['keywords'].lower():
                pcs_with_emotional.append(pc_idx)
    
    print(f"PCs associated with formal language: {[pc+1 for pc in pcs_with_formal]}")
    print(f"PCs associated with emotional content: {[pc+1 for pc in pcs_with_emotional]}")
    
    # Visualize these specific PCs
    if pcs_with_formal:
        haam.visualize_pc_umap_with_topics(
            pc_idx=pcs_with_formal[0],
            output_dir='./thematic_visualizations',
            show_top_n=8,  # Show more topics for detailed view
            show_bottom_n=8
        )


if __name__ == "__main__":
    import numpy as np
    
    # Run examples (comment out those you don't need)
    try:
        example_basic()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # example_from_csv()  # Uncomment when you have real data
    # example_batch_visualization()  # Uncomment after running HAAM
    # example_colab()  # Use this in Google Colab
    # example_explore_pcs()  # Use after initial analysis