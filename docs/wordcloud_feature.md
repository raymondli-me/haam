# Word Cloud Feature for HAAM

## Overview

The word cloud feature in HAAM generates visual representations of the topics associated with the high and low poles of each principal component (PC). This helps researchers quickly understand what each PC represents by visualizing the most relevant keywords.

## Features

- **Bipolar Visualization**: Each PC gets two word clouds - one for high pole topics (red) and one for low pole topics (blue)
- **Customizable Parameters**: Control the number of topics (k), maximum words, and figure sizes
- **Batch Generation**: Generate word clouds for multiple PCs at once
- **Grid Visualization**: Create compact grid views showing multiple PCs

## Installation

The word cloud feature requires the `wordcloud` package:

```bash
pip install wordcloud
```

This is automatically included when installing HAAM:

```bash
pip install git+https://github.com/raymondli-me/haam.git
```

## Usage

### Basic Usage - Single PC

```python
from haam import HAAM

# After running HAAM analysis
haam = HAAM(criterion, ai_judgment, human_judgment, texts=texts)

# Generate word clouds for PC1 (index 0)
fig, high_path, low_path = haam.create_pc_wordclouds(
    pc_idx=0,  # PC1
    k=10,  # Use top/bottom 10 topics
    max_words=100,  # Show up to 100 words
    output_dir='./wordclouds',
    display=True
)
```

### Batch Generation

```python
# Generate word clouds for specific PCs
pc_indices = [0, 3, 7, 12]  # PC1, PC4, PC8, PC13

output_paths = haam.create_all_pc_wordclouds(
    pc_indices=pc_indices,
    k=15,  # More topics for richer clouds
    max_words=150,
    output_dir='./wordclouds/batch',
    display=False
)

# Or auto-select top PCs
output_paths = haam.create_all_pc_wordclouds(
    pc_indices=None,  # Auto-selects top 9 by 'triple' ranking
    k=10,
    output_dir='./wordclouds/top_pcs'
)
```

### Grid Visualization

```python
# Create a grid showing multiple PCs at once
fig = haam.create_top_pcs_wordcloud_grid(
    pc_indices=None,  # Auto-select top PCs
    ranking_method='triple',  # or 'HU', 'AI', 'Y'
    n_pcs=9,  # Show 9 PCs in 3x3 grid
    k=10,
    max_words=50,
    output_file='./wordcloud_grid.png',
    display=True
)
```

## Parameters

### `create_pc_wordclouds()`

- **pc_idx** (int): PC index (0-based)
- **k** (int, default=10): Number of topics to include from each pole
- **max_words** (int, default=100): Maximum words to display in word cloud
- **figsize** (tuple, default=(10, 5)): Figure size (width, height) for each subplot
- **output_dir** (str, optional): Directory to save output files
- **display** (bool, default=True): Whether to display the plots

### `create_all_pc_wordclouds()`

- **pc_indices** (list, optional): List of PC indices. If None, uses top 9 PCs by 'triple' ranking
- **k** (int, default=10): Number of topics to include from each pole
- **max_words** (int, default=100): Maximum words to display
- **figsize** (tuple, default=(10, 5)): Figure size for each subplot
- **output_dir** (str, default='./wordclouds'): Directory to save output files
- **display** (bool, default=False): Whether to display each plot

### `create_top_pcs_wordcloud_grid()`

- **pc_indices** (list, optional): List of PC indices. If None, uses top PCs by ranking_method
- **ranking_method** (str, default='triple'): Method to rank PCs ('triple', 'HU', 'AI', 'Y')
- **n_pcs** (int, default=9): Number of top PCs to include if pc_indices not provided
- **k** (int, default=10): Number of topics to include from each pole
- **max_words** (int, default=50): Maximum words per word cloud
- **output_file** (str, optional): Path to save the grid visualization
- **display** (bool, default=True): Whether to display the plot

## Color Schemes

- **High Pole**: Red gradient (light red to dark red)
- **Low Pole**: Blue gradient (light blue to dark blue)

The color intensity represents word frequency within the aggregated topics.

## Output Files

The feature generates several types of output files:

1. **Combined Word Clouds**: `pc{N}_wordclouds.png` - Shows both high and low poles side by side
2. **Individual High Pole**: `pc{N}_high_wordcloud.png` - Just the high pole word cloud
3. **Individual Low Pole**: `pc{N}_low_wordcloud.png` - Just the low pole word cloud
4. **Grid Visualization**: `pc_wordcloud_grid.png` - Multiple PCs in one figure

## Technical Details

### Topic Aggregation

The word cloud generator aggregates keywords from multiple topics:
1. Retrieves k topics with highest/lowest average PC values
2. Weights keywords by topic rank (higher ranked topics contribute more)
3. Aggregates all keywords with frequency weighting
4. Generates word cloud from aggregated text

### Statistical Filtering

Only statistically significant topics (p < 0.05) are included in the word clouds.

## Examples

See `examples/wordcloud_example.py` for comprehensive examples including:
- Single PC visualization
- Batch processing
- Grid layouts
- Integration with full HAAM workflow
- Custom parameter configurations

## Troubleshooting

### No Word Clouds Generated

If you see "No significant high/low topics", this means:
- The PC has no topics with statistically significant associations (p > 0.05)
- Try increasing k to include more topics
- Check if topic analysis was performed (requires texts during initialization)

### Import Error

If you get `ImportError: wordcloud package not installed`:
```bash
pip install wordcloud
```

### Memory Issues

For large datasets or many PCs:
- Process PCs in smaller batches
- Reduce max_words parameter
- Set display=False to avoid showing plots