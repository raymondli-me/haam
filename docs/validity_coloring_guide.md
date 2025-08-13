# Word Cloud Validity Coloring Guide

## Overview

The validity coloring mode is an advanced feature for HAAM word clouds that helps distinguish between **valid** and **perceived** social class markers. This feature was developed to address a key research question: which linguistic markers truly indicate social class versus which are merely perceived to indicate social class?

## What is Validity Coloring?

Validity coloring uses different colors to show whether topics are:

1. **Valid markers**: Associated with high/low values across all three measures (Y/ground truth, Human ratings, AI ratings)
2. **Perceived markers**: Associated with high/low values in human and AI ratings but NOT in ground truth
3. **Mixed signals**: Inconsistent patterns across the three measures

## Color Scheme

### High-Associated Topics
- **Dark Red (#8B0000)**: Valid high markers - Topics in top quartile for ALL (Y+HU+AI)
- **Light Red (#FF6B6B)**: Perceived high markers - Topics in top quartile for HU+AI only

### Low-Associated Topics
- **Dark Blue (#00008B)**: Valid low markers - Topics in bottom quartile for ALL (Y+HU+AI)
- **Light Blue (#6B9AFF)**: Perceived low markers - Topics in bottom quartile for HU+AI only

### Mixed Signals
- **Dark Grey (#4A4A4A)**: Mixed strong signals - Conflicting strong associations
- **Light Grey (#B0B0B0)**: Mixed weak signals - No clear pattern

## How It Works

1. **Topic Percentiles**: For each PC, topics are ranked by their average PC values
2. **Quartile Calculation**: Topics in the top 25% (>75th percentile) are "high", bottom 25% (<25th percentile) are "low"
3. **PC Associations**: The method checks whether the PC is positively or negatively associated with each outcome (Y, HU, AI)
4. **Color Assignment**: Colors are assigned based on the consistency of associations across outcomes

## Usage

### Basic Usage

```python
# Generate word clouds with validity coloring
fig, high_path, low_path = haam.create_pc_wordclouds(
    pc_idx=0,
    k=10,
    max_words=100,
    color_mode='validity'  # Enable validity coloring
)
```

### Batch Generation

```python
# Generate multiple PCs with validity coloring
output_paths = haam.create_all_pc_wordclouds(
    pc_indices=[0, 1, 2, 3, 4],
    k=10,
    color_mode='validity'
)
```

### Grid Visualization

```python
# Create grid with validity coloring
grid_fig = haam.create_top_pcs_wordcloud_grid(
    pc_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    k=8,
    max_words=50,
    color_mode='validity'
)
```

## Interpreting Results

### Valid Markers (Dark Colors)
These topics are genuine indicators of social class:
- Consistent across objective measures and subjective ratings
- Reliable for understanding true social class differences
- Dark red = genuinely associated with high social class
- Dark blue = genuinely associated with low social class

### Perceived Markers (Light Colors)
These topics represent biases or stereotypes:
- Humans and AI associate them with social class
- But they don't actually correlate with true social class
- Light red = stereotypically "high class" but not actually
- Light blue = stereotypically "low class" but not actually

### Mixed Signals (Grey Colors)
These topics have inconsistent associations:
- May indicate complex relationships
- Could represent ambiguous markers
- Require further investigation

## Example Interpretation

Consider a word cloud for PC3:

- **"Private school" (Dark Red)**: Valid high-class marker - genuinely associated with higher social class
- **"Luxury brands" (Light Red)**: Perceived high-class marker - people think it indicates high class, but it doesn't
- **"Public housing" (Dark Blue)**: Valid low-class marker - genuinely associated with lower social class
- **"Fast food" (Light Blue)**: Perceived low-class marker - stereotypically "low class" but not actually predictive

## Technical Details

The validity coloring algorithm:

1. Retrieves all topics for the PC to establish quartile thresholds
2. Determines if each topic is in the high (>75th) or low (<25th) percentile
3. Checks the PC's relationship with each outcome (positive/negative coefficient)
4. Assigns colors based on consistency:
   - All three consistent = dark color (valid)
   - HU+AI consistent, Y different = light color (perceived)
   - Inconsistent = grey (mixed)

## When to Use Validity Coloring

Use validity coloring when you want to:
- Understand which linguistic markers are genuine vs stereotypical
- Study bias in human and AI judgments
- Identify topics that represent social perceptions vs reality
- Conduct research on stereotype accuracy

## Comparison with Pole Coloring

| Feature | Pole Coloring | Validity Coloring |
|---------|--------------|-------------------|
| Purpose | Show high vs low associations | Show valid vs perceived markers |
| Colors | Red/blue gradients | Multi-color scheme |
| Interpretation | Simple high/low | Complex validity analysis |
| Best for | General exploration | Bias/stereotype research |

## Requirements

- HAAM must be initialized with all three measures (criterion, AI judgment, human judgment)
- Analysis results must be available (debiased LASSO coefficients)
- Topic analysis must be performed (requires texts)

## See Also

- [Word Cloud Generation Guide](wordcloud_generation.md)
- [HAAM Analysis Guide](haam_analysis_guide.md)
- [Example Script](../examples/wordcloud_validity_example.py)