# Word Cloud Validity Coloring Guide

## Overview

The validity coloring mode is an advanced feature for HAAM word clouds that helps distinguish between **valid** and **perceived** markers of the criterion variable. This feature was developed to address a key research question: which perceptual cues truly indicate the criterion (Y) versus which are merely perceived to indicate it?

## What is Validity Coloring?

Validity coloring uses different colors to show the level of agreement across measures:

1. **Consensus markers (dark colors)**: All three measures agree (Y, Human, AI all high or all low)
2. **Any signal markers (light colors)**: At least one measure indicates high/low
3. **Opposing signals (dark grey)**: Disagreement - some measures say high, others say low
4. **Neutral markers (light grey)**: All measures in the middle range

## Color Scheme

### High-Associated Topics
- **Dark Red (#8B0000)**: Consensus high - ALL three measures (Y+Human+AI) in top quartile
- **Light Red (#FF6B6B)**: Any high signal - At least one measure in top quartile

### Low-Associated Topics
- **Dark Blue (#00008B)**: Consensus low - ALL three measures (Y+Human+AI) in bottom quartile
- **Light Blue (#6B9AFF)**: Any low signal - At least one measure in bottom quartile

### Mixed/Neutral Signals
- **Dark Grey (#4A4A4A)**: Opposing signals - Some measures high AND some low (disagreement)
- **Light Grey (#B0B0B0)**: All middle - All three measures in middle quartiles (25th-75th)

## How It Works

1. **Topic Percentiles**: For each PC, topics are ranked by their average PC values
2. **Quartile Calculation**: Topics in the top 25% (>75th percentile) are "high", bottom 25% (<25th percentile) are "low"
3. **PC Associations**: The method checks whether the PC is positively or negatively associated with each outcome (Y, Human, AI)
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

### Consensus Markers (Dark Colors)
**Dark Red/Blue**: All three measures agree - these are the most reliable indicators
- Dark red = consensus that this indicates high criterion values
- Dark blue = consensus that this indicates low criterion values
- High confidence markers for research

### Any Signal Markers (Light Colors)
**Light Red/Blue**: At least one measure indicates high/low
- Light red = some evidence of high criterion association
- Light blue = some evidence of low criterion association
- May include both valid markers and perceived associations
- Useful for exploring potential relationships

### Opposing Signals (Dark Grey)
**Dark Grey**: Measures disagree - some say high, others say low
- Indicates complex or contested markers
- May reveal disagreements between actual criterion and perceptions
- Warrants deeper investigation

### Neutral Markers (Light Grey)
**Light Grey**: All measures in the middle range
- Not strongly associated with high or low criterion values
- Common/universal features
- Less informative for distinguishing criterion levels

## Example Interpretation

Consider a word cloud for PC3 (using social class as an example criterion):

- **"Private school" (Dark Red)**: Valid high-criterion marker - genuinely associated with higher criterion values
- **"Luxury brands" (Light Red)**: Perceived high-criterion marker - people think it indicates high values, but it doesn't
- **"Public housing" (Dark Blue)**: Valid low-criterion marker - genuinely associated with lower criterion values
- **"Fast food" (Light Blue)**: Perceived low-criterion marker - stereotypically associated but not actually predictive

## Technical Details

The validity coloring algorithm:

1. Retrieves all topics for the PC to establish quartile thresholds
2. Determines if each topic is in the high (>75th) or low (<25th) percentile
3. Checks the PC's relationship with each outcome (positive/negative coefficient)
4. Assigns colors based on consistency:
   - All three consistent = dark color (valid)
   - Human+AI consistent, Y different = light color (perceived)
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