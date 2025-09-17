# HAAM to Variable Resolution Integration Guide

## Overview

This guide explains how to use HAAM (Human-AI Alignment Model) to prepare data for visualization in the Variable Resolution engine. The integration allows you to:

1. Analyze text data with human and AI ratings using HAAM
2. Extract principal components as additional variables
3. Generate topic clusters and 3D positions
4. Export everything in the Variable Resolution Data Standard format

## Key Concepts

### What HAAM Provides

- **Core Variables**: Criterion (ground truth), Human judgments, AI predictions
- **Principal Components**: Extracted from text embeddings (PC1, PC2, ..., PCn)
- **Topic Clusters**: Discovered through HDBSCAN/BERTopic
- **3D Positions**: UMAP embeddings for spatial visualization
- **Statistical Metrics**: Correlations, PoMA, regression coefficients

### What Variable Resolution Needs

- **Structured JSON**: Following the Variable Resolution Data Standard v1.0
- **Variable Definitions**: Type, range, and metadata for each variable
- **Item-level Data**: ID, content, and all variable values for each item
- **Optional Enrichment**: Clusters, 3D positions, relationships

## Basic Usage

### Step 1: Run HAAM Analysis

```python
from haam import HAAM

# Your data
criterion = [...]  # Ground truth values
human_ratings = [...]  # Human ratings
ai_ratings = [...]  # AI predictions  
texts = [...]  # Text content

# Run HAAM
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_ratings,
    human_judgment=human_ratings,
    texts=texts,
    n_components=200,  # Number of PCs to extract
    auto_run=True
)
```

### Step 2: Export to Variable Resolution

```python
from haam import haam_to_variable_resolution

# Convert and save
vr_data = haam_to_variable_resolution(
    haam,
    output_file="my_analysis.json",
    title="My HAAM Analysis",
    description="Comparing human and AI ratings",
    include_pcs=15  # Include first 15 principal components
)
```

### Step 3: Upload to Variable Resolution

The exported JSON file can now be uploaded to the Variable Resolution visualization engine using the data upload feature.

## Advanced Usage

### Including All Principal Components

```python
vr_data = haam_to_variable_resolution(
    haam,
    include_all_pcs=True,  # Include all 200 PCs
    title="Full PC Analysis"
)
```

### Manual Export with Additional Variables

```python
from haam import HAAMToVariableResolution

# Additional variables you want to include
confidence_scores = [...]  # Per-item confidence
response_times = [...]  # Response time data

# Create converter
converter = HAAMToVariableResolution()

# Convert with extra variables
vr_data = converter.convert_from_data(
    criterion=criterion,
    human_judgment=human_ratings,
    ai_judgment=ai_ratings,
    texts=texts,
    pca_features=haam.analysis.results['pca_features'],
    additional_variables={
        "confidence": confidence_scores,
        "response_time": response_times,
        "any_other_metric": [...]
    },
    title="Enhanced Analysis",
    include_pcs=20
)

# Save
converter.save_to_file("enhanced_analysis.json")
```

### Custom IDs

```python
# Use custom IDs instead of numeric indices
ids = ["doc_001", "doc_002", "doc_003", ...]

vr_data = converter.convert_from_data(
    criterion=criterion,
    human_judgment=human_ratings,
    ai_judgment=ai_ratings,
    texts=texts,
    ids=ids,  # Custom identifiers
    ...
)
```

## Understanding the Output

### Variable Types in Export

1. **Core HAAM Variables**:
   - `criterion`: The ground truth or reference values
   - `human_judgment`: Human ratings or assessments
   - `ai_judgment`: AI system predictions

2. **Principal Components**:
   - `PC1`, `PC2`, ..., `PCn`: Extracted features from text embeddings
   - Each PC captures a different pattern in the text data
   - Ordered by variance explained (PC1 explains most variance)

3. **Additional Variables** (if provided):
   - Any custom metrics you include
   - Confidence scores, response times, etc.

### Data Structure

```json
{
  "version": "1.0",
  "metadata": {
    "title": "Your Analysis Title",
    "description": "...",
    "created": "2025-01-17T12:00:00Z",
    "datasetSize": 500
  },
  "schema": {
    "variables": {
      "criterion": {
        "type": "continuous",
        "displayName": "Criterion Score",
        "range": [-100, 100]
      },
      "PC1": {
        "type": "continuous",
        "displayName": "Principal Component 1",
        "range": [-3.2, 3.5],
        "metadata": {
          "variance_explained": 0.125
        }
      }
      // ... more variables
    }
  },
  "data": {
    "items": [
      {
        "id": 1,
        "content": "The actual text content...",
        "values": {
          "criterion": 75.2,
          "human_judgment": 72.1,
          "ai_judgment": 78.5,
          "PC1": 0.234,
          "PC2": -1.567
        },
        "cluster": {
          "id": 3,
          "label": "Topic 3"
        },
        "position": {
          "x": 1.23,
          "y": -0.45,
          "z": 2.11
        }
      }
      // ... more items
    ]
  }
}
```

## Visualization in Variable Resolution

Once uploaded, you can:

1. **View in 3D Space**: Items positioned by UMAP coordinates
2. **Color by Variables**: Use any variable (including PCs) for coloring
3. **Filter by Clusters**: Focus on specific topics
4. **Analyze Relationships**: See how variables correlate
5. **Interactive Exploration**: Click items to see details

## Best Practices

1. **Choose Meaningful PCs**: Start with 10-20 PCs unless you need more
2. **Validate Before Export**: Use `converter.validate()` to check data
3. **Include Descriptions**: Add clear titles and descriptions
4. **Consider Scale**: Large datasets (>10k items) may need sampling

## Troubleshooting

### Common Issues

1. **Missing Embeddings**: Ensure texts are provided to HAAM
2. **No Clusters**: Check min_cluster_size parameter
3. **Validation Errors**: Review error messages from validate()

### Performance Tips

- For large datasets, consider sampling in Variable Resolution
- Limit PCs to what you'll actually analyze
- Use batch processing for multiple analyses

## Future Enhancements

The integration will be expanded to support:
- Multiple rating systems (e.g., multiple human raters)
- Image data alongside text
- Dynamic variable generation
- Real-time updates

## Example Workflow

```python
# 1. Load your data
data = pd.read_csv("my_data.csv")

# 2. Run HAAM analysis
haam = HAAM(
    criterion=data['truth_rating'],
    ai_judgment=data['ai_rating'],
    human_judgment=data['human_rating'],
    texts=data['text_content'],
    n_components=100
)

# 3. Export to Variable Resolution
vr_data = haam_to_variable_resolution(
    haam,
    output_file="analysis_results.json",
    title=f"Analysis of {len(data)} items",
    include_pcs=15
)

# 4. Upload to Variable Resolution
# Use the data upload feature in the Variable Resolution app
```

## Support

For questions about:
- HAAM analysis: See HAAM documentation
- Variable Resolution: See Variable Resolution docs
- Integration issues: Open an issue on GitHub