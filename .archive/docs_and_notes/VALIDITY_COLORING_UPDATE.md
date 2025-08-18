# HAAM Validity Coloring System Update

## Overview
This document describes the updates made to the HAAM package's validity coloring system to ensure consistency between word clouds and UMAP visualizations.

## Background
The original system had a misalignment where:
- **Word cloud colors** used PC-based inference (indirect measurement via PC coefficients)
- **Word cloud labels** showed actual topic quartile positions (direct measurement)
- This caused confusion when labels showed "L L L" but colors were light blue instead of dark blue

## Changes Made

### 1. Word Cloud Coloring Update (Completed)

#### Files Modified
- `/haam/haam/haam_wordcloud.py`: Updated `_calculate_topic_validity_colors` method
- `/haam/haam/haam_init.py`: Modified to pass Y/HU/AI data to word cloud generator

#### Key Changes
1. **Direct Measurement**: Now uses actual Y/HU/AI values from documents within topics
2. **Topic-Level Quartiles**: Calculates quartiles from topic means rather than document values
3. **Sparse Data Handling**: Shows actual values even with limited data (e.g., 5% HU coverage)
4. **Color Assignment**:
   ```
   Dark Red (#8B0000):    Consensus high (all measures in top quartile)
   Light Red (#FF6B6B):   Any high (≥1 measure in top quartile)
   Dark Blue (#00008B):   Consensus low (all measures in bottom quartile)  
   Light Blue (#6B9AFF):  Any low (≥1 measure in bottom quartile)
   Dark Grey (#4A4A4A):   Opposing signals (mix of high and low)
   Light Grey (#B0B0B0):  All middle (all measures in middle quartiles)
   ```

### 2. UMAP Arrow Plot Update (Completed)

#### Files Modified
- `/haam/haam/haam_visualizations.py`: Updated `create_3d_umap_with_pc_arrows` method
- `/haam/haam/haam_init.py`: Modified wrapper to pass Y/HU/AI data and color_mode

#### Key Changes
1. **Added `color_mode` parameter**:
   - `'legacy'` (default): Keep current PC-based coloring for backward compatibility
   - `'validity'`: Use new direct measurement coloring (consistent with word clouds)

2. **Enhanced sparse data handling**:
   - Handles missing HU data gracefully (checks if model exists)
   - Uses NaN-safe comparisons for quartile calculations
   - Works with as little as 5% HU coverage

3. **Consistent implementation**:
   - Same color scheme as word clouds
   - Same topic-level quartile calculation
   - Same sparse data handling logic

## Technical Details

### Why Topic-Level Quartiles?
Document values have high variance, but topic means are compressed toward center due to averaging. Using document-level quartiles meant no topics could achieve "consensus high/low" status. Topic-level quartiles solve this by comparing topics to other topics.

### Handling Sparse Data
With only 5% HU coverage, the system now:
- Uses available data instead of showing "?"
- Calculates quartiles from valid (non-NaN) values only
- Shows data counts when sparse (e.g., "HU: H (n=3)")

## Migration Guide

### For Word Clouds
No changes needed - the update is backward compatible and automatically uses the new coloring system.

### For Arrow Plots
```python
# Legacy mode (current behavior)
haam.create_3d_umap_with_pc_arrows(
    color_mode='legacy',  # or omit for default
    ...
)

# New validity coloring (consistent with word clouds)
haam.create_3d_umap_with_pc_arrows(
    color_mode='validity',
    ...
)
```

## Benefits
1. **Consistency**: Word clouds and arrow plots can use the same coloring logic
2. **Accuracy**: Direct measurement is more accurate than PC-based inference
3. **Interpretability**: Colors directly reflect actual Y/HU/AI values
4. **Flexibility**: Legacy mode preserves existing functionality