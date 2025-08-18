# UMAP PC Arrow Plot Coloring Update Plan

## Overview
Update the UMAP PC arrow plot coloring to be consistent with the word cloud validity coloring system while maintaining backward compatibility through a legacy mode.

## Current State Analysis

### Word Cloud Coloring (Updated)
- **Method**: Direct measurement of Y/HU/AI values within topics
- **Quartiles**: Based on topic means (not document values)
- **Implementation**: `_calculate_topic_validity_colors` in `haam_wordcloud.py`

### Arrow Plot Coloring (Current)
- **Method**: PC coefficient-based inference
- **Calculation**: `score = dot(topic_pc_scores, pc_coefs)`
- **Implementation**: Lines 1146-1202 in `haam_visualizations.py`

## Key Differences

### Conceptual
1. **Word clouds**: "What are the actual Y/HU/AI values in this topic?"
2. **Arrow plots**: "Based on PC loadings, what Y/HU/AI values would we expect?"

### Technical
1. **Data source**: Direct values vs PC-inferred scores
2. **Quartile basis**: Topic means vs weighted PC scores
3. **Sparse data**: Word clouds handle missing data gracefully

## Implementation Plan

### 1. Add `color_mode` Parameter
```python
def create_3d_umap_with_pc_arrows(..., color_mode='legacy', ...):
    """
    color_mode : str, default='legacy'
        - 'legacy': PC coefficient-based coloring (original)
        - 'validity': Direct Y/HU/AI measurement (like word clouds)
    """
```

### 2. Pass Y/HU/AI Data
Update `HAAM.create_3d_umap_with_pc_arrows` to pass:
```python
criterion=self.criterion if color_mode == 'validity' else None,
human_judgment=self.human_judgment if color_mode == 'validity' else None,
ai_judgment=self.ai_judgment if color_mode == 'validity' else None,
```

### 3. Implement Validity Coloring
When `color_mode='validity'`:
1. Calculate topic means for Y/HU/AI (handle NaN)
2. Establish topic-level quartiles (25th/75th percentiles)
3. Apply same color logic as word clouds

### 4. Color Scheme (Both Modes)
```
Dark Red (#8B0000):   Consensus high (all in top quartile)
Light Red (#FF6B6B):  Any high (≥1 in top quartile)
Dark Blue (#00008B):  Consensus low (all in bottom quartile)
Light Blue (#6B9AFF): Any low (≥1 in bottom quartile)
Dark Grey (#4A4A4A):  Opposing signals (high & low mixed)
Light Grey (#B0B0B0): All middle quartiles
```

### 5. Backward Compatibility
- Default to `color_mode='legacy'` to preserve existing behavior
- Existing code continues to work unchanged
- New functionality is opt-in

## Usage Examples

```python
# Legacy mode (default)
haam.create_3d_umap_with_pc_arrows()  # Uses PC-based coloring

# New validity mode (consistent with word clouds)
haam.create_3d_umap_with_pc_arrows(color_mode='validity')

# Specific PCs with validity coloring
haam.create_3d_umap_with_pc_arrows(
    pc_indices=[0, 3, 7],
    color_mode='validity',
    arrow_mode='list'
)
```

## Benefits

1. **Consistency**: Same coloring logic across visualizations
2. **Accuracy**: Direct measurement more reliable than inference
3. **Flexibility**: Users can choose which mode suits their needs
4. **Compatibility**: No breaking changes to existing code

## Files to Modify

1. `/haam/haam/haam_visualizations.py`:
   - Update `create_3d_umap_with_pc_arrows` method
   - Add validity coloring logic

2. `/haam/haam/haam_init.py`:
   - Update wrapper method to pass Y/HU/AI data
   - Add color_mode parameter

## Testing Strategy

1. Verify legacy mode produces identical output
2. Compare validity mode colors with word cloud colors
3. Test with sparse HU data (5% coverage)
4. Ensure all color categories can be achieved

## Migration Timeline

1. **Phase 1**: Implement with legacy as default
2. **Phase 2**: Update documentation and examples
3. **Phase 3**: Consider making validity the default in future version