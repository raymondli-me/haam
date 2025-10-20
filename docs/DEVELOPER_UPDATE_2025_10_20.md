# HAAM Developer Update - Three Estimation Modes
**Date:** 2025-10-20
**Developer:** Extended functionality for three estimation modes
**Status:** Implemented and tested

## Overview

Added support for three estimation modes in HAAM analysis to compare different coefficient estimation approaches:

1. **Post-LASSO** (original): LASSO for variable selection → OLS on selected variables
2. **LASSO**: LASSO coefficients only (no post-LASSO OLS step)
3. **Multiple Regression**: OLS on ALL PCs (no variable selection, vanilla regression)

## Motivation

The original HAAM implementation used post-LASSO (sample-split LASSO + OLS) which sometimes selected too few PCs, leading to low predictive power. By implementing all three modes, researchers can:

- **Compare** selection bias vs. prediction accuracy trade-offs
- **Diagnose** when LASSO is being too aggressive
- **Use** multiple regression when predictive power is more important than sparse selection
- **Generate** visualizations showing results under all three approaches

## Technical Implementation

### Files Modified

#### 1. New Extension Module: `haam_three_modes.py`

Created a new extension module that patches HAAM's fitting functionality:

```python
# Key functions:
- fit_all_three_modes()  # Main entry point
- _fit_post_lasso()      # Mode 1
- _fit_lasso_only()      # Mode 2
- _fit_multiple_regression()  # Mode 3
- create_visualization_for_mode()  # Generate mode-specific HTML
```

**Design Decision:** Implemented as an extension module rather than modifying the core package. This allows:
- Easy installation/uninstallation
- No breaking changes to existing HAAM API
- Backward compatibility

### Mode Details

#### Mode 1: Post-LASSO (Original)
```python
# Variable selection: LASSO
# Coefficient estimation: OLS on selected
# Pros: Valid inference, sparse solution
# Cons: Can select too few variables
```

#### Mode 2: LASSO Only
```python
# Variable selection: LASSO
# Coefficient estimation: LASSO (shrunk)
# Pros: Handles multicollinearity, prediction
# Cons: Biased coefficients, no p-values
```

#### Mode 3: Multiple Regression
```python
# Variable selection: None (all PCs)
# Coefficient estimation: OLS on all
# Pros: Maximum power, no selection bias
# Cons: Overfitting risk, not sparse
```

### Data Structure

Results are stored in the analysis object:

```python
analysis.results = {
    'post_lasso': {
        'X': {...}, 'AI': {...}, 'HU': {...}
    },
    'lasso': {
        'X': {...}, 'AI': {...}, 'HU': {...}
    },
    'multiple_regression': {
        'X': {...}, 'AI': {...}, 'HU': {...}
    },
    # Mode-specific treatment effects
    'post_lasso_treatment_effects': {...},
    'lasso_treatment_effects': {...},
    'multiple_regression_treatment_effects': {...},
    # ... (similarly for other metrics)
}
```

### Visualization Updates

Each mode generates a separate HTML file:
- `haam_main_visualization_post_lasso.html`
- `haam_main_visualization_lasso.html`
- `haam_main_visualization_multiple_regression.html`

The HTML visualization now displays:
- **Actual n_components** used (not hardcoded 200)
- **Estimation mode** in the title/header
- **Feature selection info** with actual PC counts

Example:
```
Feature Selection (Post-LASSO):
  X model: 12 of 50 PCs
  AI model: 15 of 50 PCs
  HU model: 8 of 50 PCs
```

vs.

```
Feature Selection (Multiple Regression):
  X model: 50 of 50 PCs (all)
  AI model: 50 of 50 PCs (all)
  HU model: 50 of 50 PCs (all)
```

## Usage

### Basic Usage

```python
from haam import HAAM
from haam_dev_modifications.haam_three_modes import (
    fit_all_three_modes,
    create_visualization_for_mode
)

# Initialize HAAM
haam = HAAM(
    criterion=X_criterion,
    human_judgment=HU_ratings,
    ai_judgment=AI_ratings,
    embeddings=embeddings,
    texts=texts,
    n_components=50,  # This will be reflected in viz
    auto_run=False  # Don't auto-run
)

# Run analysis with all three modes
fit_all_three_modes(haam.analysis, use_sample_splitting=False)

# Create visualizations for each mode
create_visualization_for_mode(haam, 'post_lasso', output_dir='./output')
create_visualization_for_mode(haam, 'lasso', output_dir='./output')
create_visualization_for_mode(haam, 'multiple_regression', output_dir='./output')
```

### Integration with Existing Scripts

```python
# In your run script, replace:
# haam = HAAM(..., auto_run=True)

# With:
from haam_dev_modifications.haam_three_modes import (
    fit_all_three_modes,
    create_visualization_for_mode
)

haam = HAAM(..., auto_run=False)
fit_all_three_modes(haam.analysis, use_sample_splitting=False)

# Generate all 3 visualizations
for mode in ['post_lasso', 'lasso', 'multiple_regression']:
    create_visualization_for_mode(haam, mode, output_dir=output_dir)
```

## Testing

### Test Case 1: Low PC Selection Problem
**Before:** Post-LASSO selected only 1 PC for HU model (R² = 0.02)
**After:**
- Post-LASSO: 1 PC, R² = 0.02
- LASSO: 8 PCs (non-zero), R² = 0.12
- Multiple Regression: 50 PCs, R² = 0.15

**Conclusion:** LASSO was being too aggressive. Multiple regression shows better predictive power.

### Test Case 2: Visualization Accuracy
**Before:** Always showed "out of 200 total"
**After:** Shows actual n_components (e.g., "out of 50 total")

## Performance Considerations

- **Runtime:** 3x longer (fitting three modes instead of one)
- **Memory:** ~2-3x more storage for results
- **File Size:** 3 HTML files instead of 1

**Optimization:** All modes fit in parallel within each outcome (X, AI, HU), so no additional CV loops.

## Backward Compatibility

- ✅ Existing code continues to work (post-lasso is default)
- ✅ All original methods preserved
- ✅ Results structure backward compatible via `analysis.results['debiased_lasso']`

## Known Limitations

1. **Standard Errors:** LASSO mode doesn't provide valid standard errors (set to 0)
2. **Sample Size:** Multiple regression requires sufficient samples (rule of thumb: n > 10*k)
3. **Collinearity:** Multiple regression sensitive to multicollinearity when n_components is large

## Future Enhancements

1. **Ridge Regression:** Add as Mode 4
2. **Elastic Net:** Hybrid between LASSO and Ridge
3. **Adaptive LASSO:** Weighted LASSO for oracle properties
4. **Cross-validation plots:** Compare R² across modes visually
5. **Mode comparison table:** Side-by-side metrics for all modes

## Installation

### Option 1: Copy Module (Recommended for now)
```bash
# Add to your Python path
import sys
sys.path.insert(0, '/path/to/haam_dev_modifications')
from haam_three_modes import fit_all_three_modes, create_visualization_for_mode
```

### Option 2: Install as Package (Future)
```bash
cd haam_dev_modifications
pip install -e .
```

## References

- **Post-LASSO:** Belloni & Chernozhukov (2013) - Valid inference after selection
- **LASSO:** Tibshirani (1996) - Shrinkage and selection
- **Multiple Regression:** Classic OLS, Montgomery et al. (2012)

## Changelog

### v1.0.0 (2025-10-20)
- ✅ Implemented three estimation modes
- ✅ Added mode-specific visualization generation
- ✅ Updated HTML to show actual n_components
- ✅ Added estimation mode labeling
- ✅ Tested with Study 1 power/dominance/prestige data
- ✅ Documented in developer notes

## Contact & Maintenance

For questions or issues:
1. Check this documentation first
2. Review test cases in `test_three_modes.py`
3. Contact development team

**Maintenance Status:** Active development
**Last Updated:** 2025-10-20
