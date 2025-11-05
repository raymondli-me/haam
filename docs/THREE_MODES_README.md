# HAAM Three Modes Extension

## Quick Start

This extension adds support for three estimation modes to HAAM:

1. **Post-LASSO** - LASSO selection + OLS on selected (original)
2. **LASSO** - LASSO coefficients only
3. **Multiple Regression** - OLS on all PCs (no selection)

### Installation

No installation needed! Just run the script:

```bash
python3 run_haam_three_modes.py
```

The script automatically adds the `haam_dev_modifications` directory to the Python path.

### What Gets Generated

For each construct (power, dominance, prestige), you'll get:

#### HTML Visualizations (3 per construct)
- `haam_main_visualization_post_lasso.html`
- `haam_main_visualization_lasso.html`
- `haam_main_visualization_multiple_regression.html`

#### Summary CSVs (3 total)
- `summary_post_lasso.csv` - Post-LASSO results
- `summary_lasso.csv` - LASSO results
- `summary_multiple_regression.csv` - Multiple regression results

#### Plus Standard HAAM Outputs
- Word clouds
- 3D UMAP visualizations
- PC analysis tables
- All other HAAM outputs

### Key Differences

| Feature | Post-LASSO | LASSO | Multiple Regression |
|---------|-----------|-------|-------------------|
| Variable Selection | Yes (LASSO) | Yes (LASSO) | No (all PCs) |
| Coefficients | OLS (unbiased) | LASSO (shrunk) | OLS (unbiased) |
| Typical # PCs | 1-15 | 5-25 | All (e.g., 50) |
| R² (CV) | Low-Medium | Medium | High |
| Inference | Valid p-values | No p-values | Valid p-values |
| Best For | Interpretation | Prediction | Maximum power |

### When to Use Each Mode

**Use Post-LASSO when:**
- You want sparse, interpretable results
- You need valid p-values
- Selection bias is acceptable

**Use LASSO when:**
- Post-LASSO selects too few PCs
- Prediction accuracy matters more than inference
- You have multicollinearity issues

**Use Multiple Regression when:**
- You want maximum predictive power
- You have enough samples (n >> k)
- You don't need sparsity

### Example Output

The visualizations now show:

```
Feature Selection (Post-LASSO):
  X model: 8 of 50 PCs
  AI model: 12 of 50 PCs
  HU model: 5 of 50 PCs
```

vs.

```
Feature Selection (Multiple Regression):
  X model: 50 of 50 PCs (all)
  AI model: 50 of 50 PCs (all)
  HU model: 50 of 50 PCs (all)
```

### Troubleshooting

**Problem:** Script fails with `ModuleNotFoundError`
**Solution:** Make sure you're running from the `Language-of-Power-main` directory

**Problem:** Visualization shows wrong mode
**Solution:** Check the filename - each mode has its own HTML file

**Problem:** Low R² in all modes
**Solution:** Check your data quality, embeddings, and n_components setting

### Advanced Usage

See `docs/DEVELOPER_UPDATE_2025_10_20.md` for:
- Detailed implementation notes
- Performance considerations
- Extension API documentation
- Future enhancements

### Files in This Directory

```
haam_dev_modifications/
├── README.md                              # This file
├── haam_three_modes.py                    # Main extension module
└── docs/
    └── DEVELOPER_UPDATE_2025_10_20.md     # Detailed documentation
```

### Questions?

1. Check `DEVELOPER_UPDATE_2025_10_20.md` for technical details
2. Review the example outputs in your generated folder
3. Compare summary CSVs to see mode differences

## That's It!

Just run `python3 run_haam_three_modes.py` and you'll get all three modes compared side-by-side.
