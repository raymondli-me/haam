# LaTeX Table Generation for HAAM

## Overview

The HAAM package now includes automatic generation of 5 publication-ready LaTeX tables that comprehensively report model results, effect sizes, and statistical tests.

## Features

✅ **5 Complete Tables** - All standard reporting tables for HAAM analyses
✅ **Automatic Extraction** - All metrics pulled from HAAM results
✅ **Publication-Ready** - Uses `booktabs`, `siunitx`, proper formatting
✅ **Single-Trait Focus** - Each table shows results for one trait
✅ **Trait-Agnostic** - Same code works for any trait name
✅ **Modular Design** - Generate all tables or individual ones
✅ **Clean Debugging** - Separate methods for each table type

## Quick Start

```python
from haam import HAAM

# Load your HAAM model
haam_model = HAAM.load("path/to/model.pkl")

# Generate all 5 tables at once
results = haam_model.create_all_latex_tables(
    trait_name="Power",
    output_dir="./latex_tables",
    display=True
)

# Or generate individual tables
table1 = haam_model.create_table_zero_order_correlations("Power")
table2 = haam_model.create_table_lasso_selection("Power")
table3 = haam_model.create_table_r2_and_poma("Power")
table4 = haam_model.create_table_dml_effects("Power")
table7 = haam_model.create_table_g_and_c("Power")
```

## The 5 Tables

### Table 1: Zero-Order Correlations

**Purpose**: Reports simple correlations between Validity, AI, and Human judgments

**Method**: `create_table_zero_order_correlations()`

**Data Extracted**:
- `total_effects['X_AI']['coefficient']` → Validity--AI correlation
- `total_effects['X_HU']['coefficient']` → Validity--Human correlation
- `total_effects['HU_AI']['coefficient']` → Human--AI correlation
- Significance stars based on p-values

**Example Output**:
```
Table 1. Zero-Order Correlations (r) Between Validity and Judgments for Power
──────────────────────────────────────────────────────────────
Construct | Validity--Human | Validity--AI | Human--AI
──────────────────────────────────────────────────────────────
Power     | .504***         | .507***      | .727***
──────────────────────────────────────────────────────────────
```

### Table 2: LASSO Feature Selection

**Purpose**: Shows how many PCs each model selected

**Method**: `create_table_lasso_selection()`

**Data Extracted**:
- `debiased_lasso['X']['n_selected']` → Number selected for Validity
- `debiased_lasso['AI']['n_selected']` → Number selected for AI
- `debiased_lasso['HU']['n_selected']` → Number selected for Human
- `pca_params['n_components']` → Total PCs available

**Example Output**:
```
Table 2. Number of Principal Components Selected by LASSO for Power
───────────────────────────────────────────────────────────────
            | Model Predicting
            | Validity | AI Judgment | Human Judgment
───────────────────────────────────────────────────────────────
Power       | 13 / 50  | 35 / 50     | 21 / 50
───────────────────────────────────────────────────────────────
```

### Table 3: R² and PoMA

**Purpose**: Reports model fit (R²) and proportion of mediated accuracy

**Method**: `create_table_r2_and_poma()`

**Data Extracted**:
- Cross-Validated R²:
  - `debiased_lasso['X']['r2_cv']`
  - `debiased_lasso['AI']['r2_cv']`
  - `debiased_lasso['HU']['r2_cv']`
- Training Set R²:
  - `debiased_lasso['X']['r2']`
  - `debiased_lasso['AI']['r2']`
  - `debiased_lasso['HU']['r2']`
- PoMA:
  - Calculated from `mediation_analysis` for each path
  - Shows min--max range

**Example Output**:
```
Table 3. R² and PoMA for Power
───────────────────────────────────────────────────────────────
Metric          | Validity | AI Perception | Human Perception | PoMA Range (%)
───────────────────────────────────────────────────────────────
Cross-Validated | 0.143    | 0.248         | 0.243            | 29.6 -- 39.2
Training Set    | 0.187    | 0.312         | 0.289            |
───────────────────────────────────────────────────────────────
```

**Key Feature**: Shows **both** CV and non-CV R² for transparency about overfitting.

### Table 4: DML Effects

**Purpose**: Reports total effects (β), DML direct effects (β̌), with full inference

**Method**: `create_table_dml_effects()`

**Data Extracted**:
- Total effects:
  - `total_effects[path]['coefficient']` → β
  - `total_effects[path]['se']` → SE
  - `total_effects[path]['p_value']` → p
- DML Direct effects:
  - `total_effects[path]['check_beta']` → β̌
  - Calculate t-statistic, p-value, 95% CI
- Indirect effect = β - β̌

**Example Output**:
```
Table 4. Total (β), DML Direct (β̌), and Indirect Effects for Power
──────────────────────────────────────────────────────────────────────────
Path          | Effect Type      | Estimate | SE    | t     | p      | 95% CI
──────────────────────────────────────────────────────────────────────────
Validity → AI | Total (β)        | .507     | .062  | 8.17  | <0.001 | [.385, .629]
              | DML Direct (β̌)  | .224     | .069  | 3.25  | .001   | [.088, .360]
──────────────────────────────────────────────────────────────────────────
Validity → HU | Total (β)        | .504     | .062  | 8.13  | <0.001 | [.382, .626]
              | DML Direct (β̌)  | .161     | .066  | 2.44  | .015   | [.031, .291]
──────────────────────────────────────────────────────────────────────────
Human → AI    | Total (β)        | .727     | .044  | 16.52 | <0.001 | [.641, .813]
              | DML Direct (β̌)  | .294     | .056  | 5.25  | <0.001 | [.184, .404]
──────────────────────────────────────────────────────────────────────────
```

**Key Features**:
- Full regression output (estimate, SE, t, p, CI)
- Both total and direct effects
- Indirect effect can be calculated as β - β̌

### Table 7: G and C Parameters

**Purpose**: Reports policy similarity (G) and residual correlation (C)

**Method**: `create_table_g_and_c()`

**Data Extracted**:
- Policy Similarity (G):
  - `policy_similarities['X_AI']` → G(Validity--AI)
  - `policy_similarities['X_HU']` → G(Validity--Human)
  - `policy_similarities['AI_HU']` → G(Human--AI)
- Residual Correlation (C):
  - `residual_correlations['X_AI']` → C(Validity--AI)
  - `residual_correlations['X_HU']` → C(Validity--Human)
  - `residual_correlations['AI_HU']` → C(Human--AI)

**Example Output**:
```
Table 7. Policy Similarity (G) and Residual Correlation (C) for Power
───────────────────────────────────────────────────────────────
Construct | Path            | G     | C
───────────────────────────────────────────────────────────────
Power     | Validity--AI    | .490  | .412
          | Validity--Human | .639  | .378
          | Human--AI       | .682  | .615
───────────────────────────────────────────────────────────────
```

**Interpretation**:
- **G**: Correlation between predicted scores (how similar are the policies?)
- **C**: Correlation between residuals (what's left unexplained after PCs?)

## Usage Patterns

### Pattern 1: Generate All Tables

```python
haam_model = HAAM.load("model.pkl")

results = haam_model.create_all_latex_tables(
    trait_name="Extraversion",
    output_dir="./tables",
    display=True
)

# Access individual file paths
print(results['table1']['tex_path'])
print(results['table2']['tex_path'])
# ... etc
```

### Pattern 2: Generate Specific Tables

```python
# Only generate the tables you need
table1 = haam_model.create_table_zero_order_correlations("Power")
table3 = haam_model.create_table_r2_and_poma("Power")
table7 = haam_model.create_table_g_and_c("Power")
```

### Pattern 3: Batch Process Multiple Traits

```python
import glob

for model_path in glob.glob("models/*/haam_model.pkl"):
    trait_name = extract_trait_name(model_path)
    haam_model = HAAM.load(model_path)

    results = haam_model.create_all_latex_tables(
        trait_name=trait_name,
        output_dir=f"./tables/{trait_name}"
    )
```

### Pattern 4: Organize by Table Type

```python
# Save each table type to its own directory
haam_model.create_table_zero_order_correlations("Power", "./tables/zero_order")
haam_model.create_table_lasso_selection("Power", "./tables/lasso")
haam_model.create_table_r2_and_poma("Power", "./tables/r2")
haam_model.create_table_dml_effects("Power", "./tables/dml")
haam_model.create_table_g_and_c("Power", "./tables/g_c")
```

## File Naming Convention

Generated files follow this pattern:
```
table1_zero_order_correlations_<trait_name>.tex
table2_lasso_selection_<trait_name>.tex
table3_r2_and_poma_<trait_name>.tex
table4_dml_effects_<trait_name>.tex
table7_g_and_c_<trait_name>.tex
```

Example for trait "Power":
```
table1_zero_order_correlations_power.tex
table2_lasso_selection_power.tex
table3_r2_and_poma_power.tex
table4_dml_effects_power.tex
table7_g_and_c_power.tex
```

## LaTeX Dependencies

Each table is a standalone document that compiles with:

```latex
\documentclass[11pt]{article}
\usepackage{booktabs}      % Professional tables
\usepackage{caption}       % Caption formatting
\usepackage{siunitx}       % Number formatting (Tables 3, 4, 7)
\usepackage[margin=1in]{geometry}
```

## Integrating into Your Paper

Each table is a standalone document, but you can extract just the table:

**Option 1: Copy the table environment**
```latex
% In your paper
\begin{table}[htbp]
... (copy from generated file)
\end{table}
```

**Option 2: Use \input**
```latex
% In your paper
\input{tables/table1_zero_order_correlations_power.tex}
```

**Option 3: Compile individually**
```bash
cd tables
pdflatex table1_zero_order_correlations_power.tex
# Then include the PDF
```

## Customization

### Modify Table Captions

Edit the generated .tex file:
```latex
% Change this:
\caption{Table 1. \textit{Zero-Order Correlations...}}

% To this:
\caption{Table 1. \textit{Bivariate Correlations...}}
```

### Adjust Number Formatting

Modify the `siunitx` column specifications:
```latex
% 3 decimals (default)
S[table-format=1.3]

% 2 decimals
S[table-format=1.2]
```

### Add/Remove Rows

The tables use simple `tabular` environments, easy to edit:
```latex
Power     & .504*** & .507*** & .727*** \\
% Add another row:
Dominance & .478*** & .391*** & .663*** \\
```

## Data Extraction Details

### Significance Stars

```python
def _get_sig_stars(p_value):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""
```

### PoMA Calculation

```python
poma = (indirect_effect / total_effect) * 100
# where indirect_effect = total_effect - direct_effect
```

### t-Statistics

```python
t = coefficient / se
p = 2 * (1 - t.cdf(abs(t), df=n-2))  # two-tailed
```

### 95% Confidence Intervals

```python
ci_lower = estimate - 1.96 * se
ci_upper = estimate + 1.96 * se
```

## Missing Data Handling

If metrics are missing from results:
- **Zero-order correlations**: Default to 0.000
- **LASSO selection**: Default to 0
- **R² values**: Default to 0.000
- **PoMA**: Default to 0.0%
- **DML effects**: Show 0.000 with appropriate warnings

## Troubleshooting

### Issue: Tables show all zeros
**Solution**: Check that HAAM model was fully run with `auto_run=True` or manually called all analysis methods

### Issue: PoMA values are negative
**Solution**: This can happen if indirect effect is negative. Check mediation analysis results.

### Issue: DML direct effect missing
**Solution**: Ensure DML analysis was run. Check `results['total_effects'][path]['check_beta']` exists.

### Issue: LaTeX compilation errors
**Solution**:
1. Check if `booktabs`, `siunitx` packages are installed
2. For siunitx issues, try removing `S[]` column specifications
3. Manually compile to see detailed errors: `pdflatex table1_...tex`

## Examples

See `example_latex_tables.py` for complete working examples:
- Generate all tables for a single trait
- Generate tables individually for debugging
- Batch generate for multiple traits
- Organize tables by type

## Comparison: Single-Trait vs Multi-Trait Tables

**Your examples** showed multi-trait tables (Prestige, Power, Dominance).

**These methods** generate single-trait tables (e.g., just Power).

**Why?**
- Cleaner debugging (one trait = one table = one file)
- Modular design (easy to combine later if needed)
- Trait-agnostic (same code works for any trait)
- Easier to edit individual trait results

**To create multi-trait tables:**
```python
# Generate for each trait
results_power = model_power.create_table_g_and_c("Power")
results_prestige = model_prestige.create_table_g_and_c("Prestige")

# Then manually combine the .tex files by editing
# Or write a custom combiner function
```

## Method Summary

| Table | Method | Purpose |
|-------|--------|---------|
| Table 1 | `create_table_zero_order_correlations()` | Bivariate correlations |
| Table 2 | `create_table_lasso_selection()` | Feature selection counts |
| Table 3 | `create_table_r2_and_poma()` | Model fit and mediation |
| Table 4 | `create_table_dml_effects()` | Regression coefficients |
| Table 7 | `create_table_g_and_c()` | Policy similarity & residual correlation |
| All | `create_all_latex_tables()` | Generate all 5 tables |

## Citation

When using these tables in publications, please cite the HAAM package:

```
Li, R. (2025). HAAM: Human-AI Alignment Model.
https://github.com/raymondli-me/haam
```

## Updates

**Version**: 1.0 (2025-11-05)
- Initial release of LaTeX table generation
- 5 publication-ready tables
- Automatic metric extraction
- Full statistical inference for DML effects
- CV and non-CV R² reporting

## Support

For issues or questions:
- GitHub: https://github.com/raymondli-me/haam/issues
- Documentation: https://raymondli-me.github.io/haam/
