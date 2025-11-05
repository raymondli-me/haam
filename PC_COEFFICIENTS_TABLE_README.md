# Comprehensive PC Coefficients Table

## Overview

Generate a multi-page table showing **ALL** principal components selected by LASSO, ranked by tri-sum, with full regression statistics for Validity, AI, and Human judgments.

## Key Features

✅ **ALL Post-LASSO PCs** - Shows every PC selected by ANY outcome
✅ **Tri-Sum Ranking** - Ordered by |coef_X| + |coef_AI| + |coef_HU|
✅ **Full Inference** - β, SE, t, p, 95% CI for each outcome
✅ **Multi-Page Support** - Uses `longtable` for automatic page breaks
✅ **Handles Missing** - Shows "--" if PC not selected for that outcome
✅ **Publication-Ready** - Professional formatting with `booktabs`, `multirow`

## Quick Start

```python
from haam import HAAM

haam_model = HAAM.load("path/to/model.pkl")

result = haam_model.create_table_pc_coefficients_comprehensive(
    trait_name="Social Class",
    output_dir="./latex_tables",
    min_trisum=0.0,  # Include all PCs (default)
    display=True
)

print(result['tex_path'])
# Output: ✓ Comprehensive PC Coefficients Table saved to: ...
#         - 45 PCs included (ranked by tri-sum)
#         - Multi-page support via longtable
```

## What It Generates

### Structure

For each PC, shows **3 rows**:
```
PC3  | Validity | -0.127 | 0.010 | -13.15*** | [-0.146, -0.108]
     | AI       | -0.376 | 0.006 | -58.51*** | [-0.389, -0.364]
     | Human    | -0.385 | 0.033 | -11.74*** | [-0.449, -0.321]
```

### Columns

| Column | Content | Description |
|--------|---------|-------------|
| PC | PC number | Bold (e.g., **PC3**) |
| Model | Outcome | Validity / AI / Human |
| β | Coefficient | Post-LASSO OLS estimate |
| SE | Standard Error | From robust covariance |
| t | t-statistic | β / SE with sig stars |
| 95% CI | Confidence Interval | [β - 1.96×SE, β + 1.96×SE] |

### Example Output

```latex
──────────────────────────────────────────────────────────────────
PC    | Model    | β       | SE    | t        | 95% CI
──────────────────────────────────────────────────────────────────
PC3   | Validity | -0.127  | 0.010 | -13.15***| [-0.146, -0.108]
      | AI       | -0.376  | 0.006 | -58.51***| [-0.389, -0.364]
      | Human    | -0.385  | 0.033 | -11.74***| [-0.449, -0.321]

PC2   | Validity | -0.080  | 0.010 | -8.21*** | [-0.099, -0.061]
      | AI       | -0.293  | 0.007 | -43.32***| [-0.306, -0.280]
      | Human    | -0.428  | 0.034 | -12.40***| [-0.495, -0.360]

PC6   | Validity | --      | --    | --       | --
      | AI       | 0.119   | 0.007 | 17.60*** | [0.106, 0.132]
      | Human    | 0.224   | 0.034 | 6.57***  | [0.157, 0.291]

PC106 | Validity | --      | --    | --       | --
      | AI       | --      | --    | --       | --
      | Human    | -0.085  | 0.035 | -2.46*   | [-0.153, -0.017]
──────────────────────────────────────────────────────────────────
```

**Note**: PCs ranked by tri-sum (highest impact first). PC3 has the largest combined effect across all three outcomes.

## Tri-Sum Ranking

### What is Tri-Sum?

```
tri_sum = |coef_Validity| + |coef_AI| + |coef_Human|
```

### Why Rank by Tri-Sum?

1. **Total Impact** - Shows PCs with largest combined effect
2. **Cross-Outcome Importance** - Captures consistency across models
3. **Prioritization** - Most impactful PCs appear first

### Example Ranking

| Rank | PC | Tri-Sum | X | AI | HU |
|------|----|---------|----|----|----|
| 1 | PC3 | 0.888 | -0.127 | -0.376 | -0.385 |
| 2 | PC2 | 0.801 | -0.080 | -0.293 | -0.428 |
| 3 | PC4 | 0.724 | 0.125 | 0.280 | 0.319 |
| ... | ... | ... | ... | ... | ... |
| 45 | PC106 | 0.085 | 0.000 | 0.000 | -0.085 |

## Missing Coefficients

### When to Show "--"

A PC shows "--" for an outcome if:
- **Not selected by LASSO** for that outcome
- Coefficient = 0 (not in the model)

### Example: PC6

```
PC6 | Validity | --    | --  | --  | --              (not selected)
    | AI       | 0.119 | ... | ... | [0.106, 0.132]  (selected)
    | Human    | 0.224 | ... | ... | [0.157, 0.291]  (selected)
```

**Interpretation**: PC6 predicts AI and Human judgments but not self-reported Validity.

## Multi-Page Support

### Using `longtable`

Unlike regular `table` environment, `longtable`:
- **Automatically breaks** across pages
- **Repeats headers** on each page
- **Shows continuation** indicators

### Header Behavior

**First Page**:
```latex
Principal Component Predictors of Social Class: Validity and Judgment Coefficients
────────────────────────────────────────────────────────────────
PC | Model | β | SE | t | 95% CI
────────────────────────────────────────────────────────────────
```

**Subsequent Pages**:
```latex
Table X -- continued from previous page
────────────────────────────────────────────────────────────────
PC | Model | β | SE | t | 95% CI
────────────────────────────────────────────────────────────────
```

**Page Footer**:
```latex
────────────────────────────────────────────────────────────────
                                          Continued on next page
```

### Compiling Multi-Page Tables

```bash
pdflatex table_pc_coefficients_comprehensive_social_class.tex
# Output: 3-page PDF (or however many pages needed)
```

## Parameters

### `create_table_pc_coefficients_comprehensive()`

```python
def create_table_pc_coefficients_comprehensive(
    trait_name: str = "Trait",      # Name for caption/labels
    output_dir: str = "./",          # Where to save .tex
    min_trisum: float = 0.0,         # Minimum tri-sum to include
    display: bool = True             # Print progress
) -> Dict[str, str]:                # Returns {'tex_path': ...}
```

### Parameter Details

**`trait_name`** (str)
- Used in table caption and note
- Example: "Social Class", "Power", "Extraversion"

**`output_dir`** (str)
- Directory to save the .tex file
- Created if doesn't exist

**`min_trisum`** (float)
- Filter out low-impact PCs
- Default: 0.0 (include all)
- Example: 0.10 (only show PCs with tri-sum ≥ 0.10)

**`display`** (bool)
- Whether to print status messages
- Shows number of PCs included

## Usage Examples

### Example 1: All PCs (Default)

```python
haam_model = HAAM.load("model.pkl")

result = haam_model.create_table_pc_coefficients_comprehensive(
    trait_name="Power"
)

# Shows ALL PCs selected by any outcome
```

### Example 2: Filter by Tri-Sum

```python
# Only show PCs with substantial impact
result = haam_model.create_table_pc_coefficients_comprehensive(
    trait_name="Social Class",
    min_trisum=0.15  # Only PCs with tri-sum ≥ 0.15
)

# Fewer PCs in table, focusing on most important
```

### Example 3: Batch Generate for Multiple Traits

```python
traits = ["Extraversion", "Agreeableness", "Conscientiousness",
          "Emotional_Stability", "Openness"]

for trait in traits:
    model_path = f"models/{trait}_*/haam_model.pkl"
    model = HAAM.load(glob.glob(model_path)[0])

    result = model.create_table_pc_coefficients_comprehensive(
        trait_name=trait,
        output_dir=f"./pc_tables/{trait}"
    )
```

### Example 4: Organize by Min Tri-Sum Threshold

```python
# Generate multiple versions with different thresholds
thresholds = [0.0, 0.10, 0.20, 0.30]

for threshold in thresholds:
    result = haam_model.create_table_pc_coefficients_comprehensive(
        trait_name="Power",
        output_dir=f"./tables/trisum_{threshold:.2f}",
        min_trisum=threshold
    )
```

## Data Extraction

### What Gets Extracted

From `self.results['debiased_lasso']`:

| Data | Source | Description |
|------|--------|-------------|
| Coefficients | `[outcome]['coefs_std']` | Post-LASSO OLS estimates |
| Standard Errors | `[outcome]['se']` | From robust covariance |
| Selected Indices | `[outcome]['selected_indices']` | Which PCs were selected |

### LASSO Selection Logic

```python
# PC is included if selected by ANY outcome
selected_x = set(results['X']['selected_indices'])       # e.g., {0, 1, 3, 4}
selected_ai = set(results['AI']['selected_indices'])     # e.g., {0, 1, 2, 5, 6}
selected_hu = set(results['HU']['selected_indices'])     # e.g., {0, 2, 4, 5}

all_selected = selected_x | selected_ai | selected_hu    # Union: {0,1,2,3,4,5,6}
# Include all 7 PCs in the table
```

### Statistics Calculation

For each PC-outcome pair:

**If selected by LASSO:**
```python
t = β / SE
p = 2 * (1 - t.cdf(|t|, df=n-2))      # two-tailed
ci_lower = β - 1.96 × SE
ci_upper = β + 1.96 × SE
sig_stars = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
```

**If NOT selected:**
```python
β = '--'
SE = '--'
t = '--'
CI = '--'
```

## LaTeX Dependencies

Required packages:
```latex
\usepackage{booktabs}       % Professional tables
\usepackage{longtable}      % Multi-page support
\usepackage{multirow}       % PC labels span 3 rows
\usepackage{array}          % Column formatting
\usepackage{threeparttable} % Table notes at bottom
\usepackage{caption}        % Caption formatting
```

## Customization

### Adjust Spacing Between PCs

Edit the generated .tex file:
```latex
% Current: 0.6em spacing
\addlinespace[0.6em]

% Change to 1em for more space
\addlinespace[1.0em]
```

### Change Column Alignment

```latex
% Current: right-aligned numbers
\begin{longtable}{llrrrr}

% Change to centered numbers
\begin{longtable}{llccccc}
```

### Modify Coefficient Precision

In the code, change `:.3f` to desired precision:
```python
# 3 decimals (current)
coef_str = f"{coef:.3f}"

# 4 decimals (more precise)
coef_str = f"{coef:.4f}"
```

### Add Row Colors (Alternating)

Add to preamble:
```latex
\usepackage[table]{xcolor}
\rowcolors{2}{gray!10}{white}
```

## File Naming

Generated files follow this pattern:
```
table_pc_coefficients_comprehensive_<trait_name>.tex
```

Examples:
```
table_pc_coefficients_comprehensive_social_class.tex
table_pc_coefficients_comprehensive_power.tex
table_pc_coefficients_comprehensive_extraversion.tex
```

## Integration with Paper

### Option 1: Direct \input

```latex
% In your paper
\input{tables/table_pc_coefficients_comprehensive_power.tex}
```

### Option 2: Extract longtable

Copy the `\begin{longtable}...\end{longtable}` block into your paper.

### Option 3: Compile Separately

```bash
cd tables
pdflatex table_pc_coefficients_comprehensive_power.tex
# Then \includegraphics{...pdf} in your paper
```

## Troubleshooting

### Issue: Table is too wide

**Solution 1**: Use smaller font
```latex
\small  % or \footnotesize or \scriptsize
\begin{longtable}{...}
```

**Solution 2**: Landscape orientation
```latex
\usepackage{pdflscape}
\begin{landscape}
\begin{longtable}{...}
\end{longtable}
\end{landscape}
```

### Issue: Too many PCs (50+ pages)

**Solution**: Use `min_trisum` filter
```python
# Only show PCs with tri-sum ≥ 0.20
result = model.create_table_pc_coefficients_comprehensive(
    trait_name="Power",
    min_trisum=0.20  # Reduces table to ~3-5 pages
)
```

### Issue: Compilation errors

**Common causes**:
1. Missing `longtable` package
2. Missing `multirow` package
3. Missing `threeparttable` package

**Solution**:
```bash
# Ubuntu
sudo apt-get install texlive-latex-extra

# macOS
brew install basictex
sudo tlmgr install longtable multirow threeparttable
```

### Issue: "--" appears for all outcomes

**Cause**: No PCs selected by LASSO (all coefficients = 0)

**Solution**: Check LASSO results:
```python
print(model.results['debiased_lasso']['X']['n_selected'])
# Should be > 0
```

## Expected Output Size

### Typical PC Counts

| Scenario | n_components | Typical Selected | Table Pages |
|----------|--------------|------------------|-------------|
| Conservative | 50 | 20-30 | 2-3 pages |
| Standard | 100 | 30-50 | 3-5 pages |
| Liberal | 200 | 50-100 | 5-10 pages |

### Estimate Pages

```
pages ≈ n_selected_pcs × 3 rows / 15 rows_per_page ≈ n_selected / 5
```

Example: 45 PCs selected → ~9 pages

## Comparison with Other Tables

| Feature | Table 2 (LASSO) | This Table (Comprehensive) |
|---------|-----------------|----------------------------|
| **Shows** | Count of PCs | Full statistics for each PC |
| **Detail** | Summary | Detailed |
| **Length** | 1 row per trait | 3 rows × n_PCs |
| **Pages** | Single page | Multi-page |
| **Purpose** | Quick overview | Deep dive |

**Use Table 2 when**: You want a quick summary of feature selection.
**Use Comprehensive when**: You need full regression output for each PC.

## Citation

When using this table in publications:

```
Li, R. (2025). HAAM: Human-AI Alignment Model.
https://github.com/raymondli-me/haam
```

## Updates

**Version**: 1.0 (2025-11-05)
- Initial release of comprehensive PC coefficients table
- Multi-page support via longtable
- Tri-sum ranking across all three outcomes
- Full inference statistics with significance testing
- Handles missing coefficients gracefully

## Support

For issues or questions:
- GitHub: https://github.com/raymondli-me/haam/issues
- Documentation: https://raymondli-me.github.io/haam/
