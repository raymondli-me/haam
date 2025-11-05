# HAAM LaTeX Generation: Complete Reference Guide

**Last Updated:** 2025-11-05
**Version:** 1.0
**Author:** HAAM Development Team

---

## 🏗️ Architecture Overview

### Two-Layer System

1. **Wrapper Layer** (`haam/haam_init.py`):
   - `HAAM` class has wrapper methods
   - Called by user/scripts
   - Delegate to `self.visualizer.*` methods

2. **Implementation Layer** (`haam/haam_visualizations.py`):
   - `HAAMVisualizer` class has actual implementations
   - Lines 2411-3816 (inside the class!)
   - Does all the heavy lifting

### Critical Class Boundary

- `HAAMVisualizer` class: **lines 35-3816** in `haam_visualizations.py`
- LaTeX methods **MUST be indented** (inside class)
- Previously had duplicate unindented versions at line 2411 (now deleted in commit `e40a810`)

---

## 📊 The 7 LaTeX Tables/Figures

### 1. Table 1: Zero-Order Correlations

**Method:** `create_table_zero_order_correlations()`

**What:** Simple bivariate correlations

**Data Source:**
```python
results['total_effects']['X_AI']['coefficient']  # Validity--AI correlation
results['total_effects']['X_HU']['coefficient']  # Validity--Human correlation
results['total_effects']['HU_AI']['coefficient'] # Human--AI correlation
# Each also has: 'p_value', 'se', 't_stat', 'ci_lower', 'ci_upper'
```

**Output:** Single-page table with 3 correlations + significance stars

**Example:**
```latex
Table 1. Zero-Order Correlations (r) Between Validity and Judgments for Extraversion
─────────────────────────────────────────────────────────────────────────────
Construct    | Validity--Human | Validity--AI | Human--AI
─────────────────────────────────────────────────────────────────────────────
Extraversion | .279***         | .201***      | .489***
─────────────────────────────────────────────────────────────────────────────
```

---

### 2. Table 2: LASSO Feature Selection

**Method:** `create_table_lasso_selection()`

**What:** Number of PCs selected by LASSO

**Data Source:**
```python
results['debiased_lasso']['X']['n_selected']       # Number selected for Validity
results['debiased_lasso']['AI']['n_selected']      # Number selected for AI
results['debiased_lasso']['HU']['n_selected']      # Number selected for Human
results['pca_params']['n_components']              # Total PCs available
```

**Output Format:** "14 / 50", "42 / 50", "38 / 50" (selected / total)

**Example:**
```latex
Table 2. Number of Principal Components Selected by LASSO for Extraversion
─────────────────────────────────────────────────────────────────────────────
            | Model Predicting
            | Validity | AI Judgment | Human Judgment
─────────────────────────────────────────────────────────────────────────────
Extraversion| 14 / 50  | 42 / 50     | 38 / 50
─────────────────────────────────────────────────────────────────────────────
```

---

### 3. Table 3: R² and PoMA

**Method:** `create_table_r2_and_poma()`

**What:** Model fit and proportion of mediated accuracy

**Data Source:**
```python
# Cross-validated R²
results['debiased_lasso']['X']['r2_cv']
results['debiased_lasso']['AI']['r2_cv']
results['debiased_lasso']['HU']['r2_cv']

# Training set R² (both keys work)
results['debiased_lasso']['X']['r2']  # or 'r2_insample'

# PoMA values
results['mediation_analysis']['poma_ai']     # X → AI path
results['mediation_analysis']['poma_hu']     # X → HU path
results['mediation_analysis']['poma_hu_ai']  # HU → AI path
```

**Output:** Single-page table with 2 R² rows + PoMA row

---

### 4. Table 4: DML Effects

**Method:** `create_table_dml_effects()`

**What:** Full regression statistics for all paths

**Data Source:**
```python
results['total_effects']['X_AI']  # Contains: coefficient, se, t_stat, p_value, ci_lower, ci_upper
results['total_effects']['X_HU']
results['total_effects']['HU_AI']

# Each path also has DML check beta:
results['total_effects']['X_AI']['check_beta']
```

**Includes:** β, SE, t, p, 95% CI for:
- X → AI (Validity → AI Judgment)
- X → HU (Validity → Human Judgment)
- HU → AI (Human → AI Judgment)

**Output:** Single-page table with full inferential statistics

---

### 5. Table 7: G and C Parameters

**Method:** `create_table_g_and_c()`

**What:** Policy similarity (G) and residual correlation (C)

**Data Source:**
```python
# Policy similarities (G)
results['policy_similarities']['g_x_ai']
results['policy_similarities']['g_x_hu']
results['policy_similarities']['g_hu_ai']

# Residual correlations (C)
results['residual_correlations']['c_x_ai']
results['residual_correlations']['c_x_hu']
results['residual_correlations']['c_ai_hu']  # NOTE: 'c_ai_hu' NOT 'c_hu_ai'!
```

**Output:** Single-page table

**Note:** Policy similarity (G) measures correlation between LASSO predictions across models. Residual correlation (C) measures remaining correlation after controlling for PCs.

---

### 6. Figure 1: TikZ HAAM Diagram

**Method:** `create_latex_diagram()`

**What:** Publication-ready TikZ path diagram

**Parameters:**
```python
create_latex_diagram(
    trait_name="Extraversion",
    output_dir="./latex",
    n_pcs=15,              # Number of top PCs to show
    display=True
)
```

**Data Flow:**
1. Ranks PCs by tri-sum: `|coef_X| + |coef_AI| + |coef_HU|`
2. Shows top N PCs in perception space box
3. Draws arrows with R², PoMA, residual correlations

**Advanced Parameter (not in wrapper):**
```python
# To change coefficient display threshold:
haam_model.visualizer.create_latex_diagram(
    trait_name="Extraversion",
    output_dir="./latex",
    n_pcs=15,
    coef_threshold=0.05,   # Show "--" for |coef| < 0.05
    render_pdf=False,      # Don't attempt PDF compilation
    display=True
)
```

**Output:** Standalone LaTeX document with TikZ graphics

---

### 7. Comprehensive PC Coefficients Table

**Method:** `create_table_pc_coefficients_comprehensive()`

**What:** Multi-page table with ALL post-LASSO PCs

**Parameters:**
```python
create_table_pc_coefficients_comprehensive(
    trait_name="Extraversion",
    output_dir="./latex",
    min_trisum=0.0,    # Filter: only show PCs with tri-sum ≥ this
    display=True
)
```

**Shows:**
- 3 rows per PC (Validity, AI, Human)
- β, SE, t, 95% CI for each outcome
- Only PCs selected by at least one outcome

**Ranking:** By tri-sum (descending)

**Uses:** `longtable` package for multi-page support

**Filter Example:**
```python
# Only high-impact PCs:
create_table_pc_coefficients_comprehensive(
    trait_name="Extraversion",
    output_dir="./latex",
    min_trisum=0.15,    # Only PCs with tri-sum ≥ 0.15
    display=True
)
```

**Output:** Can be 5-20 pages depending on number of selected PCs

---

## 🔑 Critical Dictionary Keys

### Results Structure

```python
results = {
    'total_effects': {
        'X_AI': {
            'coefficient': 0.201,
            'se': 0.041,
            't_stat': 4.85,
            'p_value': 1.206e-06,
            'ci_lower': 0.120,
            'ci_upper': 0.282,
            'check_beta': 0.138  # DML direct effect
        },
        'X_HU': {...},
        'HU_AI': {...}
    },

    'debiased_lasso': {
        'X': {
            'n_selected': 14,
            'selected': [5, 6, 12, 18, ...],  # ← 'selected' NOT 'selected_indices'!
            'r2_cv': -0.0064,
            'r2': 0.0731,           # Training R² (also 'r2_insample')
            'coefs_std': np.array([...]),  # Post-LASSO OLS coefficients
            'se': np.array([...])   # Standard errors
        },
        'AI': {...},
        'HU': {...}
    },

    'mediation_analysis': {
        'poma_ai': 31.4,      # X → AI mediation percentage
        'poma_hu': 40.0,      # X → HU mediation percentage
        'poma_hu_ai': 24.7    # HU → AI mediation percentage
    },

    'policy_similarities': {
        'g_x_ai': 0.429,
        'g_x_hu': 0.484,
        'g_hu_ai': 0.780
    },

    'residual_correlations': {
        'c_x_ai': 0.139,
        'c_x_hu': 0.209,
        'c_ai_hu': 0.322  # ← NOTE: 'c_ai_hu' NOT 'c_hu_ai'!
    },

    'pca_params': {
        'n_components': 50
    }
}
```

---

## ⚠️ Known Quirks & Gotchas

### 1. Parameter Order Matters!

**Wrapper signatures in `haam_init.py`:**

```python
# Standard order for most tables:
def create_table_zero_order_correlations(trait_name, output_dir, display)
def create_table_lasso_selection(trait_name, output_dir, display)
def create_table_r2_and_poma(trait_name, output_dir, display)
def create_table_dml_effects(trait_name, output_dir, display)
def create_table_g_and_c(trait_name, output_dir, display)

# Special cases with extra parameters:
def create_latex_diagram(trait_name, output_dir, n_pcs, display)
def create_table_pc_coefficients_comprehensive(trait_name, output_dir, min_trisum, display)
```

**Visualizer signatures in `haam_visualizations.py` MUST match exactly!**

**Why this matters:**
- If order mismatches → `TypeError: expected str, bytes or os.PathLike object, not int/float`
- Example bug: `n_pcs=15` went to `output_dir` parameter → tried to use int as file path!

**Fixed in commits:** `4d126a5`, `9042b30`

---

### 2. Dictionary Key Name Inconsistencies

**WRONG keys that DON'T exist:**
- ❌ `'selected_indices'` → **Use `'selected'`**
- ❌ `'c_hu_ai'` → **Use `'c_ai_hu'`**

**Alternative key names (both work):**
- ✅ `'r2'` or `'r2_insample'` (training set R²)
- ✅ `'r2_cv'` or `'r2_cv_lasso'` (cross-validated R²)

**Bug example:**
```python
# WRONG - crashes with empty data:
selected = debiased.get('X', {}).get('selected_indices', [])

# CORRECT:
selected = debiased.get('X', {}).get('selected', [])
```

**Fixed in commit:** `1209783`

---

### 3. Empty Data Handling

**If LASSO selects NO PCs:**
```python
if len(pc_data) == 0:
    if display:
        print("⚠ No PCs selected by LASSO")
    return {'tex_path': None}  # ← Returns None!
```

**Then calling code crashes:**
```python
# This will crash if tex_path is None:
os.path.basename(result['tex_path'])  # TypeError!
```

**Safe usage:**
```python
result = haam_model.create_table_pc_coefficients_comprehensive(...)
if result and result.get('tex_path'):
    print(f"✓ Saved: {os.path.basename(result['tex_path'])}")
else:
    print("⚠ Skipped: No PCs selected")
```

---

### 4. Coefficient Arrays vs Selected Indices

**Arrays are dense** (length = `n_components`):
```python
coefs_std = [0.0, -0.1, 0.0, 0.0, 0.2, ...]  # 50 elements, many zeros
```

**Selected is sparse** (only non-zero indices):
```python
selected = [1, 4, 12, 18, ...]  # Only indices where LASSO selected
```

**Accessing coefficients correctly:**
```python
# Get coefficient for PC 5 (0-indexed = 4):
if 4 in selected:
    coef = coefs_std[4]
else:
    coef = 0.0  # Not selected
```

---

### 5. Standard Errors May Be Missing

```python
se = debiased.get('X', {}).get('se', np.array([]))

# Always check length before indexing:
if len(se) > idx:
    se_val = se[idx]
else:
    se_val = 0.0
```

---

### 6. PC Numbering: 0-indexed vs 1-indexed

**Internally (Python):** 0-indexed
```python
selected = [0, 5, 11]  # Refers to PC1, PC6, PC12
```

**In output (LaTeX):** 1-indexed
```python
pc_num = idx + 1  # Convert for display
# Output: "PC1", "PC6", "PC12"
```

**Always add 1 when displaying to users!**

---

### 7. Tri-Sum Ranking

**Definition:**
```python
tri_sum = |coef_X| + |coef_AI| + |coef_HU|
```

**Used for:**
- Ranking PCs in TikZ diagram (shows top N by tri-sum)
- Ranking PCs in comprehensive table
- Filtering with `min_trisum` parameter

**Example:**
```python
# High-impact PC:
PC5: coef_X=-0.11, coef_AI=-0.15, coef_HU=-0.37
tri_sum = 0.11 + 0.15 + 0.37 = 0.63  # Shows up high in rankings

# Low-impact PC:
PC22: coef_X=0.01, coef_AI=0.0, coef_HU=0.02
tri_sum = 0.01 + 0.0 + 0.02 = 0.03  # May be filtered out
```

---

### 8. Significance Stars

```python
def _get_sig_stars(self, p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""
```

**In LaTeX tables:**
```latex
.504***  % p < .001
.201**   % p < .01
.089*    % p < .05
.023     % n.s.
```

---

### 9. PoMA Calculation

**Definition:** Proportion of Mediated Accuracy
```python
PoMA = (indirect_effect / total_effect) × 100%
```

**Interpretation:**
- PoMA = 31.4% → 31.4% of the X→AI relationship flows through measured PCs
- PoMA = 0% → All direct, none mediated
- PoMA = 100% → Fully mediated

**Where to find:**
```python
results['mediation_analysis']['poma_ai']    # X → AI path
results['mediation_analysis']['poma_hu']    # X → HU path
results['mediation_analysis']['poma_hu_ai'] # HU → AI path
```

**In TikZ diagram:** Shows on arrows as percentages

---

### 10. Residual Correlation (C) vs Total Correlation (r)

**Total effect (r):** Simple bivariate correlation
```python
r = results['total_effects']['X_AI']['coefficient']  # e.g., 0.201
```

**Residual correlation (C):** After controlling for PCs
```python
C = results['residual_correlations']['c_x_ai']  # e.g., 0.139
```

**Relationship:**
```
Total effect = Direct effect + Indirect effect
r = C + (mediated portion)

If PoMA = 31.4%:
  Mediated = 0.201 × 0.314 = 0.063
  Direct (C) = 0.201 - 0.063 = 0.138
```

---

## 🛠️ Helper Methods (Internal)

### In HAAMVisualizer Class

```python
_get_ranked_pcs_trisum(n_pcs, coef_threshold)
    """Get top N PCs ranked by tri-sum for TikZ diagram"""

_get_all_postlasso_pcs_trisum_ranked(min_trisum)
    """Get ALL selected PCs for comprehensive table"""

_format_coef(coef, threshold)
    """Format coefficient: '.12' or '--' if |coef| < threshold"""

_generate_latex_tikz(trait_name, metrics, pc_data, n_components_total)
    """Generate the full TikZ LaTeX code"""

_render_latex_to_pdf(tex_path, display)
    """Attempt to compile .tex → .pdf (requires pdflatex)"""

_calculate_visualization_metrics()
    """Extract all metrics from results dict"""

_get_sig_stars(p_value)
    """Returns: '***', '**', '*', or ''"""

_generate_pc_table_rows(pc, pc_num)
    """Generate 3 LaTeX table rows for one PC"""

_calculate_pc_stats(coef, se, is_selected)
    """Calculate t, p, CI for one coefficient"""

_get_sample_size()
    """Get sample size from results (with fallback to 500)"""
```

---

## 🔄 Data Flow

```
User calls:
  haam_model.create_table_zero_order_correlations("Extraversion", "./latex", True)
    ↓
HAAM wrapper (haam_init.py):
  Validates inputs
  Checks self.visualizer exists
  Calls: self.visualizer.create_table_zero_order_correlations("Extraversion", "./latex", True)
    ↓
HAAMVisualizer method (haam_visualizations.py):
  1. os.makedirs(output_dir, exist_ok=True)
  2. Extract data from self.results dictionary
  3. Generate LaTeX string with proper formatting
  4. Write to .tex file
  5. Optional: Print status if display=True
  6. Return {'tex_path': '/path/to/file.tex'}
    ↓
User receives:
  {'tex_path': '/content/.../table1_zero_order_correlations_extraversion.tex'}
```

---

## 🐛 Past Bugs (Now Fixed)

### Bug 1: Methods Outside Class
**Commit:** `e40a810`
**Date:** 2025-11-05

**Problem:**
- LaTeX methods starting at line 2411 were unindented (module-level functions)
- Python AST parser showed HAAMVisualizer class ended at line 2409
- Methods were defined but not accessible as class methods

**Symptom:**
```
AttributeError: 'HAAMVisualizer' object has no attribute 'create_table_zero_order_correlations'
```

**Fix:**
- Deleted duplicate unindented versions (440 lines, 2411-2850)
- Kept only properly indented versions that were inside the class
- Verified with AST parser: class now has 34 methods including 7 LaTeX methods

---

### Bug 2: Parameter Order Mismatch
**Commits:** `4d126a5`, `9042b30`
**Date:** 2025-11-05

**Problem:**
- Wrapper in `haam_init.py` had: `(trait_name, n_pcs, output_dir, display)`
- Visualizer in `haam_visualizations.py` had: `(trait_name, output_dir, n_pcs, display)`
- When calling with keyword args, `n_pcs=15` went to `output_dir` parameter

**Symptom:**
```
TypeError: expected str, bytes or os.PathLike object, not int
```

**Fix:**
- Reordered wrapper parameters to match visualizer signature
- Applied to both `create_latex_diagram` and `create_table_pc_coefficients_comprehensive`
- Updated all calls in `create_all_latex_tables` to use correct order

---

### Bug 3: Wrong Dictionary Key
**Commit:** `1209783`
**Date:** 2025-11-05

**Problem:**
- Code looked for `debiased.get('X', {}).get('selected_indices', [])`
- Actual key in results is `'selected'`, not `'selected_indices'`
- Returned empty list → `len(all_selected) == 0` → returned `{'tex_path': None}`

**Symptom:**
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**Root Cause:**
```python
# In haam_package.py line 340:
'selected': selected,  # ← Actual key name

# In haam_visualizations.py line 3595 (WRONG):
selected_x = set(debiased.get('X', {}).get('selected_indices', []))

# Should be:
selected_x = set(debiased.get('X', {}).get('selected', []))
```

**Fix:**
- Changed all 3 lines (X, AI, HU) to use correct key name `'selected'`

---

## 📦 LaTeX Package Requirements

All generated `.tex` files are standalone documents. Required packages:

### Required for All Tables
```latex
\usepackage{booktabs}       % Professional table rules
\usepackage{caption}        % Caption formatting
\usepackage[margin=1in]{geometry}  % Page margins
```

### Table-Specific Requirements
```latex
% Comprehensive PC table only:
\usepackage{longtable}      % Multi-page tables
\usepackage{multirow}       % Span rows
\usepackage{array}          % Column formatting

% Some tables use:
\usepackage{threeparttable} % Table notes
\usepackage{siunitx}        % Number alignment
```

### TikZ Diagram Requirements
```latex
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric}
```

### Installation

**macOS:**
```bash
brew install basictex
sudo tlmgr update --self
sudo tlmgr install booktabs longtable multirow threeparttable siunitx
```

**Ubuntu:**
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra
```

**Windows:**
- Download MiKTeX from [miktex.org](https://miktex.org)
- Packages install automatically on first compile

---

## 🚀 Usage Patterns

### Generate All Tables at Once

```python
from haam import HAAM

# Load or create HAAM model
haam_model = HAAM(...)

# Generate all 7 LaTeX outputs
results = haam_model.create_all_latex_tables(
    trait_name="Extraversion",
    output_dir="./latex_tables",
    n_pcs_diagram=15,      # For TikZ diagram
    min_trisum=0.0,        # For comprehensive table (0.0 = include all)
    display=True           # Print progress
)

# Results is a dict with all file paths:
# {
#     'tex_path_table1': '...',
#     'tex_path_table2': '...',
#     ...
# }
```

### Generate Individual Tables

```python
# Simple tables (standard 3 parameters):
table1 = haam_model.create_table_zero_order_correlations(
    trait_name="Extraversion",
    output_dir="./latex",
    display=True
)

table2 = haam_model.create_table_lasso_selection(
    trait_name="Extraversion",
    output_dir="./latex",
    display=True
)

table3 = haam_model.create_table_r2_and_poma(
    trait_name="Extraversion",
    output_dir="./latex",
    display=True
)

table4 = haam_model.create_table_dml_effects(
    trait_name="Extraversion",
    output_dir="./latex",
    display=True
)

table7 = haam_model.create_table_g_and_c(
    trait_name="Extraversion",
    output_dir="./latex",
    display=True
)

# Complex tables (extra parameters):
diagram = haam_model.create_latex_diagram(
    trait_name="Extraversion",
    output_dir="./latex",
    n_pcs=15,              # Show top 15 PCs
    display=True
)

pc_table = haam_model.create_table_pc_coefficients_comprehensive(
    trait_name="Extraversion",
    output_dir="./latex",
    min_trisum=0.0,        # Include all selected PCs
    display=True
)
```

### Safe Usage Pattern

```python
# Always check for None returns:
result = haam_model.create_table_pc_coefficients_comprehensive(
    "Extraversion", "./latex", min_trisum=0.0, display=False
)

if result and result.get('tex_path'):
    print(f"✓ Generated: {result['tex_path']}")

    # Compile to PDF (if pdflatex available):
    import subprocess
    subprocess.run(['pdflatex', result['tex_path']], cwd='./latex')
else:
    print("⚠ No PCs selected - table not generated")
```

---

## 📝 Common Modifications

### Change Number of PCs in Diagram

```python
# Show top 20 PCs instead of default 15:
diagram = haam_model.create_latex_diagram(
    trait_name="Extraversion",
    output_dir="./latex",
    n_pcs=20,
    display=True
)
```

### Filter Comprehensive Table

```python
# Only show high-impact PCs (tri-sum ≥ 0.15):
pc_table = haam_model.create_table_pc_coefficients_comprehensive(
    trait_name="Extraversion",
    output_dir="./latex",
    min_trisum=0.15,  # Filter threshold
    display=True
)

# This might reduce output from 15 pages to 3-5 pages
```

### Change Coefficient Display Threshold in Diagram

```python
# Need to call visualizer directly (not exposed in wrapper):
# Default threshold is 0.05 (shows "--" for |coef| < 0.05)

diagram = haam_model.visualizer.create_latex_diagram(
    trait_name="Extraversion",
    output_dir="./latex",
    n_pcs=15,
    coef_threshold=0.10,   # Higher threshold → more "--" shown
    render_pdf=False,      # Don't attempt PDF rendering
    display=True
)
```

### Batch Generate for Multiple Traits

```python
traits = ['Extraversion', 'Agreeableness', 'Conscientiousness',
          'Emotional_Stability', 'Openness']

for trait in traits:
    # Create trait-specific output directory
    output_dir = f"./latex_tables/{trait}"

    # Generate all tables
    results = haam_models[trait].create_all_latex_tables(
        trait_name=trait,
        output_dir=output_dir,
        n_pcs_diagram=15,
        min_trisum=0.0,
        display=True
    )

    print(f"✓ {trait}: Generated {len(results)} LaTeX files")
```

---

## 📖 Including in Your Paper

### Option 1: Direct Include

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}

\begin{document}

\section{Results}

% Include generated tables
\input{latex_tables/table1_zero_order_correlations_extraversion.tex}

\input{latex_tables/table3_r2_and_poma_extraversion.tex}

\input{latex_tables/table4_dml_effects_extraversion.tex}

\end{document}
```

### Option 2: Extract Table Body Only

If your paper has custom formatting, extract just the table body:

1. Open generated `.tex` file
2. Copy everything between `\begin{tabular}` and `\end{tabular}`
3. Paste into your paper's table environment

### Option 3: Compile Separately

```bash
cd latex_tables

# Compile each table to PDF
pdflatex table1_zero_order_correlations_extraversion.tex
pdflatex table3_r2_and_poma_extraversion.tex
pdflatex haam_diagram_extraversion.tex

# Include PDFs in paper
# In your main document:
# \includegraphics{latex_tables/table1_zero_order_correlations_extraversion.pdf}
```

---

## ✅ Testing Checklist

When modifying LaTeX generation code:

- [ ] Check class indentation (methods must be inside HAAMVisualizer)
- [ ] Verify parameter order matches between wrapper and visualizer
- [ ] Use correct dictionary keys (`'selected'` not `'selected_indices'`)
- [ ] Handle empty data (check for `None` returns)
- [ ] Test with different traits to ensure `trait_name` is used correctly
- [ ] Check file paths use `os.path.join()` not string concatenation
- [ ] Verify LaTeX compiles (test with `pdflatex` or Overleaf)
- [ ] Check for proper escaping of special LaTeX characters
- [ ] Test with different `n_components` (e.g., 25, 50, 100)
- [ ] Test with cases where few PCs are selected
- [ ] Test with cases where many PCs are selected (>40)
- [ ] Verify significance stars match p-values
- [ ] Check that tri-sum ranking is correct
- [ ] Ensure PC numbering is 1-indexed in output

---

## 🔍 Debugging Tips

### Issue: "AttributeError: object has no attribute 'create_table_*'"

**Cause:** LaTeX methods not in HAAMVisualizer class

**Check:**
```python
import ast
with open('haam/haam_visualizations.py') as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'HAAMVisualizer':
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        print(f"Methods in class: {len(methods)}")
        print("LaTeX methods:", [m for m in methods if 'create_table' in m])
```

**Fix:** Ensure proper indentation (4 spaces for methods inside class)

---

### Issue: "TypeError: expected str, bytes or os.PathLike object, not int"

**Cause:** Parameter order mismatch

**Check:**
```python
# Wrapper signature (haam_init.py):
def create_latex_diagram(self, trait_name, output_dir, n_pcs, display)

# Visualizer signature (haam_visualizations.py):
def create_latex_diagram(self, trait_name, output_dir, n_pcs, coef_threshold, render_pdf, display)

# Must match positionally!
```

**Fix:** Reorder parameters or use keyword arguments

---

### Issue: Comprehensive table is empty (returns None)

**Cause:** Wrong dictionary key or no PCs selected

**Debug:**
```python
# Check what PCs were selected:
debiased = haam_model.analysis.results['debiased_lasso']
for outcome in ['X', 'AI', 'HU']:
    selected = debiased[outcome]['selected']
    print(f"{outcome}: {len(selected)} PCs selected: {selected}")

# Should show non-empty lists
```

**Fix:** Verify LASSO actually selected some PCs; check key name is `'selected'`

---

### Issue: LaTeX won't compile

**Cause:** Missing packages or syntax errors

**Debug:**
```bash
# Try compiling manually:
pdflatex -interaction=nonstopmode table1_zero_order_correlations_extraversion.tex

# Check the .log file for errors:
grep -i error table1_zero_order_correlations_extraversion.log
```

**Common fixes:**
- Install missing packages: `sudo tlmgr install <package_name>`
- Check for special characters in `trait_name` (need escaping: `_`, `&`, `%`, etc.)

---

## 📚 References

### HAAM Paper
> Li, R., & Biesanz, J. C. (2025). High-Dimensional Perception with the Double Machine Learning Lens Model. *PsyArXiv*. https://doi.org/10.31234/osf.io/ubwgk

### LaTeX Resources
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [TikZ Manual](https://tikz.dev/)
- [booktabs Package](https://ctan.org/pkg/booktabs)
- [longtable Package](https://ctan.org/pkg/longtable)

---

## 🤝 Contributing

Found a bug or want to improve LaTeX generation?

1. Check existing issues: https://github.com/raymondli-me/haam/issues
2. Open a new issue with:
   - Description of the problem
   - Minimal reproducible example
   - Expected vs actual output
3. Submit PR with:
   - Clear description of changes
   - Tests if adding new functionality
   - Updated documentation

---

## 📜 Changelog

### Version 1.0 (2025-11-05)
- ✅ All 7 LaTeX table/figure generators working
- ✅ Fixed class indentation issues (commit `e40a810`)
- ✅ Fixed parameter order mismatches (commits `4d126a5`, `9042b30`)
- ✅ Fixed dictionary key errors (commit `1209783`)
- ✅ Comprehensive documentation created
- ✅ Tested with NCDS personality data (5 traits, 563 participants)

---

**End of Guide**
