# LaTeX/TikZ Diagram Generation for HAAM

## Overview

The HAAM package now includes automatic generation of publication-ready LaTeX/TikZ diagrams that visualize the 3-way relationship between X (criterion), AI judgments, and Human judgments.

## Features

✅ **Automatic number plugging** - Extracts all metrics from HAAM results
✅ **Tri-mode PC ranking** - Ranks PCs by sum of |X| + |AI| + |HU| coefficients
✅ **Customizable display** - Show top N PCs, adjust coefficient threshold
✅ **PDF rendering** - Optional automatic compilation to PDF (requires pdflatex)
✅ **Cross-platform** - Works on macOS, Linux, and Windows with error handling
✅ **Publication-ready** - TikZ diagrams suitable for academic journals

## Quick Start

```python
from haam import HAAM

# Load your HAAM model
haam_model = HAAM.load("path/to/model.pkl")

# Generate LaTeX diagram
result = haam_model.create_latex_diagram(
    trait_name="Extraversion",
    output_dir="./latex_output",
    n_pcs=15,              # Show top 15 PCs
    coef_threshold=0.05,   # Show "--" for |coef| < 0.05
    render_pdf=True,       # Compile to PDF
    display=True           # Print progress
)

print(f"LaTeX file: {result['tex_path']}")
print(f"PDF file: {result.get('pdf_path', 'Not rendered')}")
```

## What's Generated

### The LaTeX/TikZ diagram shows:

1. **X node** (criterion/validity)
   - R²_CV from debiased LASSO model predicting X

2. **AI and HU nodes** (judgment models)
   - R²_CV from debiased LASSO models predicting AI and HU

3. **Perception Space** (central box)
   - Top N PCs ranked by tri-mode sum: |coef_X| + |coef_AI| + |coef_HU|
   - Format: **PCX** followed by (X AI HU) coefficients
   - Coefficients below threshold shown as "--"

4. **Residual correlations** (curved arrows)
   - Correlations between residuals after controlling for PCs
   - Unmediated percentages: (100% - PoMA)

5. **Note section** (below diagram)
   - Total effects (simple regression coefficients)
   - PoMA (Proportion of Mediated Accuracy) values
   - Significance stars

## Parameters

### `create_latex_diagram()`

```python
def create_latex_diagram(
    trait_name: str = "Trait",       # Name for caption and label
    output_dir: str = "./",           # Where to save .tex file
    n_pcs: int = 15,                  # Number of PCs to display
    coef_threshold: float = 0.05,     # |coef| < this shows "--"
    render_pdf: bool = False,         # Attempt PDF compilation
    display: bool = True              # Print status messages
) -> Dict[str, str]:                 # Returns {'tex_path': ..., 'pdf_path': ...}
```

## Tri-Mode Ranking

The `_get_ranked_pcs_trisum()` method ranks PCs by:

```
tri_sum = |coef_X| + |coef_AI| + |coef_HU|
```

This differs from the existing `get_top_pcs(ranking_method='triple')` which takes top 3 from each outcome. The tri-sum method:
- Considers **all three outcomes simultaneously**
- Ranks by **total importance across all three**
- Ensures PCs with consistent effects across outcomes rank highly

## Extracted Metrics

The diagram automatically extracts:

| Metric | Source | Description |
|--------|--------|-------------|
| R²_CV (X, AI, HU) | `debiased_lasso[outcome]['r2_cv']` | Cross-validated R² |
| PC coefficients | `debiased_lasso[outcome]['coefs_std']` | Post-LASSO OLS standardized coefficients |
| Total effects | `total_effects[path]['coefficient']` | Simple regression (X→AI, X→HU, HU→AI) |
| Residual correlations | `residual_correlations[path]` | Correlation after controlling for PCs |
| PoMA | `mediation_analysis[outcome]` | Proportion of Mediated Accuracy |

## PDF Rendering

If `render_pdf=True`, the method attempts to compile the LaTeX file to PDF using `pdflatex`.

### Requirements:
- **macOS**: `brew install basictex`
- **Ubuntu**: `apt-get install texlive-latex-base texlive-latex-extra`
- **Windows**: Download [MiKTeX](https://miktex.org/)

### Error Handling:
- If `pdflatex` not found: Prints installation instructions, returns .tex only
- If compilation fails: Prints error message, returns .tex only
- If successful: Returns both .tex and .pdf paths, cleans up .aux/.log files

## Example Output

### LaTeX Diagram Structure:
```
┌────────────────────────────────────────────┐
│                                            │
│  εX        X ──── m₀(Z) ──── ┌──────────┐ │
│   │     R²=.143            │  PC1     │  │
│   │                         │ (.12 .39)│  │
│   │                         │  PC2     │  │
│   │                         │(-.24-.17)│  │
│   │                         │   ...    │  │
│   │                         └──────────┘  │
│   │                         │            │
│   └──────────────────────────┼───────────┤
│       0.412 (70.4%)         │           │
│                             g₀₁(Z) ─── AI│
│                              │      R²=.248│
│                              │           │
│                             g₀₂(Z) ─── HU│
│                                     R²=.243│
│                                            │
│         0.615 (81.9%)                      │
│        (HU ↔ AI residual corr)            │
└────────────────────────────────────────────┘
```

## Generated Files

### For trait "Extraversion":
- `haam_diagram_extraversion.tex` - Standalone LaTeX document
- `haam_diagram_extraversion.pdf` - Compiled PDF (if `render_pdf=True`)

### File Structure:
```
latex_output/
├── haam_diagram_extraversion.tex
├── haam_diagram_extraversion.pdf
├── haam_diagram_agreeableness.tex
├── haam_diagram_agreeableness.pdf
└── ...
```

## Integration with Existing Workflows

### Option 1: Generate after model fitting
```python
# Fit HAAM model
haam_model = HAAM(criterion=X, ai_judgment=AI, human_judgment=HU, ...)
haam_model.run_full_analysis()

# Generate LaTeX diagram
haam_model.create_latex_diagram(trait_name="MyTrait")
```

### Option 2: Generate from saved models
```python
# Load saved model
haam_model = HAAM.load("models/Extraversion_20251002_030543/haam_model.pkl")

# Generate LaTeX diagram
haam_model.create_latex_diagram(trait_name="Extraversion")
```

### Option 3: Batch generate for all traits
```python
import glob

for model_path in glob.glob("models/*/haam_model.pkl"):
    trait_name = model_path.split('/')[1].split('_')[0]
    haam_model = HAAM.load(model_path)
    haam_model.create_latex_diagram(trait_name=trait_name, output_dir="latex_output")
```

## Customization

### Show more/fewer PCs:
```python
# Show top 20 PCs instead of 15
haam_model.create_latex_diagram(trait_name="Extraversion", n_pcs=20)
```

### Adjust coefficient threshold:
```python
# Show "--" for |coef| < 0.10 (stricter)
haam_model.create_latex_diagram(trait_name="Extraversion", coef_threshold=0.10)
```

### LaTeX-only (no PDF):
```python
# Generate .tex file only, skip PDF compilation
haam_model.create_latex_diagram(trait_name="Extraversion", render_pdf=False)
```

## Modifying the LaTeX

The generated `.tex` file is a complete standalone document. You can:

1. **Edit numbers manually** (though auto-generated is recommended)
2. **Adjust TikZ positioning** (coordinates in the diagram)
3. **Change fonts/colors** (modify TikZ styles)
4. **Add custom annotations** (using TikZ commands)
5. **Integrate into larger documents** (copy the `\begin{figure}...\end{figure}` block)

## Technical Details

### Methods Added to `HAAMVisualizer`:

1. **`create_latex_diagram()`** - Main public method
2. **`_get_ranked_pcs_trisum()`** - Ranks PCs by sum of absolute coefficients
3. **`_format_coef()`** - Formats coefficients as ".XX" or "--"
4. **`_generate_latex_tikz()`** - Generates complete LaTeX/TikZ code
5. **`_render_latex_to_pdf()`** - Compiles LaTeX to PDF with error handling

### Dependencies:
- `numpy` - Array operations
- `subprocess` - For pdflatex execution (optional)
- `shutil` - For checking pdflatex availability (optional)

No additional Python packages required!

## Troubleshooting

### Issue: PDF not rendering
**Solution**: Check if pdflatex is installed:
```bash
which pdflatex  # macOS/Linux
where pdflatex  # Windows
```

### Issue: Coefficients showing as "--"
**Solution**: Lower the `coef_threshold`:
```python
haam_model.create_latex_diagram(coef_threshold=0.01)
```

### Issue: Not enough PCs showing
**Solution**: Increase `n_pcs`:
```python
haam_model.create_latex_diagram(n_pcs=20)
```

### Issue: LaTeX compilation errors
**Solution**:
1. Check if all required packages are installed
2. Manually compile to see detailed error:
   ```bash
   cd latex_output
   pdflatex haam_diagram_extraversion.tex
   ```

## Comparison with Existing Visualizations

| Feature | HTML Diagram | LaTeX Diagram |
|---------|-------------|---------------|
| Interactive | ✅ | ❌ |
| Publication-ready | ❌ | ✅ |
| Editable | Limited | Full control |
| Format | HTML/SVG | LaTeX/TikZ |
| File size | ~500KB | ~5KB |
| Rendering | Browser | pdflatex |
| Use case | Exploration | Papers/presentations |

## Examples

See `example_latex_generation.py` for complete working examples:
- Generate for single trait
- Batch generate for all Big Five traits
- Custom settings

## Citation

When using these diagrams in publications, please cite the HAAM package:

```
Li, R. (2025). HAAM: Human-AI Alignment Model.
https://github.com/raymondli-me/haam
```

## Updates

**Version**: 1.0 (2025-11-05)
- Initial release of LaTeX diagram generation
- Tri-mode PC ranking by sum of absolute coefficients
- Automatic PDF rendering with cross-platform error handling

## Support

For issues or questions:
- GitHub: https://github.com/raymondli-me/haam/issues
- Documentation: https://raymondli-me.github.io/haam/
