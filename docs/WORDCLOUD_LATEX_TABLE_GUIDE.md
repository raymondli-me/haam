# Word Cloud + Coefficients LaTeX Table Guide

**Created:** 2025-11-05
**Feature:** `create_table_pc_coefficients_with_wordclouds()`
**Status:** ✅ Production-ready

---

## Overview

This guide documents the creation of a portrait-oriented LaTeX table that combines PC coefficients (β, SE, t, 95% CI) with embedded word cloud thumbnail images for low/high poles.

**Final Result:**
- **Portrait orientation** (fits standard page)
- **8 columns:** PC | Model | β | SE | t | 95% CI | Low Pole | High Pole
- **Thumbnail images:** 0.6in height (perfect balance)
- **APA 7th edition formatting**
- **Graceful degradation:** Shows "N/A" for missing images

---

## Implementation Journey

### Iterations & Lessons Learned

#### 1. **Initial Attempt: Landscape with 1.2in images**
**Problem:** Images dominated the table, overwhelming the coefficient data
**Lesson:** Word clouds should complement, not replace, the statistical content

#### 2. **Refinement: Portrait with 0.6in images**
**Solution:** Reduced image height by 50% (1.2in → 0.6in)
**Result:** Perfect thumbnail size - coefficients readable, word clouds visible

#### 3. **Spacing Issues**
**Problem:** Bottom rule touched PC7 word clouds
**Solution:** Added `\addlinespace[0.4em]` before `\end{longtable}`

#### 4. **Note Width Mismatch**
**Problem:** Note text narrower than table (constrained by `\LTcapwidth`)
**Solution:** Removed `minipage` wrapper, let note flow naturally in `flushleft`

---

## Critical LaTeX Quirks

### ✅ What Works

```latex
% Portrait orientation (no landscape needed)
\begin{flushleft}
\LTleft=0pt  % Force left alignment

% Simple longtable (no ThreePartTable wrapper)
\begin{longtable}{@{}llrrrrcc@{}}

% Multi-row images spanning 3 rows
\multirow{3}{*}{\includegraphics[height=0.6in]{wordclouds/pc1_low_wordcloud.png}}

% Spacing before bottom rule
\addlinespace[0.4em]
\end{longtable}

% Note without minipage
\vspace{0.5em}
\scriptsize
\textit{Note.} [Your note text here...]
\end{flushleft}
```

### ❌ What Doesn't Work

1. **ThreePartTable environment**
   - Caused undefined environment errors
   - Not needed for simple notes
   - Solution: Use plain text after longtable

2. **`\insertTableNotes` in `\endlastfoot`**
   - Only works with ThreePartTable
   - Solution: Put note after `\end{longtable}`

3. **Minipage for note**
   - `\begin{minipage}{\LTcapwidth}` makes note too narrow
   - Solution: Let text flow naturally within `flushleft`

4. **Landscape for narrow tables**
   - Unnecessary for 8-column table with small images
   - Solution: Portrait works perfectly

---

## Image Path Handling

### For Overleaf Workflow

**Expected usage pattern:**
1. Generate word clouds locally → `output_dir/wordclouds/pc{N}_{pole}_wordcloud.png`
2. Generate LaTeX table → `output_dir/table_pc_coefficients_wordclouds_{trait}.tex`
3. **Upload to Overleaf:**
   - Drag-and-drop LaTeX file to project root
   - Drag-and-drop `wordclouds/` folder to project root

**LaTeX path syntax:**
```latex
% ✅ CORRECT: Relative path from LaTeX file location
\includegraphics[height=0.6in]{wordclouds/pc1_low_wordcloud.png}

% ❌ WRONG: Absolute paths won't work in Overleaf
\includegraphics[height=0.6in]{/Users/.../wordclouds/pc1_low_wordcloud.png}
```

**Current implementation:**
- Uses `os.path.relpath()` to generate relative paths
- Assumes LaTeX file and `wordclouds/` folder at same level
- **TODO for Colab integration:** Ensure paths are relative, not absolute

---

## Package Requirements

### Minimal Set (Portrait)

```latex
\usepackage{booktabs}       % Professional table rules
\usepackage{longtable}      % Multi-page tables
\usepackage{multirow}       % Spanning rows
\usepackage{array}          % Column formatting
\usepackage{caption}        % Caption formatting
\usepackage{graphicx}       % Image inclusion
\usepackage[margin=1in]{geometry}
```

### Removed (Not Needed)

```latex
% ❌ Removed - not needed for portrait
\usepackage{pdflscape}

% ❌ Removed - caused errors
\usepackage{threeparttable}
```

---

## Parameter Tuning Guide

### Image Height

**Tested values:**
- `1.2in` → Too large, dominates table
- `0.6in` → ✅ **Perfect balance** (default)
- `0.5in` → Too small, hard to read word clouds

**Recommendation:** Start with `0.6in`, adjust ±0.1in if needed

### Column Spacing

**Between PCs:**
```latex
\addlinespace[0.6em]  % Standard spacing between PC groups
```

**Before bottom rule:**
```latex
\addlinespace[0.4em]  % Prevents word cloud overlap
```

### Note Formatting

```latex
\vspace{0.5em}        % Space between table and note
\scriptsize           % Small font size (APA standard)
```

---

## Error Handling

### Missing Images

**Behavior:**
- Checks `os.path.exists()` before including image
- Shows `\textit{N/A}` if file missing
- Continues table generation (no crash)
- Reports count of missing images in console

**Code pattern:**
```python
if os.path.exists(low_pole_path):
    low_img = f"\\includegraphics[height={image_height}]{{{rel_low_path}}}"
else:
    low_img = "\\textit{N/A}"
    missing['low'] = 1
```

### No PCs Selected

**Behavior:**
- Returns `{'tex_path': None}`
- Prints warning: "⚠ No PCs selected by LASSO"
- No LaTeX file created

---

## APA 7th Edition Compliance

### ✅ Requirements Met

1. **Caption separator:** Period not colon (`Table 1.` not `Table 1:`)
2. **Left-alignment:** All content (caption, table, note)
3. **Table note:** Below table, not above
4. **Margins:** 1 inch all sides
5. **Column spacing:** Professional (`@{}` for flush edges)
6. **Math mode:** Proper symbols (`$<$` for less-than, `^{***}` for superscripts)
7. **Font size:** `\scriptsize` for notes

### Caption Setup

```latex
\captionsetup{labelsep=period, justification=raggedright, singlelinecheck=false}
```

---

## Method Signature

```python
def create_table_pc_coefficients_with_wordclouds(
    self,
    trait_name: str = "Trait",
    output_dir: str = "./",
    wordcloud_dir: str = None,           # Default: {output_dir}/wordclouds
    min_trisum: float = 0.0,
    image_height: str = "0.6in",         # Adjust if needed
    display: bool = True
) -> Dict[str, str]:
```

**Returns:**
```python
{'tex_path': '/path/to/table_pc_coefficients_wordclouds_trait.tex'}
# or
{'tex_path': None}  # If no PCs selected
```

---

## Usage Examples

### Basic Usage

```python
# After HAAM analysis
haam = HAAM(...)
haam.run_analysis(...)

# Generate word clouds FIRST (prerequisite)
# [User must do this separately - not automatic]

# Generate table
result = haam.create_table_pc_coefficients_with_wordclouds(
    trait_name="Agreeableness",
    output_dir="./output",
    display=True
)

# Result: ./output/table_pc_coefficients_wordclouds_agreeableness.tex
```

### Custom Paths

```python
result = haam.create_table_pc_coefficients_with_wordclouds(
    trait_name="Conscientiousness",
    output_dir="./tables",
    wordcloud_dir="./my_wordclouds",  # Custom location
    image_height="0.7in"              # Slightly larger
)
```

### Filtering by Tri-sum

```python
# Only include PCs with tri-sum >= 0.5
result = haam.create_table_pc_coefficients_with_wordclouds(
    trait_name="Extraversion",
    min_trisum=0.5,
    output_dir="./output"
)
```

---

## File Structure Expected

```
output_dir/
├── table_pc_coefficients_wordclouds_trait.tex    (generated LaTeX)
└── wordclouds/                                    (prerequisite)
    ├── pc1_low_wordcloud.png
    ├── pc1_high_wordcloud.png
    ├── pc2_low_wordcloud.png
    ├── pc2_high_wordcloud.png
    └── ... (etc)
```

**For Overleaf:**
```
Overleaf Project/
├── table_pc_coefficients_wordclouds_trait.tex    (upload)
└── wordclouds/                                    (drag-and-drop folder)
    ├── pc1_low_wordcloud.png
    ├── pc1_high_wordcloud.png
    └── ...
```

---

## Compilation

### Local (pdflatex)

```bash
cd output_dir
pdflatex table_pc_coefficients_wordclouds_trait.tex
```

### Overleaf

1. **Upload files:**
   - LaTeX file to project root
   - `wordclouds/` folder to project root
2. **Click "Recompile"**
3. PDF generates automatically

**Note:** Overleaf handles relative paths correctly if folder structure matches

---

## Next Goal: Colab Integration

### TODO for Next Session

**Objective:** Modify Colab script to auto-generate this LaTeX table after word clouds

**Key considerations:**

1. **Dependency order:**
   ```python
   # Step 1: Run HAAM analysis
   haam.run_analysis(...)

   # Step 2: Generate word clouds
   # [current word cloud generation code]

   # Step 3: Generate LaTeX table (NEW)
   haam.create_table_pc_coefficients_with_wordclouds(...)
   ```

2. **Path handling:**
   - Ensure `output_dir` and `wordcloud_dir` align
   - Use **relative paths** for Overleaf compatibility
   - Test that `os.path.relpath()` works correctly in Colab environment

3. **Error handling:**
   - Check if word clouds exist before calling method
   - Warn user if images missing
   - Don't crash if some PCs lack word clouds

4. **Output messaging:**
   ```python
   print("✓ Generated LaTeX table with word clouds")
   print(f"  Upload to Overleaf: {tex_path}")
   print(f"  Also upload: wordclouds/ folder")
   ```

### Integration Pseudocode

```python
# In Colab script, after word cloud generation:

# Generate comprehensive LaTeX table with word clouds
try:
    result = haam.create_table_pc_coefficients_with_wordclouds(
        trait_name=trait_name,
        output_dir=output_dir,
        wordcloud_dir=os.path.join(output_dir, "wordclouds"),
        min_trisum=0.0,
        display=True
    )

    if result['tex_path']:
        print(f"\n{'='*60}")
        print("✓ LaTeX Table Generated")
        print(f"{'='*60}")
        print(f"File: {result['tex_path']}")
        print(f"\nTo use in Overleaf:")
        print(f"  1. Upload {os.path.basename(result['tex_path'])}")
        print(f"  2. Upload wordclouds/ folder")
        print(f"  3. Compile!")
    else:
        print("⚠ No LaTeX table generated (no PCs selected)")

except Exception as e:
    print(f"⚠ LaTeX generation failed: {e}")
    # Don't crash entire script
```

---

## Debugging Tips

### LaTeX Compilation Errors

**Common issues:**

1. **"Undefined control sequence"**
   - Check for special characters in trait_name
   - Escape underscores: `trait\_name`

2. **"File not found" for images**
   - Verify relative paths match folder structure
   - Check image filenames: `pc{N}_low_wordcloud.png` (lowercase, underscore)

3. **"Overfull \vbox" warnings**
   - Not critical - just LaTeX complaining about image height
   - Adjust `image_height` if bothered by warnings

### Python Errors

**"No PCs selected"**
- Normal if LASSO didn't select any PCs
- Check `min_trisum` threshold

**"File exists" errors**
- Use `os.makedirs(output_dir, exist_ok=True)`
- Already handled in implementation

---

## Performance Notes

- **Table generation:** < 1 second
- **PDF compilation:** ~2-5 seconds (depends on # of images)
- **File size:** ~10MB for 5 PCs with word clouds

---

## Summary of Commits

1. **`0953498`** - Initial implementation (landscape, 1.2in images)
2. **`3704bfe`** - Refined to portrait, 0.6in images
3. **`b6b225f`** - Fixed note width to match table

**Total lines added:** ~270 (implementation + wrapper)

---

## Related Documentation

- `LATEX_SESSION_SUMMARY.md` - Full bug-fixing journey (10 bugs)
- `LATEX_GENERATION_GUIDE.md` - Comprehensive LaTeX table reference
- `PC_COEFFICIENTS_TABLE_README.md` - Original table documentation

---

## Quick Reference Card

| Feature | Value |
|---------|-------|
| **Orientation** | Portrait |
| **Image height** | 0.6in (default) |
| **Columns** | 8 (PC, Model, β, SE, t, CI, Low, High) |
| **Rows per PC** | 3 (Validity, AI, Human) |
| **Missing images** | Shows "N/A" |
| **LaTeX packages** | booktabs, longtable, multirow, array, caption, graphicx |
| **APA compliant** | ✅ Yes |
| **Overleaf ready** | ✅ Yes (use relative paths) |
| **Multi-page** | ✅ Yes (longtable) |

---

**End of Guide**

*Next step: Integrate into Colab workflow after word cloud generation*
