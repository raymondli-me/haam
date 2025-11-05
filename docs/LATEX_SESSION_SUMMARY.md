# LaTeX Generation Session Summary - 2025-11-05

## Session Overview

Fixed **10 major bugs** in HAAM LaTeX table generation, progressing from completely broken output to publication-ready APA 7th edition formatted tables and figures.

---

## Bugs Fixed (In Order)

### Bug 1: Methods Outside Class Boundary
**Commit:** `e40a810`
- **Problem:** LaTeX methods were module-level functions (unindented at line 2411+)
- **Discovery:** Python AST parser showed class ending at line 2409
- **Fix:** Deleted 440 lines of duplicate unindented code
- **Impact:** Made all 7 LaTeX methods accessible

### Bug 2: Parameter Order Mismatch
**Commits:** `4d126a5`, `9042b30`
- **Problem:** Wrapper methods had different parameter order than visualizer methods
- **Symptom:** `TypeError: expected str, bytes or os.PathLike object, not int`
- **Fix:** Reordered wrapper parameters to match visualizer signature
- **Impact:** Fixed 2 methods (diagram + comprehensive table)

### Bug 3: Wrong Dictionary Key ('selected_indices' vs 'selected')
**Commit:** `1209783`
- **Problem:** Looking for `'selected_indices'` but actual key is `'selected'`
- **Symptom:** Empty PC list → `{'tex_path': None}` → crash
- **Fix:** Changed all 3 occurrences to correct key name
- **Impact:** Comprehensive PC table started working

### Bug 4: Comprehensive PC Table All Dashes
**Commit:** `f527149`
- **Problem:** Looking for `'se'` but actual key is `'ses_std'`
- **Symptom:** Empty SE arrays → all PCs show `-- & -- & -- & --`
- **Fix:** Changed lines 3590-3592 to use `'ses_std'`
- **Impact:** Table now shows actual coefficients, SEs, t-values, CIs

### Bug 5: Table 3 Zero In-sample R²
**Commit:** `84820cc`
- **Problem:** Looking for `'r2'` but actual key is `'r2_insample'`
- **Symptom:** All in-sample R² showing 0.000
- **Fix:** Changed to `'r2_insample'`, label to "In-sample"
- **Impact:** Table 3 now shows real in-sample fit statistics

### Bug 6: Table 4 LaTeX Formatting
**Commit:** `8862e61`
- **Problem 1:** `\betacheck` not valid LaTeX command
- **Problem 2:** `{<}0.001` showing as upside-down exclamation mark
- **Fix:** Changed to `\check{\beta}` and `$<$0.001`
- **Impact:** Proper rendering of checked beta and p-values

### Bug 7: TikZ Diagram Formatting
**Commit:** `1e24d95`
- **Problem 1:** `\vdots` (vertical dots) in PC list
- **Problem 2:** Asterisks not superscripted: `$r=0.201***$`
- **Fix:** Changed to `\dots` and `$r=0.201^{***}$`
- **Impact:** Better aesthetics and proper superscripting

### Bug 8: Non-APA Formatting
**Commit:** `d396788`
- **Problem 1:** Colon separators ("Table 1:" instead of "Table 1.")
- **Problem 2:** Table note above instead of below
- **Problem 3:** Centered instead of left-aligned
- **Fix:** Added `\captionsetup{labelsep=period, justification=raggedright}`, moved notes, added `\raggedright`, `@{}` column specs
- **Impact:** Full APA 7th edition compliance

### Bug 9: Table 4 Incorrect P-Values (1.000)
**Commit:** `3f5dc2d`
- **Problem:** Used stored p-value from dictionary (wrong/missing → defaults to 1.0)
- **Symptom:** t=4.85 but p=1.000 (should be <0.001)
- **Fix:** Calculate from t-statistic: `p = 2 * (1 - stats.t.cdf(abs(t), df=n-2))`
- **Impact:** Correct p-values for all effects

### Bug 10: Comprehensive PC Table Not Left-Aligned
**Commit:** `3f5dc2d`, `4e69826`
- **Problem:** Caption left-aligned but table block centered
- **Fix:** Wrapped in `\begin{flushleft}`, added `\LTleft=0pt`, changed margin to 1in
- **Impact:** Full left-alignment per APA

---

## Critical Dictionary Key Reference

**Most common mistakes:**

| Feature | WRONG Key | CORRECT Key |
|---------|-----------|-------------|
| Selected PC indices | `'selected_indices'` | `'selected'` |
| Standard errors | `'se'` | `'ses_std'` |
| In-sample R² | `'r2'` | `'r2_insample'` |
| AI-HU residual corr | `'c_hu_ai'` | `'c_ai_hu'` |

---

## APA 7th Edition Formatting Checklist

✅ **Caption separators:** Period not colon (`Table 1.` not `Table 1:`)
✅ **Left-alignment:** All content (caption, table, note)
✅ **Table notes:** Below table, not above
✅ **Margins:** 1 inch all sides
✅ **longtable:** Use `\LTleft=0pt` for left alignment
✅ **Column specs:** Use `@{}` for flush left/right edges
✅ **Math mode:** `$<$` for less-than signs, `^{***}` for superscripts

---

## LaTeX Package Requirements

### All Tables
```latex
\usepackage{booktabs}       % Professional table rules
\usepackage{caption}        % Caption formatting
```

### Comprehensive PC Table (longtable)
```latex
\usepackage{longtable}      % Multi-page tables
\usepackage{multirow}       % Spanning rows
\usepackage{threeparttable} % Table notes
```

### TikZ Diagram
```latex
\usepackage{tikz}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{threeparttable}
\usepackage{float}
```

### APA Formatting
```latex
\usepackage[margin=1in]{geometry}
\captionsetup{labelsep=period, justification=raggedright, singlelinecheck=false}
```

---

## Key LaTeX Commands Learned

### For APA Left-Alignment
```latex
% Caption text left-aligned
\captionsetup{labelsep=period, justification=raggedright, singlelinecheck=false}

% Table block left-aligned
\begin{flushleft}
\LTleft=0pt  % For longtable specifically
\begin{longtable}{@{}llrrrr@{}}
...
\end{longtable}
\end{flushleft}

% Figure left-aligned
\begin{figure}[H]
\raggedright
...
\end{figure}
```

### For Proper Math/Symbol Rendering
```latex
% Less-than sign (not upside-down !)
\pval{$<$0.001}

% Checked beta
\check{\beta}

% Superscripted asterisks
$r=0.201^{***}$

% Horizontal dots (not vertical)
\dots
```

---

## Session Statistics

- **Total bugs fixed:** 10
- **Files modified:** 2 (`haam_init.py`, `haam_visualizations.py`)
- **Documentation created:** 1 (`LATEX_GENERATION_GUIDE.md`, 1100+ lines)
- **Git commits:** 20
- **Lines changed:** ~500
- **Tables working:** 7/7 ✅
- **APA compliance:** 100% ✅

---

## Next Goal: Word Cloud Integration

### Objective
Create a new LaTeX table that combines the comprehensive PC coefficients table with word cloud visualizations for each PC.

### Requirements

**Table Structure:**
```
PC | Model | β | SE | t | 95% CI | Word Cloud (Low) | Word Cloud (High)
---------------------------------------------------------------------------
PC5 | Validity | ... | ... | ... | [...] | [image: low pole] | [image: high pole]
    | AI       | ... | ... | ... | [...] |                  |
    | Human    | ... | ... | ... | [...] |                  |
```

**Technical Challenges:**

1. **Image Inclusion:**
   - Use `\includegraphics` with appropriate sizing
   - Images must be resized (mini versions) to fit in table
   - Need `graphicx` package

2. **Dependency Management:**
   - Word clouds must be generated FIRST
   - Method should gracefully handle missing images
   - Check if files exist before including
   - Provide fallback (placeholder or "N/A") if missing

3. **Table Layout:**
   - Multi-column spanning for images (or separate columns)
   - Maintain readability with images
   - Keep APA left-alignment
   - May need `\multirow` for vertical alignment

4. **Image Paths:**
   - Word clouds saved in `wordclouds/` subdirectory
   - Naming: `pc{X}_wordcloud_{pole}.png` (X = 1-indexed)
   - Relative paths in LaTeX

**Implementation Strategy:**

```python
def create_table_pc_coefficients_with_wordclouds(
    self,
    trait_name: str = "Trait",
    output_dir: str = "./",
    wordcloud_dir: str = "./wordclouds",
    min_trisum: float = 0.0,
    image_width: str = "0.15\\textwidth",
    display: bool = True
) -> Dict[str, str]:
    """
    Generate comprehensive PC table with word cloud images.

    Parameters
    ----------
    wordcloud_dir : str
        Directory containing word cloud images
    image_width : str
        LaTeX width specification for images

    Notes
    -----
    - Checks if word cloud images exist before including
    - Shows "N/A" placeholder if images missing
    - Requires word clouds to be generated first
    """
```

**Error Handling:**
```python
# Check if word cloud files exist
low_pole_path = os.path.join(wordcloud_dir, f"pc{pc_num}_wordcloud_low.png")
high_pole_path = os.path.join(wordcloud_dir, f"pc{pc_num}_wordcloud_high.png")

if os.path.exists(low_pole_path):
    low_img = f"\\includegraphics[width={image_width}]{{{low_pole_path}}}"
else:
    low_img = "\\textit{N/A}"
```

**LaTeX Additions Needed:**
```latex
\usepackage{graphicx}  % For \includegraphics
```

**Potential Issues to Solve:**
- Image sizing (may need trial and error)
- Table width (may exceed page width with images)
- Landscape orientation? (`\usepackage{pdflscape}`)
- Image quality/resolution
- Compilation time (many images)

### Success Criteria
✅ Table compiles without errors
✅ Handles missing word clouds gracefully
✅ Images sized appropriately
✅ Maintains APA formatting
✅ Readable with both coefficients and images
✅ Multi-page support if needed

---

## Files to Review for Next Session

1. **Word cloud generation code:** Check naming convention, output directory
2. **Current comprehensive table method:** `create_table_pc_coefficients_comprehensive()`
3. **Image path handling:** Relative vs absolute paths in LaTeX

---

## End of Session Summary

**Status:** All 7 LaTeX tables generating correctly with full APA 7th edition formatting

**Next:** Integrate word cloud images into comprehensive PC coefficients table
