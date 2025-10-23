# HAAM Tutorial

**Quick guide to using HAAM (Human AI Accuracy Model) for mediation analysis.**

## What's Included

- `data.csv` - Example dataset (German hierarchy self-descriptions)
- `01_basic_example.py` - Minimal HAAM workflow (~60 lines)
- `02_full_analysis.py` - Complete 3-construct analysis (~150 lines)
- `03_with_visualizations.py` - Full analysis + wordclouds + UMAP + topic clustering
- `run_tutorial.sh` - Run all tutorials sequentially

## Quick Start

### Requirements

⚠️ **Python 3.10 - 3.13 ONLY** (Python 3.14+ not supported due to numba)

### Setup Instructions

**IMPORTANT:** You MUST use a virtual environment on macOS Sonoma+ to avoid `externally-managed-environment` errors.

```bash
# 1. Clone repository (if you haven't already)
git clone https://github.com/raymondli-me/haam.git && cd haam

# 2. Check your Python version (must be 3.10-3.13)
python3 --version

# 3. Create virtual environment
#    If you have Python 3.14, use python3.13 instead (see below)
python3 -m venv venv

# 4. Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
#    You should see (venv) prefix in your terminal

# 5. Install HAAM and dependencies (use pip OR pip3, both work in venv)
pip install -e . sentence-transformers

# 6. Run tutorials
cd tutorials/ && ./run_tutorial.sh
```

### If You Have Python 3.14

HAAM doesn't work with Python 3.14 yet. Install Python 3.13:

```bash
# macOS
brew install python@3.13

# Then use python3.13 instead of python3
python3.13 -m venv venv
source venv/bin/activate
pip install -e . sentence-transformers
cd tutorials/ && ./run_tutorial.sh
```

### Running Individual Scripts

After setup, run tutorials individually:
```bash
# Make sure venv is activated first! (you should see (venv) prefix)
python3 01_basic_example.py       # Basic example (Prestige only)
python3 02_full_analysis.py        # Full analysis (all 3 constructs)
python3 03_with_visualizations.py  # With wordclouds, UMAP, mediation diagrams
```

### Troubleshooting

<details>
<summary><b>❌ "Cannot install on Python version 3.14"</b></summary>

HAAM requires Python 3.10-3.13 due to the numba dependency. Install Python 3.13:
```bash
brew install python@3.13
python3.13 -m venv venv
source venv/bin/activate
pip install -e . sentence-transformers
```
</details>

<details>
<summary><b>❌ "externally-managed-environment"</b></summary>

This means you're trying to install without a virtual environment. macOS Sonoma+ requires venv:
```bash
# Create venv
python3 -m venv venv

# Activate it (you'll see (venv) prefix appear)
source venv/bin/activate

# Now install
pip install -e . sentence-transformers
```
</details>

<details>
<summary><b>❌ "command not found: pip"</b></summary>

Two possible causes:
1. **You haven't activated the venv** - Run `source venv/bin/activate` first
2. **Use pip3 instead** - Some systems only have `pip3`, not `pip`

After activating venv, both `pip` and `pip3` work.
</details>

<details>
<summary><b>❌ "ModuleNotFoundError: No module named 'haam'"</b></summary>

Make sure you:
1. Ran `pip install -e .` from the main haam directory (not tutorials/)
2. Activated the venv before running scripts
3. Are using the same Python that created the venv
</details>

## Data Format

Your CSV needs these columns:

- `text` - Text data for embedding
- `self_*` - Self-report ratings (criterion)
- `judge_*` - Human judgment ratings
- `AI_*_mean` - AI judgment ratings

Example constructs: prestige, power, dominance

## HAAM Workflow (5 Steps)

1. **Load data** - Read CSV with criterion, judgments, and text
2. **Generate embeddings** - Use sentence-transformers model
3. **Extract variables** - Self-report (X), Human (HU), AI (AI)
4. **Run HAAM** - Initialize with auto_run=True
5. **Access results** - Get correlations, R², effects from `haam.analysis.results`

## Key Parameters

```python
HAAM(
    criterion=X,              # Self-report (validity criterion)
    human_judgment=Y_HU,      # Human ratings
    ai_judgment=Y_AI,         # AI ratings
    embeddings=embeddings,    # From sentence-transformers
    texts=texts,              # Raw text (optional)
    n_components=50,          # Number of PCs to extract
    standardize=True,         # Standardize variables
    auto_run=True            # Run analysis immediately
)
```

## What HAAM Does

- **PCA** on embeddings → linguistic features
- **LASSO** for feature selection
- **DML** (Double Machine Learning) for debiased direct effects
- **Mediation analysis** - Calculates total, direct, indirect effects
- **PoMA** (Proportion of Mediated Accuracy) - How much language mediates judgment

## Results Structure

```python
results = haam.analysis.results

# Correlations
results['total_effects']            # Total effects (β)
results['total_effects'][path]['check_beta']  # DML direct effects (β̌)

# Model fit
results['debiased_lasso']['X']['n_selected']  # PCs selected
results['debiased_lasso']['X']['r2_cv']       # Cross-validated R²

# Mediation
results['mediation_analysis']       # Indirect effects
results['policy_similarities']      # G (policy similarity)
results['residual_correlations']    # C (residual correlation)
```

## Example Output

```
PRESTIGE - KEY RESULTS
==============================================================
Validity--Human:  r = 0.523
Validity--AI:     r = 0.492

PCs selected: Validity=12/50, AI=8/50, Human=10/50

Cross-validated R²: 0.385

✓ Analysis complete!
```

## Reproducibility

- Set `np.random.seed(42)` for consistent results
- Use same embedding model across runs
- LASSO selection may vary slightly across CV folds

## Visualizations (Script 03)

**03_with_visualizations.py** generates the same outputs as the Brunswik Newsletter analysis:

- **Mediation path diagram** - Interactive HTML showing X → PCs → AI/HU with PoMA values
- **Word cloud strips** - 100 PNG files (50 PCs × 2 ends: high/low)
- **3D UMAP** - Interactive HTML with PCA arrows and topic clustering
- **PC table** - Comprehensive analysis table (PNG)
- **Analysis report** - Text summary of validity analysis

Settings match the paper:
- 50 principal components
- 5 topics per word cloud
- 100 words per cloud
- UMAP arrow clustering (k=1)
- HDBSCAN clustering (min_cluster_size=3, min_samples=1)

## Next Steps

1. **Modify for your data** - Update column names in scripts
2. **Change construct** - Swap prestige → your variable
3. **Adjust parameters** - Try different `n_components`, models
4. **Generate visualizations** - Run `03_with_visualizations.py`

## Citation

If you use HAAM, please cite:
- Li, R., & Biesanz, J. C. (2025). High-Dimensional Perception with the Double Machine Learning Lens Model. *PsyArXiv*. https://osf.io/preprints/psyarxiv/ubwgk
- GitHub: https://github.com/raymondli-me/haam

## Questions?

See main HAAM documentation or open an issue on GitHub.
