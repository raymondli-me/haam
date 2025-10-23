# HAAM Tutorial

**Quick guide to using HAAM (Human AI Accuracy Model) for mediation analysis.**

## What's Included

- `data.csv` - Example dataset (German hierarchy self-descriptions)
- `01_basic_example.py` - Minimal HAAM workflow (~60 lines)
- `02_full_analysis.py` - Complete 3-construct analysis (~150 lines)
- `03_with_visualizations.py` - Full analysis + wordclouds + UMAP + topic clustering
- `run_tutorial.sh` - Run all tutorials sequentially

## Quick Start

**Requirements:** Python 3.10 - 3.13 (Python 3.14+ not yet supported)

```bash
# Clone repository (if you haven't already)
git clone https://github.com/raymondli-me/haam.git && cd haam

# Create and activate virtual environment
python3.13 -m venv venv  # Or python3, python3.12, python3.11, python3.10
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install HAAM and dependencies
pip install -e . sentence-transformers

# Run tutorials
cd tutorials/ && ./run_tutorial.sh
```

> **Note:** If you have Python 3.14, install Python 3.13 first: `brew install python@3.13`

Or run individually:
```bash
python3 01_basic_example.py       # Basic example
python3 02_full_analysis.py        # Full analysis
python3 03_with_visualizations.py  # With wordclouds, UMAP, topics
```

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
