# HAAM: Human-AI Accuracy Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://raymondli-me.github.io/haam/)

Implementation of the Double Machine Learning Lens Model Equation (DML-LME) for analyzing perceptual accuracy in high-dimensional settings. HAAM quantifies how humans and AI systems achieve accuracy when making judgments, decomposing their decision-making processes into interpretable components.

## What is HAAM?

The **Human-AI Accuracy Model** addresses a fundamental question: When humans and AI achieve similar accuracy levels, are they using the same perceptual cues and cognitive strategies? 

HAAM provides a rigorous statistical framework to:
- Decompose judgment accuracy into direct and mediated pathways
- Quantify the **Percentage of Mediated Accuracy (PoMA)** for any perceiver
- Compare how humans vs AI utilize high-dimensional perceptual features
- Handle thousands of features using debiased machine learning

## Key Features

- **🎯 DML-LME Implementation**: Double Machine Learning Lens Model Equation for high-dimensional perception
- **📊 PoMA Calculation**: Quantify what percentage of accuracy flows through measured perceptual cues
- **🧠 Human-AI Comparison**: Statistical framework for comparing perceptual strategies
- **📈 Rich Visualizations**: 3D UMAP projections, PCA analysis, word clouds
- **🔍 Topic Modeling**: Automatic discovery and labeling of content themes via BERTopic
- **📉 Comprehensive Metrics**: Correlations, regression coefficients, cross-validated R², PoMA decomposition

## Installation

### Quick Install
```bash
pip install haam (TBA - not available yet)
```

### Development Install
```bash
git clone https://github.com/raymondli-me/haam.git
cd haam
pip install -e .
```

## Mini-Example: Human-GPT Perception of Anger in Reddit Comments

```python
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud", "-q"])
!pip install git+https://github.com/raymondli-me/haam.git
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os
import re
from haam import HAAM
import warnings
warnings.filterwarnings('ignore')


github_url_processed = 'https://raw.githubusercontent.com/raymondli-me/haam/main/data/anger_family_with_angry_word_count.csv'

try:
    df_loaded_github = pd.read_csv(github_url_processed)
    df_filtered_github = df_loaded_github.copy() # Use a consistent variable name
    print(f"✓ Loaded data from GitHub CSV: {df_filtered_github.shape[0]} rows, {df_filtered_github.shape[1]} columns")
except Exception as e:
    print(f"Error loading data from GitHub: {e}")
    print(f"Please ensure the URL is correct: {github_url_processed}")
    df_filtered_github = None # Set to None to prevent further execution if file not found

if df_filtered_github is not None:
    # Prepare data for HAAM using the loaded data
    # Ensure column names match those in your CSV
    if 'angry_word_count' in df_filtered_github.columns and \
       'human_sum_score' in df_filtered_github.columns and \
       'gpt_sum_score' in df_filtered_github.columns and \
       'text' in df_filtered_github.columns:

        texts_github = df_filtered_github['text'].values.tolist()
        criterion_github = df_filtered_github['angry_word_count'].values.astype(float)  # X: angry word count
        human_judgment_github = df_filtered_github['human_sum_score'].values.astype(float)  # HU: human ratings
        ai_judgment_github = df_filtered_github['gpt_sum_score'].values.astype(float)  # AI: GPT ratings

        haam_github = HAAM(
            criterion=criterion_github,        # X: The ground truth variable (angry word count in this case)
            ai_judgment=ai_judgment_github,    # AI: AI system's predictions/ratings
            human_judgment=human_judgment_github, # HU: Human ratings/judgments
            texts=texts_github,                # Raw text data for topic modeling
            n_components=200,                  # Number of principal components to extract
            min_cluster_size=10,               # HDBSCAN parameter for topic clustering
            min_samples=2,                     # HDBSCAN parameter for core points
            umap_n_components=3,               # 3D UMAP for topic embedding
            standardize=True,                  # Standardize variables for DML
            sample_split_post_lasso=False,     # Use full sample for max power
            auto_run=True                      # Run analysis immediately
        )

        results_github = haam_github.create_comprehensive_pc_analysis(
            #pc_indices=specific_pcs,          # Which PCs to analyze (commented = use first 15)
            k_topics=3,                        # Number of topics per pole in word clouds
            max_words=100,                     # Maximum words per word cloud
            generate_wordclouds=True,          # Create word cloud table (like your Colab script)
            generate_3d_umap=True,             # Create 3D UMAP with PC arrows
            umap_arrow_k=1,                    # Single topic endpoints for arrows (clean visualization)
            show_data_counts=True,             # Show "HU: n=3" warnings for sparse data
            output_dir='haam_wordclouds_final_aligned_sparse_github', # Output directory
            display=True                       # Show plots in notebook/colab
            )
```

## Theoretical Background

HAAM implements the framework that is described in an upcoming research paper, _High-Dimensional Perception with the Double Machine Learning Lens Model Equation (DML-LME)._ 

The method combines:
1. **Brunswik's Lens Model** - Perceptual cues mediate between environment and judgment
2. **Mediation Analysis** - Decompose total effects into pathways  
3. **Double Machine Learning** - Debiased estimation in high dimensions

## Citation

```bibtex
@software{haam_package,
  title={HAAM: Human-AI Accuracy Model},
  author={[Raymond Li]},
  year={2025},
  version={1.0},
  url={https://github.com/raymondli-me/haam}
}
```

## Documentation

Full documentation available at: [https://raymondli-me.github.io/haam/](https://raymondli-me.github.io/haam/)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
