# setup.py
"""
Setup configuration for HAAM package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="haam",
    version="0.1.0",
    author="Raymond Li",
    author_email="",
    description="Human-AI Accuracy Model analysis package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raymondli-me/haam",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.12.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "sentence-transformers>=2.0.0",
        "umap-learn>=0.5.0",
        "hdbscan>=0.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)

# requirements.txt
"""
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
statsmodels>=0.12.0
scipy>=1.5.0
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0
sentence-transformers>=2.0.0
umap-learn>=0.5.0
hdbscan>=0.8.0
"""

# README.md
"""
# HAAM - Human-AI Accuracy Model Analysis Package

A lightweight Python package for analyzing human-AI accuracy models using sample-split post-lasso regression and interactive visualizations.

## Features

- **Sample-split post-lasso regression** for valid statistical inference
- **Automatic topic modeling** and PC interpretation
- **Interactive visualizations** including:
  - Main HAAM framework diagram
  - Mini coefficient grid
  - PC effects bar charts
  - 3D UMAP visualizations
- **Export functionality** for all results and visualizations
- **R compatibility** via reticulate

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/raymondli-me/haam.git

# Or install locally
git clone https://github.com/raymondli-me/haam.git
cd haam
pip install -e .
```

## Quick Start

```python
from haam import HAAM

# Load your data
criterion = [...]  # e.g., social class
ai_judgment = [...]  # AI predictions
human_judgment = [...]  # Human ratings
texts = [...]  # Optional: text data for topic modeling

# Run analysis
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=texts,
    n_components=200
)

# Export all results
haam.export_all_results('./output')
```

## Google Colab Usage

```python
!pip install git+https://github.com/yourusername/haam.git

from haam import HAAM
import pandas as pd

# Load your data
df = pd.read_csv('/content/your_data.csv')

# Run analysis
haam = HAAM(
    criterion=df['social_class'],
    ai_judgment=df['ai_rating'],
    human_judgment=df['human_rating'],
    texts=df['text'].tolist()
)

# Create visualizations
haam.create_main_visualization()
haam.create_mini_grid()
```

## R Usage (via reticulate)

```r
library(reticulate)

# Install the package
py_install("git+https://github.com/yourusername/haam.git")

# Import the package
haam <- import("haam")

# Create analysis object
analysis <- haam$HAAM(
  criterion = criterion_vector,
  ai_judgment = ai_vector,
  human_judgment = human_vector,
  texts = text_list
)

# Export results
analysis$export_all_results("./output")
```

## Core Functions

### Main Analysis Class

- `HAAM()`: Main analysis class
  - `run_full_analysis()`: Run complete analysis pipeline
  - `export_all_results()`: Export all results and visualizations
  - `create_main_visualization()`: Generate main HAAM diagram
  - `create_mini_grid()`: Generate coefficient grid visualization
  - `explore_pc_topics()`: Explore topic associations for PCs
  - `plot_pc_effects()`: Create bar charts for PC effects
  - `create_umap_visualization()`: Generate UMAP visualizations

### Helper Functions

- `HAAMAnalysis.generate_embeddings()`: Generate embeddings from text using MiniLM

## Output Structure

The package creates the following outputs:

```
output_directory/
├── haam_results_coefficients_[timestamp].csv  # PC coefficients
├── haam_results_summary_[timestamp].csv      # Model summary
├── pc_topic_exploration.csv                  # Topic associations
├── haam_main_visualization.html              # Main diagram
├── haam_mini_grid.html                       # Coefficient grid
├── haam_umap_3d_SC.html                     # UMAP colored by SC
├── haam_umap_3d_AI.html                     # UMAP colored by AI
└── haam_umap_3d_HU.html                     # UMAP colored by HU
```

## Requirements

- Python >= 3.7
- See requirements.txt for full package dependencies

## License

MIT License

## Citation

If you use this package, please cite:

```bibtex
@software{haam2024,
  title={HAAM: Human-AI Accuracy Model Analysis Package},
  author={Li, Raymond},
  year={2024},
  url={https://github.com/raymondli-me/haam}
}
```
"""

# Package structure file: __init__.py
"""
from .haam_main import HAAM, example_usage
from .haam_core import HAAMAnalysis
from .haam_topics import TopicAnalyzer
from .haam_visualizations import HAAMVisualizer

__version__ = "0.1.0"
__all__ = ["HAAM", "HAAMAnalysis", "TopicAnalyzer", "HAAMVisualizer", "example_usage"]
"""