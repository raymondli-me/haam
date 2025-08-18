# HAAM: Human-AI Accuracy Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/haam/badge/?version=latest)](https://haam.readthedocs.io/en/latest/?badge=latest)

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
pip install haam
```

### Development Install
```bash
git clone https://github.com/raymondli-me/haam.git
cd haam
pip install -e .
```

## Quick Start

```python
from haam import HAAMAnalysis

# Prepare your data
criterion = [...]           # Ground truth labels (environmental criterion)
human_judgment = [...]      # Human predictions/ratings  
ai_judgment = [...]         # AI model outputs
perceptual_cues = [...]     # High-dimensional features (or raw text)

# Run analysis
analysis = HAAMAnalysis(
    criterion=criterion,
    human_judgment=human_judgment,
    ai_judgment=ai_judgment,
    perceptual_cues=perceptual_cues
)

# Calculate Percentage of Mediated Accuracy
results = analysis.calculate_poma()
print(f"Human PoMA: {results['human_poma']:.1%}")  # e.g., 73%
print(f"AI PoMA: {results['ai_poma']:.1%}")        # e.g., 91%

# Generate comprehensive report
analysis.display()
```

## Understanding the Output

HAAM reveals how perceivers achieve accuracy:

- **High PoMA (>80%)**: Perceiver heavily relies on measured cue space
- **Moderate PoMA (40-80%)**: Balanced use of measured and unmeasured information  
- **Low PoMA (<40%)**: Perceiver uses information beyond measured cues

Typically, AI shows higher PoMA (more cue-dependent) while humans show lower PoMA (more contextual processing).

## Core Methods

### The DML-LME Framework

```python
# Initialize with theoretical grounding
analysis = HAAMAnalysis(
    criterion=environmental_truth,      # Y_e: What is being judged
    human_judgment=human_predictions,   # Y_h: Human ratings
    ai_judgment=ai_outputs,            # Y_ai: AI predictions
    perceptual_cues=feature_matrix,    # X: High-dimensional mediators
    
    # Advanced options
    ml_method='random_forest',         # Nuisance function estimator
    n_folds=5,                        # Cross-fitting folds
    n_bootstrap=1000,                 # Bootstrap iterations
    confidence_level=0.95             # CI coverage
)

# Get detailed decomposition
decomposition = analysis.get_accuracy_decomposition()
print(f"Total Effect: {decomposition['total_effect']:.3f}")
print(f"Direct Effect: {decomposition['direct_effect']:.3f}")  
print(f"Indirect Effect: {decomposition['indirect_effect']:.3f}")
```

### Visualization Suite

```python
# 3D UMAP with PC arrows showing perceptual dimensions
analysis.create_3d_umap_with_pc_arrows(top_pcs=5)

# Word clouds for each principal component
analysis.create_pc_wordclouds(pc_indices=[0, 1, 2, 3, 4])

# Framework diagram showing mediation pathways
analysis.create_main_visualization()

# Comprehensive grid of all 200 PCs
analysis.create_mini_grid()
```

### Statistical Analysis

```python
# Full statistical report
stats = analysis.get_statistical_summary()

# Includes:
# - Correlations (achievement accuracy)
# - Cross-validated R² for all models
# - Post-lasso regression coefficients
# - PoMA (Percentage of Mediated Accuracy)
# - Lens model statistics (G, C, RX, RY)
```

## Advanced Usage

### Working with Text Data

```python
# HAAM automatically extracts features from text
analysis = HAAMAnalysis(
    criterion=labels,
    human_judgment=human_ratings,
    ai_judgment=gpt_outputs,
    texts=documents,  # Raw text input
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
)

# Access extracted topics
topics = analysis.get_topic_labels()
```

### Custom Principal Component Analysis

```python
# Name your PCs for better interpretability
pc_names = {
    0: "Formality",
    1: "Complexity",  
    2: "Emotional Tone",
    3: "Technicality"
}

analysis.create_main_visualization(pc_names=pc_names)
```

### Exporting Results

```python
# Export comprehensive results
analysis.export_all_results('./output/')

# Creates:
# - coefficients.csv (PC loadings)
# - metrics_summary.json (all statistics)
# - visualizations/ (all plots)
# - topic_analysis/ (word clouds, topic labels)
```

## Theoretical Background

HAAM implements the framework that is described in an upcoming research paper, the High-Dimensional Perception with the Double Machine Learning Lens Model Equation. 

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

Full documentation available at: [https://haam.readthedocs.io](https://haam.readthedocs.io)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
