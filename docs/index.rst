HAAM: Human-AI Accuracy Model
========================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
   :alt: License

Implementation of the Double Machine Learning Lens Model Equation (DML-LME) for analyzing perceptual accuracy in high-dimensional settings. HAAM quantifies how humans and AI systems achieve accuracy when making judgments, decomposing their decision-making processes into interpretable components.

What is HAAM?
-------------

The **Human-AI Accuracy Model** addresses a fundamental question: When humans and AI achieve similar accuracy levels, are they using the same perceptual cues and cognitive strategies? 

HAAM provides a rigorous statistical framework to:

- Decompose judgment accuracy into direct and mediated pathways
- Quantify the **Percentage of Mediated Accuracy (PoMA)** for any perceiver
- Compare how humans vs AI utilize high-dimensional perceptual features
- Handle thousands of features using debiased machine learning

Key Features
------------

- **🎯 DML-LME Implementation**: Double Machine Learning Lens Model Equation for high-dimensional perception
- **📊 PoMA Calculation**: Quantify what percentage of accuracy flows through measured perceptual cues
- **🧠 Human-AI Comparison**: Statistical framework for comparing perceptual strategies
- **📈 Rich Visualizations**: 3D UMAP projections, PCA analysis, word clouds
- **🔍 Topic Modeling**: Automatic discovery and labeling of content themes via BERTopic
- **📉 Comprehensive Metrics**: Correlations, regression coefficients, cross-validated R², PoMA decomposition

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`