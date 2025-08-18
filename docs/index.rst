.. HAAM documentation master file

HAAM: Human-AI Accuracy Model
========================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
   :alt: License

**HAAM** (Human-AI Accuracy Model) implements the Double Machine Learning Lens Model Equation (DML-LME) to analyze and compare how humans and artificial intelligence systems achieve perceptual accuracy in high-dimensional judgment tasks.

This package provides a rigorous statistical framework for decomposing judgment accuracy into direct and mediated components, enabling researchers to understand the different pathways through which humans and AI systems process information to make accurate decisions.

Key Innovation
--------------

HAAM addresses a fundamental question in AI evaluation: When both humans and AI achieve similar accuracy levels, are they using the same perceptual cues and cognitive strategies? By implementing the DML-LME framework, HAAM can:

* Quantify the **Percentage of Mediated Accuracy (PoMA)** for any perceiver
* Decompose accuracy into direct and cue-mediated pathways
* Handle high-dimensional perceptual features using machine learning
* Provide statistical inference with bootstrap confidence intervals

Core Features
-------------

* **Double Machine Learning**: Debiased estimation through cross-fitting and post-lasso regression
* **Lens Model Analysis**: Decompose judgment accuracy using Brunswikian theory
* **Human-AI Comparison**: Statistical framework for comparing perceptual strategies
* **Rich Visualizations**: 3D UMAP projections, principal component analysis, and interpretable word clouds
* **Topic Modeling**: Automatic discovery of content themes in judgment tasks
* **Comprehensive Metrics**: Correlations, regression coefficients, cross-validated R², and PoMA calculations

Quick Example
-------------

.. code-block:: python

   from haam import HAAMAnalysis
   
   # Compare human and AI perceptual accuracy
   analysis = HAAMAnalysis(
       criterion=ground_truth_labels,        # Environmental criterion
       human_judgment=human_predictions,     # Human judgments
       ai_judgment=ai_model_outputs,        # AI predictions
       perceptual_cues=feature_matrix       # High-dimensional cues
   )
   
   # Calculate Percentage of Mediated Accuracy
   human_poma, ai_poma = analysis.calculate_poma()
   print(f"Human PoMA: {human_poma:.1%}")  # e.g., 73%
   print(f"AI PoMA: {ai_poma:.1%}")        # e.g., 91%
   
   # Generate comprehensive analysis report
   analysis.display()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Theory & Methods
   
   theoretical_framework
   methodology
   interpretation_guide

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   guide/data_preparation
   guide/running_analysis
   guide/visualizations
   guide/advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   
   citation
   changelog
   contributing
   faq

Research Applications
---------------------

HAAM has been designed for researchers studying:

* **AI Alignment**: Understanding how AI systems differ from humans in their use of information
* **Explainable AI**: Decomposing black-box predictions into interpretable pathways
* **Cognitive Science**: Comparing human and machine perception strategies
* **Social Science**: Analyzing judgment and decision-making in high-dimensional contexts

The method is particularly powerful when you have:
- Ground truth labels (criterion variable)
- Human judgments or ratings
- AI model predictions
- Access to the features/cues used (text, images, or extracted features)

Theoretical Foundation
----------------------

HAAM builds on three theoretical pillars:

1. **Brunswik's Lens Model** (1952): Environmental cues mediate between distal criteria and perceptual judgments
2. **Classical Mediation Analysis**: Decomposing total effects into direct and indirect pathways
3. **Double Machine Learning** (Chernozhukov et al., 2018): Debiased causal inference with ML

For a complete theoretical treatment, see :doc:`theoretical_framework`.

Getting Help
------------

* **Issues**: Report bugs or request features on `GitHub Issues <https://github.com/raymondli-me/haam/issues>`_
* **Discussions**: Join the community on `GitHub Discussions <https://github.com/raymondli-me/haam/discussions>`_
* **Email**: Contact the maintainers at raymond.li@psych.ubc.ca

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`