haam.haam\_package module
=========================

Core Analysis Module
--------------------

This module implements the main ``HAAMAnalysis`` class, which performs the Double Machine Learning Lens Model Equation (DML-LME) analysis. It is the primary interface for decomposing human and AI judgment accuracy into direct and mediated components.

**Key Concepts:**

- **Direct Effects**: Accuracy not explained by measured perceptual cues (unmeasured pathways)
- **Indirect Effects**: Accuracy mediated through the high-dimensional cue space
- **PoMA (Percentage of Mediated Accuracy)**: Proportion of accuracy flowing through measured cues

The analysis follows a four-stage process:

1. **Feature Extraction**: Convert raw inputs (text, embeddings) into principal components
2. **Nuisance Estimation**: Use ML to estimate conditional expectations with cross-fitting
3. **Orthogonalization**: Remove regularization bias via double ML
4. **Inference**: Bootstrap confidence intervals for all estimates

This implementation handles the high-dimensional setting (p >> n) that breaks traditional mediation analysis.

.. automodule:: haam.haam_package
   :members:
   :undoc-members:
   :show-inheritance:
