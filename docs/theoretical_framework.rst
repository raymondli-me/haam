Theoretical Framework
=====================

The Human-AI Accuracy Model (HAAM) implements the Double Machine Learning Lens Model Equation (DML-LME) to understand how different perceivers achieve accuracy in judgment tasks. This framework combines classical psychometric theory with modern causal machine learning.

The Lens Model
--------------

Egon Brunswik's (1952) Lens Model provides the conceptual foundation for understanding perceptual accuracy. The model posits that perceivers cannot directly access distal environmental criteria but must rely on proximal cues that mediate between the environment and perception.

.. figure:: _static/lens_model_diagram.png
   :alt: Brunswik's Lens Model
   :align: center
   
   The Lens Model: Cues mediate between environmental criteria and perceptual judgments

In high-dimensional settings (e.g., text analysis with thousands of features), the classical Lens Model faces two challenges:

1. **Dimensionality**: More cues than observations (p >> n)
2. **Selection bias**: Which cues to include in the model?

The Lens Model Equation (LME)
------------------------------

The classical LME decomposes perceptual accuracy into interpretable components:

.. math::

   r_a = G \cdot R_e \cdot R_s + C \sqrt{(1-R_e^2)(1-R_s^2)}

Where:

* :math:`r_a` = Achievement accuracy (correlation between judgment and criterion)
* :math:`G` = Matching coefficient (correlation between predicted and actual cue validities)
* :math:`R_e` = Environmental predictability (multiple correlation of cues with criterion)
* :math:`R_s` = Response consistency (multiple correlation of cues with judgment)
* :math:`C` = Unmodeled knowledge (residual accuracy)

Mediation Analysis Framework
----------------------------

To understand how perceivers achieve accuracy, we decompose the total effect of perceptual cues on accuracy into:

1. **Direct Effect**: Accuracy not explained by measured cues
2. **Indirect Effect**: Accuracy mediated through perceptual cues

The Percentage of Mediated Accuracy (PoMA) quantifies this decomposition:

.. math::

   \text{PoMA} = \frac{\text{Indirect Effect}}{\text{Total Effect}} \times 100\%

Double Machine Learning Extension
---------------------------------

Traditional mediation analysis fails in high-dimensional settings due to:

* Regularization bias from feature selection
* Post-selection inference problems
* Overfitting when p >> n

HAAM addresses these issues using Double Machine Learning (Chernozhukov et al., 2018), which provides:

1. **Orthogonalization**: Removes regularization bias
2. **Cross-fitting**: Prevents overfitting
3. **Valid inference**: Enables confidence intervals

The DML-LME Algorithm
---------------------

The core algorithm proceeds in four stages:

**Stage 1: Nuisance Function Estimation**

Using cross-fitting, estimate:

.. math::

   \begin{align}
   g_0(X) &= \mathbb{E}[Y|X] \quad \text{(Outcome model)}\\
   m_0(X) &= \mathbb{E}[D|X] \quad \text{(Treatment model)}\\
   \ell_0(X) &= \mathbb{E}[DY|X] \quad \text{(Interaction model)}
   \end{align}

Where:
- :math:`Y` = Accuracy (judgment × criterion)
- :math:`D` = Judgment
- :math:`X` = High-dimensional perceptual cues

**Stage 2: Orthogonalization**

Compute orthogonalized scores:

.. math::

   \psi(W; \theta, \eta) = (Y - \ell_0(X)) - \theta(D - m_0(X))

**Stage 3: Debiased Estimation**

Estimate the parameter:

.. math::

   \hat{\theta} = \frac{\mathbb{E}_n[\psi]}{\mathbb{E}_n[D - m_0(X)]}

**Stage 4: Inference**

Bootstrap confidence intervals accounting for the two-stage procedure.

Interpretation
--------------

The DML-LME provides several key insights:

1. **Total Accuracy**: How well does the perceiver predict the criterion?
2. **Direct Pathway**: Accuracy from unmeasured processes or cues
3. **Mediated Pathway**: Accuracy flowing through measured perceptual features
4. **PoMA**: What percentage of accuracy is explained by the measured cue space?

High PoMA (e.g., 90%) indicates the perceiver heavily relies on the measured features. Low PoMA (e.g., 30%) suggests the perceiver uses information beyond the measured cue space.

Comparing Human and AI Systems
------------------------------

HAAM's key innovation is enabling fair comparison between humans and AI by:

1. **Common Cue Space**: Both perceivers evaluated on same features
2. **Debiased Estimates**: ML doesn't favor either perceiver
3. **Statistical Testing**: Bootstrap tests for significant differences

Example interpretation:
- Human PoMA = 73%: Humans moderately rely on measured cues
- AI PoMA = 91%: AI heavily relies on measured cues
- Difference suggests humans integrate unmeasured contextual information

Mathematical Properties
-----------------------

The DML-LME estimator has several desirable properties:

1. **√n-consistency**: Converges at parametric rate despite high dimensions
2. **Asymptotic normality**: Enables standard inference
3. **Double robustness**: Consistent if either nuisance function is correctly specified
4. **Neyman orthogonality**: Robust to nuisance function estimation errors

For formal proofs and additional technical details, see Chernozhukov et al. (2018) and our paper.

References
----------

* Brunswik, E. (1952). The conceptual framework of psychology. University of Chicago Press.
* Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.
* Li, R. V., & Biesanz, J. C. (2025). High-dimensional perception with the Double Machine Learning Lens Model Equation (DML-LME). *Psychometrika*.