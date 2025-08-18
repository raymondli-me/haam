Interpretation Guide
====================

This guide helps you understand and interpret HAAM analysis results in the context of the Human-AI Accuracy Model.

Understanding PoMA Values
-------------------------

The Percentage of Mediated Accuracy (PoMA) is the key metric from HAAM analysis. It tells you what percentage of a perceiver's accuracy flows through the measured perceptual cue space.

**Interpreting PoMA Ranges:**

.. list-table:: PoMA Interpretation Guide
   :widths: 20 30 50
   :header-rows: 1

   * - PoMA Range
     - Interpretation
     - Implications
   * - 0-20%
     - Very Low Mediation
     - Perceiver relies heavily on unmeasured information, intuition, or direct knowledge
   * - 20-40%
     - Low Mediation
     - Significant accuracy from sources outside measured cues
   * - 40-60%
     - Moderate Mediation
     - Balanced use of measured and unmeasured information
   * - 60-80%
     - High Mediation
     - Most accuracy flows through measured perceptual features
   * - 80-100%
     - Very High Mediation
     - Nearly all accuracy explained by measured cue space

Comparing Human vs AI Patterns
------------------------------

Typical patterns observed in research:

**AI Systems Often Show:**
- Higher PoMA (80-95%)
- Greater reliance on textual/surface features
- More consistent use of available cues
- Less contextual processing

**Humans Often Show:**
- Lower PoMA (40-70%)
- Integration of background knowledge
- Variable cue usage across contexts
- More holistic processing

**Example Interpretation:**

.. code-block:: text

   Human PoMA: 62%
   AI PoMA: 89%
   
   Interpretation: Humans achieve 62% of their accuracy through the measured
   text features, while AI achieves 89%. This suggests humans are using 
   additional information sources (context, world knowledge) that aren't 
   captured in the text embeddings.

Reading the Visualizations
--------------------------

**1. Main Framework Diagram**

Shows the top 9 principal components mediating between criterion and judgments:

- **Arrow thickness**: Strength of relationship
- **Red boxes**: Positive effects
- **Blue boxes**: Negative effects
- **PC labels**: Custom names or topic descriptions

**2. 3D UMAP Projections**

Reveals structure in the high-dimensional space:

- **Point colors**: Values of selected variable (Y, AI, Human)
- **Clustering**: Similar texts group together
- **PC arrows**: Direction of maximum variance for each component
- **Spread**: Greater spread indicates more variation

**3. Word Clouds**

For each principal component:

- **Red words**: Associated with high PC values
- **Blue words**: Associated with low PC values
- **Size**: Strength of association
- **Validity coloring**: Darker = more reliable measurement

Statistical Significance
------------------------

**Key Metrics to Check:**

1. **Correlation Significance**
   
   - p < 0.05: Significant relationship
   - p < 0.01: Strong evidence
   - p < 0.001: Very strong evidence

2. **Bootstrap Confidence Intervals**
   
   - If 95% CI excludes zero: Effect is significant
   - Narrower CIs indicate more precise estimates

3. **Model Performance (R²)**
   
   - Cross-validated R² is most reliable
   - Compare human vs AI R² values
   - Higher R² means better prediction

**Example Output Interpretation:**

.. code-block:: text

   AI Mediation Analysis:
   - Total Effect: 0.73 [0.68, 0.78]
   - Direct Effect: 0.08 [0.02, 0.14]  
   - Indirect Effect: 0.65 [0.59, 0.71]
   - PoMA: 89% [84%, 94%]
   
   Interpretation: AI's total accuracy effect is 0.73. Of this, 0.65 flows
   through measured cues (indirect) and only 0.08 is direct. The 95% CI
   for PoMA [84%, 94%] doesn't include low values, confirming high mediation.

Practical Guidelines
--------------------

**When PoMA is High (>80%):**
- Your cue measurement is comprehensive
- The perceiver heavily relies on measured features
- Good for interpretability and explainability
- May indicate algorithmic/systematic processing

**When PoMA is Low (<40%):**
- Important information exists outside measured cues
- Consider expanding feature set
- May indicate expertise, intuition, or context use
- Common for human experts in specialized domains

**When Human-AI PoMA Differs Greatly (>30% gap):**
- Fundamental differences in information processing
- Humans may use unmeasured context
- AI may over-rely on surface features
- Opportunity for complementary strengths

Reporting Results
-----------------

When reporting HAAM results in a paper:

.. code-block:: text

   "We applied the Human-AI Accuracy Model (HAAM) to decompose judgment
   accuracy into direct and mediated pathways. Human judges achieved a
   PoMA of 58% (95% CI: [52%, 64%]), indicating moderate reliance on
   the measured textual features. In contrast, the AI model showed a
   PoMA of 91% (95% CI: [88%, 94%]), suggesting nearly complete dependence
   on the extracted features. This 33 percentage point difference
   (p < 0.001) reveals fundamentally different information processing
   strategies between human and artificial perceivers."

Common Pitfalls
---------------

**1. Over-interpreting Small Differences**
   
   - Check confidence intervals for overlap
   - Consider practical significance, not just statistical

**2. Ignoring Cross-validation**
   
   - Always use CV metrics, not in-sample
   - In-sample R² will be inflated

**3. Assuming Causation**
   
   - PoMA shows mediation, not causation
   - Consider alternative explanations

**4. Neglecting Domain Context**
   
   - Interpret results within your specific domain
   - What counts as "high" PoMA varies by task

Next Steps
----------

After obtaining initial results:

1. **Examine Individual PCs**: Which components drive accuracy?
2. **Compare Subgroups**: Does PoMA vary across data subsets?
3. **Sensitivity Analysis**: How robust are results to parameter choices?
4. **Feature Engineering**: Can you capture unmeasured information?