Methodology Guide
=================

This guide provides practical details on how HAAM implements the Double Machine Learning Lens Model Equation (DML-LME) and related statistical methods.

Data Requirements
-----------------

To run a HAAM analysis, you need:

1. **Criterion Variable** (``Y_e``): Ground truth labels or objective measurements
   
   - Binary, ordinal, or continuous
   - Represents the "environmental reality" being judged
   - Examples: actual social class, true sentiment, diagnostic category

2. **Human Judgments** (``Y_h``): Human predictions or ratings of the criterion
   
   - Same scale as criterion
   - Can be aggregated across multiple raters
   - Missing values allowed (will be excluded)

3. **AI Judgments** (``Y_ai``): AI model predictions
   
   - Same scale as criterion
   - Can be probabilities or class predictions
   - From any AI system (GPT, BERT, custom model)

4. **Perceptual Cues** (``X``): High-dimensional features
   
   - Text embeddings (BERT, Word2Vec, etc.)
   - Extracted features (linguistic, acoustic, visual)
   - Can be generated automatically from text if not provided

Calculating Percentage of Mediated Accuracy (PoMA)
--------------------------------------------------

The core metric in HAAM is PoMA, which quantifies what proportion of a perceiver's accuracy flows through the measured cue space.

**Step 1: Define Accuracy**

Accuracy is operationalized as the product of judgment and criterion:

.. math::

   A_i = Y_{judgment,i} \times Y_{criterion,i}

This captures whether the judgment correctly predicts the criterion direction and magnitude.

**Step 2: Estimate Total Effect**

The total effect of judgment on accuracy (without considering cues):

.. math::

   \text{Total Effect} = \mathbb{E}[A|D=1] - \mathbb{E}[A|D=0]

Where D is the judgment variable (treated as "treatment").

**Step 3: Estimate Direct and Indirect Effects**

Using DML, decompose into:

- **Direct Effect**: Accuracy not mediated by cues
- **Indirect Effect**: Accuracy mediated through cues

**Step 4: Calculate PoMA**

.. math::

   \text{PoMA} = \frac{\text{Indirect Effect}}{\text{Total Effect}} \times 100\%

Implementation Details
----------------------

**Machine Learning Models**

HAAM uses several ML algorithms for nuisance function estimation:

1. **Random Forests**: Default for general nonlinear relationships
2. **Gradient Boosting**: Alternative for complex interactions  
3. **Elastic Net**: When linear relationships are expected
4. **Neural Networks**: For very high-dimensional text/image data

The choice is made via cross-validation or can be specified by the user.

**Cross-Fitting Procedure**

To prevent overfitting, HAAM implements K-fold cross-fitting:

1. Split data into K folds (default K=5)
2. For each fold k:
   
   - Train nuisance functions on all folds except k
   - Generate predictions for fold k
   
3. Combine predictions across all folds
4. Estimate treatment effect on out-of-sample predictions

**Bootstrap Inference**

Confidence intervals use stratified bootstrap:

1. Resample observations with replacement (B=1000 iterations)
2. Maintain treatment/control balance in each bootstrap sample
3. Re-estimate PoMA for each bootstrap sample
4. Use percentile method for 95% CI

Statistical Tests
-----------------

HAAM includes several hypothesis tests:

**1. Mediation Significance Test**

Tests whether indirect effect differs from zero:

.. math::

   H_0: \text{Indirect Effect} = 0

Uses bootstrap distribution of indirect effects.

**2. Human-AI Comparison Test**

Tests whether humans and AI have different PoMA values:

.. math::

   H_0: \text{PoMA}_{human} = \text{PoMA}_{AI}

Uses paired bootstrap test.

**3. Cue Importance Tests**

For each principal component or cue dimension:

- Correlation with accuracy
- Partial correlation controlling for other cues
- Permutation importance

Advanced Features
-----------------

**Handling Missing Data**

HAAM handles missing values through:

- Listwise deletion for criterion/judgment variables
- Imputation for cue variables (optional)
- Bootstrap samples maintain missing data pattern

**Validity Diagnostics**

Several checks ensure results are valid:

1. **Overlap Assumption**: Treatment and control have common support
2. **Positivity**: Sufficient variation in treatment given cues
3. **Consistency**: Stable estimates across CV folds
4. **Convergence**: ML models converge properly

**Sensitivity Analysis**

HAAM provides tools to assess robustness:

- Leave-one-cue-out analysis
- Different ML algorithm comparison
- Alternative accuracy definitions
- Varying number of CV folds

Interpretation Guidelines
-------------------------

**High PoMA (>80%)**
- Perceiver heavily relies on measured cue space
- Good news for interpretability
- Suggests comprehensive cue measurement

**Moderate PoMA (40-80%)**
- Balanced use of measured and unmeasured information
- Typical for human perceivers
- May indicate contextual processing

**Low PoMA (<40%)**
- Perceiver uses information beyond measured cues
- Could indicate intuition, expertise, or missing measurements
- Warrants investigation of additional cues

**Human vs AI Patterns**
- AI typically shows higher PoMA (more cue-dependent)
- Humans often show lower PoMA (more contextual)
- Differences reveal complementary strengths

Computational Considerations
----------------------------

**Scalability**

HAAM scales to large datasets through:

- Sparse matrix representations for text
- Parallel processing for bootstrap
- Efficient ML implementations (scikit-learn)
- Optional GPU acceleration for neural networks

**Memory Requirements**

Approximate memory usage:

- 1,000 observations, 1,000 cues: ~100MB
- 10,000 observations, 5,000 cues: ~2GB  
- 100,000 observations, 10,000 cues: ~20GB

**Computation Time**

Typical runtime on modern hardware:

- Small dataset (<1K obs): 1-5 minutes
- Medium dataset (1-10K obs): 5-30 minutes
- Large dataset (>10K obs): 30 minutes - 2 hours

Best Practices
--------------

1. **Cue Selection**: Include all potentially relevant perceptual features
2. **Sample Size**: Minimum 500 observations for stable estimates
3. **Cross-Validation**: Use 5-10 folds for reliability
4. **Bootstrap**: 1000+ iterations for accurate CIs
5. **Diagnostics**: Always check validity assumptions
6. **Interpretation**: Consider domain knowledge alongside statistics