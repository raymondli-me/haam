# Topic Labeling Methodology in HAAM

## Overview

The HAAM framework automatically generates interpretable topic labels for principal components (PCs) by analyzing which document clusters (topics) are most strongly associated with high and low values of each PC. This document details the technical methodology behind this process.

## 1. Document Clustering Pipeline

### 1.1 Dimensionality Reduction
- **Method**: UMAP (Uniform Manifold Approximation and Projection)
- **Parameters**:
  ```python
  n_neighbors=5      # Balance between local and global structure
  n_components=5     # Reduce to 5D for clustering
  min_dist=0.0      # Allow tight clusters
  metric='cosine'   # Similarity metric for embeddings
  ```

### 1.2 Clustering Algorithm
- **Method**: HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- **Parameters**:
  ```python
  min_cluster_size=10   # Minimum documents per topic
  min_samples=2         # Core point definition
  metric='euclidean'    # Distance in UMAP space
  ```
- **Outlier Handling**: Documents not assigned to any cluster are labeled as -1 (noise)

## 2. Keyword Extraction via c-TF-IDF

### 2.1 Text Preprocessing
- **Vectorization**: CountVectorizer with:
  ```python
  max_features=1000     # Vocabulary size limit
  ngram_range=(1,2)     # Unigrams and bigrams
  stop_words='english'  # Standard English stopwords
  min_df=2             # Minimum document frequency
  ```

### 2.2 c-TF-IDF Calculation
The class-based TF-IDF formula (adapted from BERTopic):

```
c-TF-IDF = tf_t,c * log(1 + (A / tf_t))
```

Where:
- `tf_t,c` = frequency of term t in cluster c
- `tf_t` = frequency of term t across all documents  
- `A` = average number of words per cluster

### 2.3 Keyword Selection
- Extract top 10 keywords per topic cluster
- Keywords are ranked by c-TF-IDF score
- Both single words and 2-word phrases are included

## 3. PC-Topic Association Analysis

### 3.1 Statistical Testing
For each PC and topic cluster combination:

1. **Calculate Topic Mean**: Average PC value for documents in the topic
2. **Compute Percentile**: Where this mean falls in the PC's distribution
3. **Effect Size**: Cohen's d comparing topic vs non-topic documents
   ```
   d = (mean_topic - mean_non_topic) / pooled_std
   ```
4. **Significance Test**: Two-sample t-test (p < 0.05 threshold)

### 3.2 Topic Categorization
Topics are classified as:
- **High Topics**: Average percentile > 75th percentile
- **Low Topics**: Average percentile < 25th percentile
- **Excluded**: Topics with < 5 documents or non-significant p-values

### 3.3 Ranking Within Categories
Topics are ranked by:
1. Statistical significance (p-value)
2. Effect size magnitude (|Cohen's d|)
3. Percentile extremity (distance from 50th percentile)

## 4. Visualization Display Rules

### 4.1 Topic Selection for Display
- **Maximum Topics**: Top 3 high topics and bottom 3 low topics per PC
- **Display Priority**: Only the first (most significant) topic from each category
- **Fallback**: Empty string if no topics meet criteria

### 4.2 Keyword Display Formatting
From the selected topic's 10 keywords:
- **Display Count**: Show only the first 3 keywords
- **Format**: Comma-separated list (e.g., "keyword1, keyword2, keyword3")
- **Rationale**: Balance between informativeness and visual clarity

### 4.3 HTML Implementation
```javascript
// Example PC data structure
pc_data = {
    'pc_number': 1,
    'topic_high': 'excellent, amazing, perfect',  // First 3 of 10 keywords
    'topic_low': 'terrible, worst, awful',        // First 3 of 10 keywords
    // ... other PC properties
}
```

## 5. Configuration Options

### 5.1 TopicAnalyzer Parameters
```python
analyzer = TopicAnalyzer(
    min_cluster_size=10,      # Minimum topic size
    n_neighbors=5,            # UMAP local structure
    ngram_range=(1, 2),       # Include bigrams
    max_features=1000,        # Vocabulary size
    top_n_keywords=10         # Keywords per topic
)
```

### 5.2 Association Analysis Parameters
```python
associations = analyzer.get_pc_topic_associations(
    embeddings=embeddings,
    min_topic_size=5,         # Filter small topics
    p_value_threshold=0.05,   # Significance level
    top_k_per_side=3         # Topics per PC side
)
```

## 6. Interpretation Guidelines

### 6.1 High Topics
Represent content strongly associated with positive PC values. Interpretation depends on what the PC captures:
- If PC1 captures sentiment → High topics show positive language
- If PC2 captures technicality → High topics show technical terms

### 6.2 Low Topics  
Represent content strongly associated with negative PC values:
- Sentiment PC → Low topics show negative language
- Technicality PC → Low topics show casual/simple language

### 6.3 Missing Topics
Empty topic labels indicate:
- No statistically significant associations found
- All document clusters are near the PC median
- Insufficient data (< 5 documents) in extreme clusters

## 7. Technical Considerations

### 7.1 Computational Efficiency
- UMAP reduction is performed once and cached
- c-TF-IDF matrix is sparse for memory efficiency
- Statistical tests use vectorized operations

### 7.2 Statistical Validity
- Multiple testing correction not applied (intentional for exploratory analysis)
- Effect size ensures practical significance beyond statistical significance
- Percentile thresholds prevent spurious associations

### 7.3 Limitations
- Requires sufficient text data (minimum ~100 documents)
- Topics may not emerge for PCs that don't align with semantic content
- Bigrams may miss longer meaningful phrases
- Language-specific (currently English stopwords only)

## 8. Example Output

For a dataset of product reviews:

```
PC1 (Sentiment):
  High: "love, amazing, perfect"
  Low: "hate, terrible, waste"

PC2 (Product Type):  
  High: "laptop, computer, keyboard"
  Low: "shoe, size, fit"

PC3 (Technical Detail):
  High: "cpu, ram, gigahertz"  
  Low: "easy, simple, basic"
```

## References

- UMAP: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- HDBSCAN: Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates.
- c-TF-IDF: Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.