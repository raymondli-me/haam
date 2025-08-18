Quick Start Guide
=================

This guide will get you up and running with HAAM in 5 minutes.

Installation
------------

Install HAAM using pip:

.. code-block:: bash

   pip install haam

Or install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/haam.git
   cd haam
   pip install -e .

Basic Usage
-----------

Here's a minimal example to get started:

.. code-block:: python

   import numpy as np
   from haam import HAAMAnalysis
   
   # Example data
   criterion = np.array([1, 2, 3, 4, 5] * 20)  # Ground truth
   ai_judgment = criterion + np.random.normal(0, 0.5, 100)  # AI predictions
   human_judgment = criterion + np.random.normal(0, 0.3, 100)  # Human ratings
   
   # Create analysis
   analysis = HAAMAnalysis(
       criterion=criterion,
       ai_judgment=ai_judgment,
       human_judgment=human_judgment
   )
   
   # Display comprehensive results
   analysis.display()

Understanding the Output
------------------------

HAAM generates several types of output:

1. **Statistical Summary**: Correlations, ANOVA results, and validity metrics
2. **Visualizations**: 3D PCA plots showing relationships between variables
3. **Performance Metrics**: Classification accuracy, regression R²
4. **Topic Analysis**: If text data is provided, automatic topic discovery

Key Parameters
--------------

When initializing ``HAAMAnalysis``, you can customize behavior:

.. code-block:: python

   analysis = HAAMAnalysis(
       criterion=data,           # Required: ground truth values
       ai_judgment=ai_data,      # Required: AI predictions
       human_judgment=human_data, # Optional: human ratings
       
       # Visualization options
       n_neighbors=5,            # UMAP parameter
       min_dist=0.0,            # UMAP parameter
       
       # Topic modeling
       min_cluster_size=10,      # HDBSCAN parameter
       nr_topics=None,           # Auto-detect topics if None
       
       # Output options
       show_plots=True,          # Display visualizations
       save_plots=False,         # Save plots to files
       output_dir='haam_output'  # Directory for outputs
   )

Working with Text Data
----------------------

HAAM excels at analyzing text-based AI outputs:

.. code-block:: python

   from haam import HAAMAnalysis
   
   # Load your data
   texts = ["AI generated text 1", "AI generated text 2", ...]
   criterion = [1, 2, ...]  # Ground truth labels
   ai_scores = [1.1, 2.3, ...]  # AI predictions
   
   # Create analysis with text
   analysis = HAAMAnalysis(
       criterion=criterion,
       ai_judgment=ai_scores,
       text_data=texts
   )
   
   # Generate word clouds by topic
   analysis.create_wordclouds()

Interpreting Results
--------------------

Key metrics to look for:

* **Correlation > 0.7**: Strong alignment between AI and ground truth
* **ANOVA p < 0.05**: AI can distinguish between criterion levels
* **High Topic Coherence**: Well-defined content clusters
* **Validity Coloring**: Visual indicator of measurement quality

Next Steps
----------

* See the :doc:`examples/index` for detailed use cases
* Read the :doc:`guide/concepts` to understand the methodology
* Explore the :doc:`api/modules` for advanced features