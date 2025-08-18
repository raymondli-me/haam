haam.haam\_visualizations module
================================

Visualization Module
--------------------

Creates interpretable visualizations of the DML-LME results to reveal how different perceivers (human vs AI) utilize the perceptual cue space. These visualizations are designed to communicate complex statistical results to both technical and non-technical audiences.

**Main Visualizations:**

- **3D UMAP Projections**: Shows how judgments vary across the high-dimensional embedding space, with PC arrows indicating directions of maximum variance
- **Framework Diagrams**: Illustrates the mediation pathways from criterion through PCs to judgments
- **Coefficient Grids**: Displays all 200 PC effects in a compact, interpretable format
- **Word Clouds**: Reveals semantic content associated with each principal component

The visualizations implement design principles from the paper, using:

- Red/blue color schemes for high/low PC values
- Arrow overlays showing PC directions in UMAP space
- Interactive Plotly figures for exploration
- Validity coloring to indicate measurement quality

All visualizations support custom PC naming for domain-specific interpretation.

.. automodule:: haam.haam_visualizations
   :members:
   :undoc-members:
   :show-inheritance:
