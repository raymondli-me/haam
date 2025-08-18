# Archived Files

This directory contains files that were identified as orphans (no import relationships) during codebase cleanup on 2025-08-18.

## Contents

### `/scripts/`
- **Codebase Analyzers**: Various iterations of codebase analysis tools
  - `analyze_codebase_better.py`
  - `analyze_codebase_final.py`
  - `codebase_analyzer_complete.py`
  - `generate_codebase_viz.py`
  - `visualize_codebase*.py` (multiple versions)

- **Word Cloud Generators**: Experimental validity coloring implementations
  - `generate_wordclouds_validity_aligned.py`
  - `generate_wordclouds_validity_aligned_v2.py`
  - `generate_wordclouds_validity_enhanced.py`
  - `generate_wordclouds_validity_final_aligned.py`
  - `generate_wordclouds_validity_nvembed.py`
  - `haam_wordcloud_aligned_patch.py`

- **Other Scripts**:
  - `colab_cell_criterion_ranking.py` - Small Colab utility

### `/modules/`
- `haam_setup.py` - Setup utilities (orphaned)
- `haam_usage_guide.py` - Usage documentation as code
- `haam_visualizations_backup.py` - Backup of visualization module
- `haam_wordcloud_backup.py` - Backup of word cloud module

### `/other/`
- Configuration and setup files if needed

## Note
These files were archived because they:
1. Had no import relationships with the main codebase (orphans)
2. Represented older iterations or experimental features
3. Were backup copies of active modules

They are preserved here for reference and potential future use.