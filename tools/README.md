# Tools Directory

This directory contains utility scripts and generated outputs for the HAAM project.

## Structure

- **`analyzers/`** - Codebase analysis scripts
  - `unified_analyzer.py` - Original analyzer
  - `unified_analyzer_clean.py` - Improved version with better UI
  
- **`codebase_visualization/`** - Generated HTML visualizations of the codebase structure
  - See README.md in that folder for details on the latest visualization

## Usage

To generate a new codebase visualization:
```bash
cd ..  # Go to project root
python tools/analyzers/unified_analyzer_clean.py . -p haam -o tools/codebase_visualization/haam_visualization_new.html --open
```