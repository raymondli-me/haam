# Codebase Visualization Outputs

This folder contains generated HTML visualizations of the HAAM codebase structure. These are **outputs** from visualization scripts, not the scripts themselves.

## Latest Visualization

**Current/Latest**: `haam_visualization_updated.html` (Generated: Aug 17, 2025)

## Visualization Scripts Location

The scripts that generate these visualizations are in the **root directory**:
- `unified_analyzer.py` - Original unified analyzer
- `unified_analyzer_clean.py` - Cleaned version with better UI (RECOMMENDED)

## How to Generate New Visualization

From the project root directory:

```bash
# Generate new visualization with the clean analyzer
python unified_analyzer_clean.py . -p haam -o tools/codebase_visualization/haam_visualization_latest.html --json --open

# Or use the original analyzer
python unified_analyzer.py
```

## Files in This Directory

### Active/Current Visualizations
- `haam_visualization_updated.html` - Latest comprehensive visualization
- `haam_visualization_updated_data.json` - Associated data file

### Historical/Development Versions
Various iterations created during development:
- `codebase_*.html` - Early codebase analysis attempts
- `haam_structure*.html` - Structure-focused visualizations  
- `haam_*viz*.html` - Dynamic visualization experiments
- `haam_analysis.html` - Analysis-focused view

## Viewing Visualizations

Simply open any HTML file in a web browser. They are self-contained with all dependencies included.

```bash
# Open the latest visualization
open haam_visualization_updated.html
```

## Notes

- These are static HTML files with embedded JavaScript (Cytoscape.js)
- No server required - just open in browser
- Files can be shared/viewed anywhere
- Keep the associated JSON files with their HTML counterparts