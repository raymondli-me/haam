# HAAM Codebase Visualization Guide

## 🎯 What We've Created

I've built an interactive Cytoscape.js visualization that analyzes your Python codebase and creates a reactive HTML dashboard. Here's what it does:

### Features:
1. **AST-based Analysis**: Parses Python files to extract:
   - Import relationships
   - Function/class counts
   - Cyclomatic complexity
   - File sizes and line counts

2. **Interactive Visualization**:
   - Multiple layout algorithms (hierarchical, force-directed, circle, grid)
   - Node sizing based on file size
   - Color coding by file type (module, script, package, test)
   - Click nodes for detailed information

3. **Analysis Tools**:
   - **Find Circular Dependencies**: Detects import cycles
   - **Show Complex Files**: Highlights files with high cyclomatic complexity
   - **Show Potentially Unused**: Finds modules with no imports

## 🚀 Using the Visualization

1. **Open the HTML file**:
   ```bash
   open codebase_visualization.html
   ```

2. **Navigate the Graph**:
   - Pan: Click and drag background
   - Zoom: Scroll or pinch
   - Select: Click on nodes
   - Move nodes: Drag them around

3. **Try Different Layouts**:
   - **Hierarchical**: Shows dependency tree structure
   - **Force-Directed**: Reveals natural clusters
   - **Circle/Grid**: Alternative arrangements

## 📊 Key Insights from Your Codebase

Based on the analysis:
- **42 Python files** analyzed
- **13,360 total lines** of code
- **Average complexity**: 24.2 (quite high!)
- **Very few imports detected** (only 3) - suggests:
  - Many standalone scripts
  - Possible import parsing issues with Colab-style scripts
  - Good modularity (files are independent)

## 🔍 What to Look For

### 1. **Cluttered Scripts** (Red Nodes)
Your `generate_wordclouds_*.py` scripts appear as isolated nodes - perfect candidates for cleanup!

### 2. **Core Package Structure** (Blue/Green Nodes)
The `haam/` package modules should show import relationships

### 3. **Test Files** (Orange Nodes)
Test scripts that might not be part of a formal test suite

## 🛠️ Advanced Visualization Options

### Option 1: Enhanced with NetworkX Analysis
```python
import networkx as nx
from pyvis.network import Network

# Convert to NetworkX for advanced analysis
G = nx.DiGraph()
# Add PageRank, betweenness centrality, etc.
```

### Option 2: D3.js Force Graph
More customizable but requires more setup:
```javascript
d3.json("graph_data.json").then(function(graph) {
    const simulation = d3.forceSimulation(graph.nodes)
        .force("link", d3.forceLink(graph.edges))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter());
});
```

### Option 3: Gephi Export
For professional network analysis:
```python
# Export to GEXF format
import networkx as nx
nx.write_gexf(G, "haam_codebase.gexf")
```

## 🎨 Other Popular Tools

### For Python Codebases:
1. **pydeps**: Creates dependency graphs
   ```bash
   pip install pydeps
   pydeps haam --cluster
   ```

2. **pyreverse** (Part of Pylint): UML diagrams
   ```bash
   pyreverse -o png -p HAAM haam/
   ```

3. **code2flow**: Dynamic call graphs
   ```bash
   pip install code2flow
   code2flow haam/
   ```

### For General Codebases:
1. **CodeSee**: AI-powered code maps (SaaS)
2. **Sourcetrail**: Interactive code explorer
3. **Gource**: Animated visualization of repo history
4. **SonarQube**: Code quality visualization

## 🏆 Best Practices

### 1. **Regular Updates**
Run the visualization after major refactoring to track improvements

### 2. **Metrics to Track**:
- **Complexity Reduction**: Aim for < 10 per file
- **Import Cycles**: Should be zero
- **Orphaned Files**: Identify and remove
- **God Modules**: Files with too many imports

### 3. **Integration Ideas**:
- Add to CI/CD pipeline
- Generate on each commit
- Track metrics over time
- Create dashboards for team

## 🔧 Customization

The script is easily extensible. You can:
- Add more metrics (test coverage, doc strings)
- Include git history (last modified, authors)
- Add semantic analysis (function similarity)
- Export to different formats (GraphML, JSON, DOT)

## 📈 Next Steps

1. **Clean up the clutter**: Use the visualization to identify redundant `generate_wordclouds_*.py` files
2. **Improve imports**: The low import count suggests opportunity for better code reuse
3. **Reduce complexity**: Target files with complexity > 20
4. **Add tests**: Orange test nodes should be part of a proper test suite

The visualization is now your "living documentation" - use it to make data-driven decisions about code organization!