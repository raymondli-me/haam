#!/usr/bin/env python3
"""
HAAM Codebase Visualizer using Cytoscape.js
===========================================
Analyzes Python codebase structure and generates an interactive HTML visualization.
"""

import ast
import os
import json
import re
from pathlib import Path
from collections import defaultdict
import hashlib

class CodebaseAnalyzer:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path)
        self.nodes = []
        self.edges = []
        self.node_data = {}
        
    def analyze(self, ignore_patterns=None):
        """Analyze the codebase structure."""
        if ignore_patterns is None:
            ignore_patterns = ['__pycache__', '.git', '.pytest_cache', '*.pyc', 'venv', 'env']
        
        # Find all Python files
        python_files = self._find_python_files(ignore_patterns)
        
        # Analyze each file
        for filepath in python_files:
            self._analyze_file(filepath)
        
        # Build the graph data
        return self._build_graph_data()
    
    def _find_python_files(self, ignore_patterns):
        """Find all Python files in the codebase."""
        python_files = []
        
        for root, dirs, files in os.walk(self.root_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if not any(re.match(pattern.replace('*', '.*'), d) 
                                                  for pattern in ignore_patterns)]
            
            for file in files:
                if file.endswith('.py') and not any(re.match(pattern.replace('*', '.*'), file) 
                                                    for pattern in ignore_patterns):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _analyze_file(self, filepath):
        """Analyze a single Python file."""
        relative_path = filepath.relative_to(self.root_path)
        node_id = str(relative_path).replace('/', '.').replace('.py', '')
        
        # Get file stats
        file_stats = filepath.stat()
        file_size = file_stats.st_size
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.splitlines())
                
            # Skip Colab notebook cells (files with !pip or %%magic commands)
            if content.strip().startswith('!') or '!pip install' in content[:200] or '%%' in content[:50]:
                print(f"Skipping Colab-style file: {filepath.name}")
                return
                
            # Parse AST
            tree = ast.parse(content)
            
            # Extract metadata
            imports = self._extract_imports(tree, filepath)
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            complexity = self._calculate_complexity(tree)
            
            # Determine node type
            if filepath.name == '__init__.py':
                node_type = 'package'
            elif 'test' in str(filepath).lower():
                node_type = 'test'
            elif any(script in filepath.name for script in ['generate_', 'run_', 'test_']):
                node_type = 'script'
            else:
                node_type = 'module'
            
            # Add node
            self.nodes.append({
                'data': {
                    'id': node_id,
                    'label': filepath.name,
                    'path': str(relative_path),
                    'size': file_size,
                    'lines': lines,
                    'functions': len(functions),
                    'classes': len(classes),
                    'complexity': complexity,
                    'type': node_type,
                    'imports_count': len(imports)
                }
            })
            
            # Add edges for imports
            for imp in imports:
                if imp['type'] == 'internal':
                    self.edges.append({
                        'data': {
                            'source': node_id,
                            'target': imp['target'],
                            'type': 'import',
                            'label': imp.get('names', '')
                        }
                    })
                    
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            # Still add the node even if parsing fails
            self.nodes.append({
                'data': {
                    'id': node_id,
                    'label': filepath.name,
                    'path': str(relative_path),
                    'size': file_size,
                    'lines': 0,
                    'error': str(e),
                    'type': 'error'
                }
            })
    
    def _extract_imports(self, tree, filepath):
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imp_data = self._resolve_import(alias.name, filepath)
                    if imp_data:
                        imports.append(imp_data)
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imp_data = self._resolve_import(node.module, filepath, 
                                                   names=[n.name for n in node.names])
                    if imp_data:
                        imports.append(imp_data)
        
        return imports
    
    def _resolve_import(self, module_name, current_file, names=None):
        """Resolve an import to determine if it's internal or external."""
        # Check if it's a relative import or internal to the project
        parts = module_name.split('.')
        
        # Try to find the module in the project
        if parts[0] in ['haam', '.', '..']:  # Internal imports
            if module_name.startswith('.'):
                # Relative import - resolve based on current file location
                current_package = current_file.parent
                target = module_name.replace('.', '')
            else:
                # Absolute import
                target = module_name.replace('.', '.')
            
            return {
                'type': 'internal',
                'module': module_name,
                'target': target,
                'names': ','.join(names) if names else ''
            }
        else:
            # External import
            return {
                'type': 'external',
                'module': module_name,
                'names': ','.join(names) if names else ''
            }
    
    def _extract_functions(self, tree):
        """Extract function definitions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': len(node.args.args),
                    'decorators': len(node.decorator_list)
                })
        return functions
    
    def _extract_classes(self, tree):
        """Extract class definitions."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    'name': node.name,
                    'methods': len(methods),
                    'bases': len(node.bases)
                })
        return classes
    
    def _calculate_complexity(self, tree):
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
    
    def _build_graph_data(self):
        """Build the final graph data structure."""
        # Filter out external nodes
        internal_node_ids = {node['data']['id'] for node in self.nodes}
        filtered_edges = [edge for edge in self.edges 
                         if edge['data']['target'] in internal_node_ids]
        
        # Calculate additional metrics
        self._calculate_metrics(filtered_edges)
        
        return {
            'nodes': self.nodes,
            'edges': filtered_edges,
            'metadata': {
                'total_files': len(self.nodes),
                'total_imports': len(filtered_edges),
                'total_lines': sum(n['data'].get('lines', 0) for n in self.nodes),
                'avg_complexity': sum(n['data'].get('complexity', 0) for n in self.nodes) / len(self.nodes) if self.nodes else 0
            }
        }
    
    def _calculate_metrics(self, edges):
        """Calculate graph metrics for each node."""
        # Calculate in-degree and out-degree
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for edge in edges:
            out_degree[edge['data']['source']] += 1
            in_degree[edge['data']['target']] += 1
        
        # Add metrics to nodes
        for node in self.nodes:
            node_id = node['data']['id']
            node['data']['in_degree'] = in_degree.get(node_id, 0)
            node['data']['out_degree'] = out_degree.get(node_id, 0)
            node['data']['total_degree'] = node['data']['in_degree'] + node['data']['out_degree']


def generate_visualization_html(graph_data, output_file='codebase_visualization.html'):
    """Generate the HTML file with Cytoscape.js visualization."""
    
    # Ensure graph_data is properly formatted
    graph_json = json.dumps(graph_data, indent=2)
    
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>HAAM Codebase Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-cola@2.5.1/cytoscape-cola.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/webcola@3.4.0/WebCola/cola.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
        }
        #cy {
            flex: 1;
            background-color: #f0f0f0;
        }
        #sidebar {
            width: 300px;
            background: white;
            padding: 20px;
            overflow-y: auto;
            box-shadow: -2px 0 5px rgba(0,0,0,0.1);
        }
        #controls {
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .control-group {
            margin-bottom: 10px;
        }
        button {
            padding: 8px 16px;
            margin: 2px;
            cursor: pointer;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
        }
        button:hover {
            background: #0056b3;
        }
        .metric {
            margin: 5px 0;
            font-size: 14px;
        }
        .metric-label {
            font-weight: bold;
            color: #666;
        }
        #node-info {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
        .legend {
            margin-top: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div id="cy"></div>
    <div id="sidebar">
        <h2>HAAM Codebase Explorer</h2>
        
        <div id="controls">
            <h3>Layout</h3>
            <div class="control-group">
                <button onclick="runLayout('dagre')">Hierarchical</button>
                <button onclick="runLayout('cose')">Force-Directed</button>
                <button onclick="runLayout('circle')">Circle</button>
                <button onclick="runLayout('grid')">Grid</button>
            </div>
            
            <h3>Analysis</h3>
            <div class="control-group">
                <button onclick="findCycles()">Find Circular Dependencies</button>
                <button onclick="highlightComplex()">Show Complex Files</button>
                <button onclick="highlightUnused()">Show Potentially Unused</button>
                <button onclick="resetHighlight()">Reset</button>
            </div>
            
            <h3>View</h3>
            <div class="control-group">
                <button onclick="cy.fit()">Fit to Screen</button>
                <button onclick="toggleEdgeLabels()">Toggle Edge Labels</button>
            </div>
        </div>
        
        <div id="stats">
            <h3>Statistics</h3>
            <div class="metric">
                <span class="metric-label">Total Files:</span> 
                <span id="total-files">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Lines:</span> 
                <span id="total-lines">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Imports:</span> 
                <span id="total-imports">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Avg Complexity:</span> 
                <span id="avg-complexity">0</span>
            </div>
        </div>
        
        <div class="legend">
            <h3>Node Types</h3>
            <div class="legend-item">
                <div class="legend-color" style="background: #3498db;"></div>
                <span>Module</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e74c3c;"></div>
                <span>Script</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2ecc71;"></div>
                <span>Package</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f39c12;"></div>
                <span>Test</span>
            </div>
        </div>
        
        <div id="node-info">
            <h3>Node Details</h3>
            <p>Click on a node to see details</p>
        </div>
    </div>
    
    <script>
        const graphData = ''' + graph_json + ''';
        
        // Wait for DOM to be ready
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize Cytoscape
            const cy = window.cy = cytoscape({
                container: document.getElementById('cy'),
                elements: graphData,
            style: [
                {
                    selector: 'node',
                    style: {
                        'label': 'data(label)',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'font-size': '10px',
                        'width': 'mapData(lines, 0, 1000, 20, 80)',
                        'height': 'mapData(lines, 0, 1000, 20, 80)',
                        'background-color': function(ele) {
                            const type = ele.data('type');
                            switch(type) {
                                case 'module': return '#3498db';
                                case 'script': return '#e74c3c';
                                case 'package': return '#2ecc71';
                                case 'test': return '#f39c12';
                                case 'error': return '#95a5a6';
                                default: return '#34495e';
                            }
                        },
                        'border-width': 2,
                        'border-color': '#2c3e50'
                    }
                },
                {
                    selector: 'node:selected',
                    style: {
                        'border-width': 4,
                        'border-color': '#e74c3c'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#95a5a6',
                        'target-arrow-color': '#95a5a6',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': '',
                        'font-size': '8px',
                        'text-rotation': 'autorotate'
                    }
                },
                {
                    selector: '.highlighted',
                    style: {
                        'background-color': '#e74c3c',
                        'line-color': '#e74c3c',
                        'target-arrow-color': '#e74c3c'
                    }
                },
                {
                    selector: '.dimmed',
                    style: {
                        'opacity': 0.2
                    }
                }
            ],
            layout: {
                name: 'dagre',
                rankDir: 'TB',
                animate: true,
                animationDuration: 1000
            }
        });
        
        // Update statistics
        document.getElementById('total-files').textContent = graphData.metadata.total_files;
        document.getElementById('total-lines').textContent = graphData.metadata.total_lines.toLocaleString();
        document.getElementById('total-imports').textContent = graphData.metadata.total_imports;
        document.getElementById('avg-complexity').textContent = graphData.metadata.avg_complexity.toFixed(1);
        
        // Node click handler
        cy.on('tap', 'node', function(evt) {
            const node = evt.target;
            const data = node.data();
            
            let html = `
                <h3>${data.label}</h3>
                <div class="metric"><span class="metric-label">Path:</span> ${data.path}</div>
                <div class="metric"><span class="metric-label">Type:</span> ${data.type}</div>
                <div class="metric"><span class="metric-label">Lines:</span> ${data.lines}</div>
                <div class="metric"><span class="metric-label">Size:</span> ${(data.size / 1024).toFixed(1)} KB</div>
                <div class="metric"><span class="metric-label">Functions:</span> ${data.functions || 0}</div>
                <div class="metric"><span class="metric-label">Classes:</span> ${data.classes || 0}</div>
                <div class="metric"><span class="metric-label">Complexity:</span> ${data.complexity || 0}</div>
                <div class="metric"><span class="metric-label">Imports:</span> ${data.out_degree || 0}</div>
                <div class="metric"><span class="metric-label">Imported by:</span> ${data.in_degree || 0}</div>
            `;
            
            if (data.error) {
                html += `<div class="metric" style="color: red;"><span class="metric-label">Error:</span> ${data.error}</div>`;
            }
            
            document.getElementById('node-info').innerHTML = html;
        });
        
        // Layout functions
        window.runLayout = function(name) {
            let options = { name: name, animate: true, animationDuration: 1000 };
            
            if (name === 'dagre') {
                options.rankDir = 'TB';
                options.spacingFactor = 1.5;
            } else if (name === 'cose') {
                options.nodeRepulsion = 4000;
                options.idealEdgeLength = 100;
            }
            
            cy.layout(options).run();
        }
        
        // Analysis functions
        window.findCycles = function() {
            // Simple cycle detection (not comprehensive)
            resetHighlight();
            
            const visited = new Set();
            const recursionStack = new Set();
            const cycleNodes = new Set();
            
            function hasCycle(nodeId) {
                visited.add(nodeId);
                recursionStack.add(nodeId);
                
                const node = cy.getElementById(nodeId);
                const outgoers = node.outgoers('node');
                
                for (let i = 0; i < outgoers.length; i++) {
                    const neighbor = outgoers[i];
                    const neighborId = neighbor.id();
                    
                    if (!visited.has(neighborId)) {
                        if (hasCycle(neighborId)) {
                            cycleNodes.add(nodeId);
                            return true;
                        }
                    } else if (recursionStack.has(neighborId)) {
                        cycleNodes.add(nodeId);
                        cycleNodes.add(neighborId);
                        return true;
                    }
                }
                
                recursionStack.delete(nodeId);
                return false;
            }
            
            cy.nodes().forEach(node => {
                if (!visited.has(node.id())) {
                    hasCycle(node.id());
                }
            });
            
            if (cycleNodes.size > 0) {
                cy.elements().addClass('dimmed');
                cycleNodes.forEach(nodeId => {
                    cy.getElementById(nodeId).removeClass('dimmed').addClass('highlighted');
                });
                alert(`Found potential circular dependencies involving ${cycleNodes.size} files`);
            } else {
                alert('No circular dependencies found');
            }
        }
        
        window.highlightComplex = function() {
            resetHighlight();
            
            const threshold = 20; // Complexity threshold
            let complexNodes = 0;
            
            cy.nodes().forEach(node => {
                if (node.data('complexity') > threshold) {
                    node.addClass('highlighted');
                    complexNodes++;
                }
            });
            
            if (complexNodes > 0) {
                cy.elements().addClass('dimmed');
                cy.elements('.highlighted').removeClass('dimmed');
                alert(`Found ${complexNodes} files with high complexity (> ${threshold})`);
            } else {
                alert('No highly complex files found');
            }
        }
        
        window.highlightUnused = function() {
            resetHighlight();
            
            let unusedNodes = 0;
            
            cy.nodes().forEach(node => {
                if (node.data('in_degree') === 0 && node.data('type') !== 'script' && node.data('type') !== 'test') {
                    node.addClass('highlighted');
                    unusedNodes++;
                }
            });
            
            if (unusedNodes > 0) {
                cy.elements().addClass('dimmed');
                cy.elements('.highlighted').removeClass('dimmed');
                alert(`Found ${unusedNodes} potentially unused modules (no imports)`);
            } else {
                alert('No potentially unused modules found');
            }
        }
        
        window.resetHighlight = function() {
            cy.elements().removeClass('highlighted dimmed');
        }
        
        let showEdgeLabels = false;
        window.toggleEdgeLabels = function() {
            showEdgeLabels = !showEdgeLabels;
            cy.style()
                .selector('edge')
                .style('label', showEdgeLabels ? 'data(label)' : '')
                .update();
        }
        
            // Initial fit
            cy.ready(() => {
                cy.fit();
            });
        }); // End of DOMContentLoaded
    </script>
</body>
</html>'''
    
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"Visualization saved to: {output_file}")


if __name__ == "__main__":
    # Analyze the HAAM codebase
    analyzer = CodebaseAnalyzer(".")
    
    print("Analyzing HAAM codebase...")
    graph_data = analyzer.analyze(ignore_patterns=[
        '__pycache__', '.git', '*.pyc', 'venv', 'env', 
        'build', 'dist', '*.egg-info'
    ])
    
    print(f"\nFound {len(graph_data['nodes'])} files and {len(graph_data['edges'])} imports")
    print(f"Total lines of code: {graph_data['metadata']['total_lines']:,}")
    print(f"Average complexity: {graph_data['metadata']['avg_complexity']:.1f}")
    
    # Generate visualization
    generate_visualization_html(graph_data)
    print("\nVisualization complete! Open 'codebase_visualization.html' in your browser.")