#!/usr/bin/env python3
"""
Better Codebase Analyzer for HAAM
=================================
Properly detects imports including relative imports, from imports, and __init__.py exports.
"""

import ast
import os
import json
import argparse
import webbrowser
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

class BetterCodebaseAnalyzer:
    """Improved analyzer that catches more import patterns."""
    
    def __init__(self, root_path=".", package_name=None):
        self.root_path = Path(root_path).resolve()
        self.package_name = package_name or self.root_path.name
        self.nodes = []
        self.edges = []
        self.edge_set = set()  # To avoid duplicates
        self.stats = defaultdict(int)
        self.module_map = {}  # Maps module paths to node IDs
        
    def analyze(self):
        """Analyze the codebase with better import detection."""
        print(f"Analyzing codebase at: {self.root_path}")
        print(f"Package name: {self.package_name}")
        
        # First pass: collect all Python files and create nodes
        python_files = self.collect_python_files()
        print(f"Found {len(python_files)} Python files")
        
        # Create nodes for all files
        for filepath in python_files:
            self.create_node(filepath)
        
        # Second pass: analyze imports
        for filepath in python_files:
            self.analyze_imports(filepath)
        
        # Add statistics
        self.calculate_metrics()
        
        print(f"\nAnalysis complete:")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Edges: {len(self.edges)}")
        print(f"  Unique imports found: {len(self.edge_set)}")
        
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'stats': dict(self.stats)
        }
    
    def collect_python_files(self):
        """Collect all Python files, excluding common ignore patterns."""
        ignore_patterns = {
            '__pycache__', '.git', 'venv', 'env', '.tox',
            'build', 'dist', '.egg-info', 'node_modules'
        }
        
        python_files = []
        for root, dirs, files in os.walk(self.root_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_patterns]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    # Skip Colab/Jupyter files
                    if not self.is_notebook_export(filepath):
                        python_files.append(filepath)
                    else:
                        self.stats['skipped_notebook_exports'] += 1
        
        return python_files
    
    def is_notebook_export(self, filepath):
        """Check if file is a Jupyter/Colab export."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = f.read(200)
                return any(marker in first_lines for marker in [
                    '# coding: utf-8',
                    'get_ipython()',
                    '# coding=utf-8',
                    '#!/usr/bin/env python\n# coding:',
                ])
        except:
            return False
    
    def create_node(self, filepath):
        """Create a node for a Python file."""
        relative_path = filepath.relative_to(self.root_path)
        node_id = self.path_to_node_id(relative_path)
        
        # Map the file path to node ID for import resolution
        module_path = self.path_to_module_path(relative_path)
        self.module_map[module_path] = node_id
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = len(content.splitlines())
            size = len(content.encode('utf-8'))
            
            # Parse AST for metrics
            try:
                tree = ast.parse(content)
                functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                
                # Calculate complexity
                complexity = 1
                for node in ast.walk(tree):
                    if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
            except:
                functions = 0
                classes = 0
                complexity = 0
            
            # Determine node type and color
            node_type, color = self.determine_node_type(filepath, relative_path)
            
            node = {
                'data': {
                    'id': node_id,
                    'label': filepath.name,
                    'path': str(relative_path),
                    'module_path': module_path,
                    'type': node_type,
                    'lines': lines,
                    'size': size,
                    'functions': functions,
                    'classes': classes,
                    'complexity': complexity,
                    'color': color
                }
            }
            
            self.nodes.append(node)
            self.stats[f'{node_type}_count'] += 1
            self.stats['total_lines'] += lines
            
        except Exception as e:
            print(f"Error creating node for {filepath}: {e}")
    
    def path_to_node_id(self, path):
        """Convert file path to node ID."""
        return str(path).replace('/', '_').replace('\\', '_').replace('.py', '')
    
    def path_to_module_path(self, path):
        """Convert file path to Python module path."""
        parts = list(path.parts)
        if parts[-1] == '__init__.py':
            # For __init__.py, the module is the parent directory
            return '.'.join(parts[:-1])
        else:
            # For regular files, remove .py extension
            parts[-1] = parts[-1].replace('.py', '')
            return '.'.join(parts)
    
    def determine_node_type(self, filepath, relative_path):
        """Determine node type and color based on file location and name."""
        if filepath.name == '__init__.py':
            return 'package', '#9b59b6'
        elif filepath.name.startswith('test_') or 'test' in filepath.parts:
            return 'test', '#f39c12'
        elif filepath.name == 'setup.py':
            return 'setup', '#e74c3c'
        elif 'example' in str(relative_path).lower():
            return 'example', '#95a5a6'
        elif len(relative_path.parts) > 1:  # In a subdirectory
            return 'core', '#3498db'
        else:
            return 'script', '#2ecc71'
    
    def analyze_imports(self, filepath):
        """Analyze imports in a Python file with comprehensive detection."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            source_path = filepath.relative_to(self.root_path)
            source_id = self.path_to_node_id(source_path)
            source_dir = source_path.parent
            
            # Process all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # import module1, module2
                    for alias in node.names:
                        self.process_import(source_id, source_dir, alias.name, 'import')
                        
                elif isinstance(node, ast.ImportFrom):
                    # from module import name1, name2
                    if node.module:
                        # Absolute import
                        self.process_import_from(source_id, source_dir, node.module, node.level, 
                                               [n.name for n in node.names])
                    elif node.level > 0:
                        # Relative import like "from . import something"
                        self.process_relative_import(source_id, source_dir, node.level,
                                                   [n.name for n in node.names])
        
        except Exception as e:
            print(f"Error analyzing imports in {filepath}: {e}")
    
    def process_import(self, source_id, source_dir, module_name, import_type):
        """Process a regular import statement."""
        # Check if it's an internal module
        if self.is_internal_module(module_name):
            target_id = self.resolve_module_to_node_id(module_name, source_dir)
            if target_id and target_id != source_id:
                self.add_edge(source_id, target_id, import_type, module_name)
    
    def process_import_from(self, source_id, source_dir, module_name, level, names):
        """Process a 'from module import names' statement."""
        if level == 0:
            # Absolute import
            if self.is_internal_module(module_name):
                target_id = self.resolve_module_to_node_id(module_name, source_dir)
                if target_id and target_id != source_id:
                    self.add_edge(source_id, target_id, 'from_import', 
                                f"from {module_name} import {', '.join(names[:3])}")
        else:
            # Relative import
            self.process_relative_import_with_module(source_id, source_dir, level, module_name, names)
    
    def process_relative_import(self, source_id, source_dir, level, names):
        """Process relative imports like 'from . import something'."""
        # Go up 'level' directories
        target_dir = source_dir
        for _ in range(level - 1):
            target_dir = target_dir.parent
        
        # Look for the imported names
        for name in names:
            # Check if it's a file
            target_file = target_dir / f"{name}.py"
            if target_file.exists():
                target_path = target_file.relative_to(self.root_path)
                target_id = self.path_to_node_id(target_path)
                if target_id in {n['data']['id'] for n in self.nodes}:
                    self.add_edge(source_id, target_id, 'relative_import', f"from . import {name}")
            
            # Check if it's a package
            target_package = target_dir / name / "__init__.py"
            if target_package.exists():
                target_path = target_package.relative_to(self.root_path)
                target_id = self.path_to_node_id(target_path)
                if target_id in {n['data']['id'] for n in self.nodes}:
                    self.add_edge(source_id, target_id, 'relative_import', f"from . import {name}")
    
    def process_relative_import_with_module(self, source_id, source_dir, level, module_name, names):
        """Process relative imports like 'from ..module import something'."""
        # Go up 'level' directories
        target_dir = source_dir
        for _ in range(level):
            target_dir = target_dir.parent
        
        # Resolve the module path
        module_parts = module_name.split('.')
        for part in module_parts:
            target_dir = target_dir / part
        
        # Check if it's a file
        target_file = target_dir.with_suffix('.py')
        if target_file.exists():
            target_path = target_file.relative_to(self.root_path)
            target_id = self.path_to_node_id(target_path)
            if target_id in {n['data']['id'] for n in self.nodes}:
                self.add_edge(source_id, target_id, 'relative_import', 
                            f"from {'.' * level}{module_name} import {', '.join(names[:3])}")
            return
        
        # Check if it's a package
        target_package = target_dir / "__init__.py"
        if target_package.exists():
            target_path = target_package.relative_to(self.root_path)
            target_id = self.path_to_node_id(target_path)
            if target_id in {n['data']['id'] for n in self.nodes}:
                self.add_edge(source_id, target_id, 'relative_import',
                            f"from {'.' * level}{module_name} import {', '.join(names[:3])}")
    
    def is_internal_module(self, module_name):
        """Check if a module is internal to the project."""
        if not module_name:
            return False
        
        # Check if it starts with the package name
        if module_name.startswith(self.package_name):
            return True
        
        # Check if it's in our module map
        if module_name in self.module_map:
            return True
        
        # Check partial matches (e.g., 'haam.topics' when we have 'haam/haam_topics.py')
        for module_path in self.module_map:
            if module_path.endswith(module_name) or module_name.endswith(module_path):
                return True
        
        return False
    
    def resolve_module_to_node_id(self, module_name, source_dir):
        """Resolve a module name to a node ID."""
        # Direct match
        if module_name in self.module_map:
            return self.module_map[module_name]
        
        # Try with package prefix
        if not module_name.startswith(self.package_name):
            prefixed = f"{self.package_name}.{module_name}"
            if prefixed in self.module_map:
                return self.module_map[prefixed]
        
        # Try to resolve file paths
        parts = module_name.split('.')
        
        # Check from root
        possible_paths = [
            Path(*parts[1:]) / "__init__.py" if len(parts) > 1 else None,
            Path(*parts[1:]).with_suffix('.py') if len(parts) > 1 else None,
            Path(*parts) / "__init__.py",
            Path(*parts).with_suffix('.py'),
            Path(self.package_name) / Path(*parts[1:]).with_suffix('.py') if len(parts) > 1 else None,
        ]
        
        for path in possible_paths:
            if path:
                module_path = '.'.join(path.parts).replace('.py', '').replace('.__init__', '')
                if module_path in self.module_map:
                    return self.module_map[module_path]
        
        # Special handling for haam package
        if module_name.startswith('haam.'):
            # Try haam/haam_modulename.py pattern
            module_suffix = module_name.split('.')[-1]
            test_path = f"haam.haam_{module_suffix}"
            if test_path in self.module_map:
                return self.module_map[test_path]
        
        return None
    
    def add_edge(self, source_id, target_id, edge_type, label):
        """Add an edge if it doesn't exist."""
        edge_key = (source_id, target_id)
        if edge_key not in self.edge_set:
            self.edge_set.add(edge_key)
            edge = {
                'data': {
                    'id': f"{source_id}_to_{target_id}",
                    'source': source_id,
                    'target': target_id,
                    'type': edge_type,
                    'label': label
                }
            }
            self.edges.append(edge)
            self.stats['total_imports'] += 1
            self.stats[f'{edge_type}_count'] += 1
    
    def calculate_metrics(self):
        """Calculate graph metrics."""
        # In/out degree
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for edge in self.edges:
            out_degree[edge['data']['source']] += 1
            in_degree[edge['data']['target']] += 1
        
        # Update nodes with degree info
        for node in self.nodes:
            node_id = node['data']['id']
            node['data']['in_degree'] = in_degree.get(node_id, 0)
            node['data']['out_degree'] = out_degree.get(node_id, 0)
            node['data']['total_degree'] = node['data']['in_degree'] + node['data']['out_degree']
        
        # Calculate stats
        orphan_nodes = sum(1 for n in self.nodes if n['data']['total_degree'] == 0)
        hub_nodes = sum(1 for n in self.nodes if n['data']['total_degree'] >= 5)
        
        self.stats.update({
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'orphan_nodes': orphan_nodes,
            'hub_nodes': hub_nodes,
            'avg_degree': sum(n['data']['total_degree'] for n in self.nodes) / len(self.nodes) if self.nodes else 0
        })


def generate_visualization(graph_data, output_file='codebase_viz_better.html'):
    """Generate HTML visualization."""
    
    template = r'''<!DOCTYPE html>
<html>
<head>
    <title>HAAM Codebase Structure - Complete View</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #1a1a1a;
            color: #e0e0e0;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #cy {{
            flex: 1;
            background: #0d0d0d;
        }}
        #sidebar {{
            width: 400px;
            padding: 20px;
            background: #1e1e1e;
            overflow-y: auto;
            box-shadow: -2px 0 10px rgba(0,0,0,0.5);
        }
        h1 {{
            color: #3498db;
            font-size: 24px;
            margin-bottom: 20px;
        }
        .stats {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 14px;
        }
        .stat-label {
            color: #999;
        }
        .stat-value {
            font-weight: bold;
            color: #3498db;
        }
        button {
            display: inline-block;
            padding: 8px 16px;
            margin: 2px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        button:hover {
            background: #2980b9;
            transform: translateY(-1px);
        }
        .node-info {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border: 1px solid #333;
            min-height: 200px;
        }
        .legend {
            margin-top: 20px;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 13px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 4px;
            border: 1px solid #444;
        }
        #search-box {
            width: 100%;
            padding: 10px;
            margin-bottom: 15px;
            background: #2a2a2a;
            border: 1px solid #444;
            color: white;
            border-radius: 4px;
        }
        .controls {
            margin-bottom: 20px;
        }
        .controls h3 {
            margin-top: 20px;
            margin-bottom: 10px;
            color: #fff;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
</head>
<body>
    <div id="container">
        <div id="cy"></div>
        <div id="sidebar">
            <h1>🏗️ HAAM Codebase Map</h1>
            <p>Generated: {timestamp}</p>
            
            <input type="text" id="search-box" placeholder="Search files..." />
            
            <div class="stats">
                <h3>📊 Statistics</h3>
                <div class="stat-row">
                    <span class="stat-label">Total Files:</span>
                    <span class="stat-value">{total_nodes}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Lines:</span>
                    <span class="stat-value">{total_lines:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Import Links:</span>
                    <span class="stat-value">{total_edges}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Orphan Files:</span>
                    <span class="stat-value">{orphan_nodes}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Hub Files:</span>
                    <span class="stat-value">{hub_nodes}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Avg Connections:</span>
                    <span class="stat-value">{avg_degree:.1f}</span>
                </div>
            </div>
            
            <div class="controls">
                <h3>📐 Layout</h3>
                <button onclick="applyLayout('cose')">🌐 Force-Directed</button>
                <button onclick="applyLayout('breadthfirst')">🌳 Hierarchy</button>
                <button onclick="applyLayout('circle')">⭕ Circle</button>
                <button onclick="applyLayout('grid')">⚏ Grid</button>
                <button onclick="applyLayout('concentric')">🎯 Concentric</button>
                
                <h3>🎯 Highlight</h3>
                <button onclick="highlightType('core')">Core Modules</button>
                <button onclick="highlightType('test')">Test Files</button>
                <button onclick="highlightType('package')">Package Files</button>
                <button onclick="highlightOrphans()">Orphan Files</button>
                <button onclick="highlightHubs()">Hub Files</button>
                <button onclick="resetHighlight()">Reset All</button>
                
                <h3>🎮 View</h3>
                <button onclick="cy.fit()">Fit to Screen</button>
                <button onclick="toggleEdges()">Toggle Edges</button>
                <button onclick="exportPNG()">Export PNG</button>
            </div>
            
            <div class="node-info" id="node-info">
                <h3>📄 Node Details</h3>
                <p style="color: #666;">Click on a node to see details...</p>
            </div>
            
            <div class="legend">
                <h3>🎨 Legend</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background:#3498db;"></div>
                    <span>Core Package</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#2ecc71;"></div>
                    <span>Scripts</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#f39c12;"></div>
                    <span>Tests</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#9b59b6;"></div>
                    <span>Package Init</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#e74c3c;"></div>
                    <span>Setup</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const graphData = {graph_data_json};
        
        console.log('Loading graph with', graphData.nodes.length, 'nodes and', graphData.edges.length, 'edges');
        
        // Initialize Cytoscape
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {{
                nodes: graphData.nodes,
                edges: graphData.edges
            }},
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'background-color': 'data(color)',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'font-size': '11px',
                        'color': '#fff',
                        'text-outline-width': 2,
                        'text-outline-color': '#000',
                        'width': ele => Math.max(30, Math.min(100, Math.sqrt(ele.data('lines') || 50) * 4)),
                        'height': ele => Math.max(30, Math.min(100, Math.sqrt(ele.data('lines') || 50) * 4)),
                        'border-width': 2,
                        'border-color': '#000'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#666',
                        'target-arrow-color': '#666',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '8px',
                        'text-rotation': 'autorotate',
                        'text-margin-y': -10,
                        'color': '#999',
                        'text-outline-width': 1,
                        'text-outline-color': '#000'
                    }}
                }},
                {{
                    selector: ':selected',
                    style: {{
                        'background-color': '#fff',
                        'line-color': '#fff',
                        'target-arrow-color': '#fff',
                        'border-color': '#3498db',
                        'border-width': 4
                    }}
                }},
                {{
                    selector: '.highlighted',
                    style: {{
                        'background-color': '#f1c40f',
                        'border-color': '#f39c12',
                        'z-index': 999
                    }}
                }},
                {{
                    selector: '.dimmed',
                    style: {{
                        'opacity': 0.2
                    }}
                }},
                {{
                    selector: '.hidden-edge',
                    style: {{
                        'display': 'none'
                    }}
                }}
            ],
            layout: {{
                name: 'cose',
                animate: true,
                animationDuration: 1000,
                nodeRepulsion: 8000,
                idealEdgeLength: 100,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            }},
            wheelSensitivity: 0.2
        }});
        
        // Event handlers
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            
            const imports = cy.edges(`[source = "${{data.id}}"]`).length;
            const importedBy = cy.edges(`[target = "${{data.id}}"]`).length;
            
            document.getElementById('node-info').innerHTML = `
                <h3>📄 ${{data.label}}</h3>
                <p><strong>Type:</strong> ${{data.type}}</p>
                <p><strong>Path:</strong> ${{data.path}}</p>
                <p><strong>Module:</strong> ${{data.module_path || 'N/A'}}</p>
                <p><strong>Lines:</strong> ${{data.lines || 0}}</p>
                <p><strong>Functions:</strong> ${{data.functions || 0}}</p>
                <p><strong>Classes:</strong> ${{data.classes || 0}}</p>
                <p><strong>Complexity:</strong> ${{data.complexity || 0}}</p>
                <p><strong>Imports:</strong> ${{imports}} modules</p>
                <p><strong>Imported by:</strong> ${{importedBy}} modules</p>
                <p><strong>Total connections:</strong> ${{data.total_degree || 0}}</p>
            `;
        }});
        
        // Search functionality
        document.getElementById('search-box').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            
            cy.elements().removeClass('highlighted dimmed');
            
            if (searchTerm) {{
                cy.elements().addClass('dimmed');
                cy.nodes().forEach(node => {{
                    if (node.data('label').toLowerCase().includes(searchTerm) ||
                        node.data('path').toLowerCase().includes(searchTerm)) {{
                        node.removeClass('dimmed').addClass('highlighted');
                        node.connectedEdges().removeClass('dimmed');
                    }}
                }});
            }}
        }});
        
        // Layout functions
        function applyLayout(layoutName) {{
            const layouts = {{
                'cose': {{
                    name: 'cose',
                    animate: true,
                    nodeRepulsion: 8000,
                    idealEdgeLength: 100
                }},
                'breadthfirst': {{
                    name: 'breadthfirst',
                    directed: true,
                    spacingFactor: 1.5,
                    animate: true
                }},
                'circle': {{
                    name: 'circle',
                    animate: true
                }},
                'grid': {{
                    name: 'grid',
                    animate: true,
                    padding: 30
                }},
                'concentric': {{
                    name: 'concentric',
                    animate: true,
                    concentric: function(node) {{
                        return node.degree();
                    }},
                    levelWidth: function() {{ return 1; }}
                }}
            }};
            
            cy.layout(layouts[layoutName]).run();
        }}
        
        // Highlight functions
        function highlightType(type) {{
            cy.elements().removeClass('highlighted dimmed');
            cy.elements().addClass('dimmed');
            cy.nodes(`[type = "${{type}}"]`).removeClass('dimmed').addClass('highlighted');
            cy.nodes(`[type = "${{type}}"]`).connectedEdges().removeClass('dimmed');
        }}
        
        function highlightOrphans() {{
            cy.elements().removeClass('highlighted dimmed');
            cy.elements().addClass('dimmed');
            cy.nodes().filter(n => n.degree() === 0).removeClass('dimmed').addClass('highlighted');
        }}
        
        function highlightHubs() {{
            cy.elements().removeClass('highlighted dimmed');
            cy.elements().addClass('dimmed');
            cy.nodes().filter(n => n.degree() >= 5).removeClass('dimmed').addClass('highlighted');
            cy.nodes().filter(n => n.degree() >= 5).connectedEdges().removeClass('dimmed');
        }}
        
        function resetHighlight() {{
            cy.elements().removeClass('highlighted dimmed');
        }}
        
        let edgesVisible = true;
        function toggleEdges() {{
            edgesVisible = !edgesVisible;
            if (edgesVisible) {{
                cy.edges().removeClass('hidden-edge');
            }} else {{
                cy.edges().addClass('hidden-edge');
            }}
        }}
        
        function exportPNG() {{
            const png64 = cy.png({{
                output: 'blob',
                bg: '#0d0d0d',
                scale: 2
            }});
            
            const a = document.createElement('a');
            a.href = URL.createObjectURL(png64);
            a.download = 'haam_codebase_map.png';
            a.click();
        }}
        
        // Initial setup
        cy.ready(function() {{
            console.log('Cytoscape ready with', cy.nodes().length, 'nodes and', cy.edges().length, 'edges');
            cy.fit();
        }});
    </script>
</body>
</html>'''
    
    # Fill template
    html = template.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        graph_data_json=json.dumps(graph_data),
        total_nodes=graph_data['stats'].get('total_nodes', 0),
        total_lines=graph_data['stats'].get('total_lines', 0),
        total_edges=graph_data['stats'].get('total_edges', 0),
        orphan_nodes=graph_data['stats'].get('orphan_nodes', 0),
        hub_nodes=graph_data['stats'].get('hub_nodes', 0),
        avg_degree=graph_data['stats'].get('avg_degree', 0)
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Better codebase analyzer with comprehensive import detection')
    parser.add_argument('path', nargs='?', default='.', help='Path to analyze')
    parser.add_argument('-p', '--package', help='Package name (default: directory name)')
    parser.add_argument('-o', '--output', default='codebase_analysis.html', help='Output file')
    parser.add_argument('--open', action='store_true', help='Open in browser')
    
    args = parser.parse_args()
    
    # Analyze
    analyzer = BetterCodebaseAnalyzer(args.path, args.package)
    graph_data = analyzer.analyze()
    
    # Generate visualization
    output_file = generate_visualization(graph_data, args.output)
    print(f"\nVisualization saved to: {output_file}")
    
    # Save data
    json_file = args.output.replace('.html', '_data.json')
    with open(json_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"Data saved to: {json_file}")
    
    # Print import types found
    print("\nImport types found:")
    for key, value in graph_data['stats'].items():
        if key.endswith('_count') and 'import' in key:
            print(f"  {key}: {value}")
    
    if args.open:
        webbrowser.open(f'file://{Path(output_file).absolute()}')


if __name__ == '__main__':
    main()