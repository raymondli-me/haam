#!/usr/bin/env python3
"""
Complete Codebase Import Analyzer
==================================
Properly detects ALL import patterns including relative imports, from imports, and __init__.py exports.
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


class CompleteCodebaseAnalyzer:
    """Comprehensive analyzer for Python codebases with full import detection."""
    
    def __init__(self, root_path=".", package_name=None):
        self.root_path = Path(root_path).resolve()
        self.package_name = package_name or self.root_path.name
        self.nodes = []
        self.edges = []
        self.edge_set = set()  # Avoid duplicate edges
        self.stats = defaultdict(int)
        self.module_map = {}  # Maps module paths to node IDs
        self.file_map = {}  # Maps file paths to node IDs
        
    def analyze(self):
        """Analyze the entire codebase with comprehensive import detection."""
        print(f"\nAnalyzing codebase at: {self.root_path}")
        print(f"Package name: {self.package_name}")
        print("-" * 50)
        
        # Collect all Python files
        python_files = self.collect_python_files()
        print(f"Found {len(python_files)} Python files")
        
        # First pass: Create nodes for all files
        for filepath in python_files:
            self.create_node(filepath)
        
        # Second pass: Analyze imports
        print("\nAnalyzing imports...")
        for filepath in python_files:
            self.analyze_file_imports(filepath)
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"  Total files: {len(self.nodes)}")
        print(f"  Total imports: {len(self.edges)}")
        print(f"  Orphan files: {self.stats['orphan_nodes']}")
        print(f"  Hub files: {self.stats['hub_nodes']}")
        
        # Print import breakdown
        print("\nImport types found:")
        for key, value in sorted(self.stats.items()):
            if 'import' in key.lower() and value > 0:
                print(f"  {key}: {value}")
        
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'stats': dict(self.stats)
        }
    
    def collect_python_files(self):
        """Collect all Python files, excluding common ignore patterns."""
        ignore_dirs = {
            '__pycache__', '.git', 'venv', 'env', '.venv',
            'build', 'dist', '.egg-info', 'node_modules',
            '.tox', '.pytest_cache', '.mypy_cache'
        }
        
        python_files = []
        for root, dirs, files in os.walk(self.root_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.endswith('.egg-info')]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    # Skip obvious notebook exports
                    if not self.is_notebook_export(filepath):
                        python_files.append(filepath)
        
        return sorted(python_files)
    
    def is_notebook_export(self, filepath):
        """Check if file is a Jupyter/Colab export."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = f.read(500)
                # Look for Jupyter/Colab markers
                markers = [
                    '# -*- coding: utf-8 -*-',
                    'get_ipython()',
                    '"""# ',  # Common Colab section header
                    'from google.colab import',
                    '!pip install',
                    '!apt-get'
                ]
                return any(marker in first_lines for marker in markers)
        except:
            return False
    
    def create_node(self, filepath):
        """Create a node for a Python file with full metrics."""
        relative_path = filepath.relative_to(self.root_path)
        node_id = str(relative_path).replace(os.sep, '_').replace('.py', '')
        
        # Store mappings for import resolution
        self.file_map[str(relative_path)] = node_id
        
        # Create module path for import matching
        parts = list(relative_path.parts)
        if parts[-1] == '__init__.py':
            module_path = '.'.join(parts[:-1])
        else:
            parts[-1] = parts[-1].replace('.py', '')
            module_path = '.'.join(parts)
        
        self.module_map[module_path] = node_id
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = len(content.splitlines())
            
            # Parse AST for metrics
            try:
                tree = ast.parse(content)
                functions = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
                classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
                
                # Calculate cyclomatic complexity
                complexity = 1
                for node in ast.walk(tree):
                    if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(node, ast.BoolOp):
                        complexity += len(node.values) - 1
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
                    'functions': functions,
                    'classes': classes,
                    'complexity': complexity,
                    'color': color
                }
            }
            
            self.nodes.append(node)
            
            # Update stats
            self.stats['total_lines'] += lines
            self.stats['total_functions'] += functions
            self.stats['total_classes'] += classes
            self.stats[f'{node_type}_count'] += 1
            
        except Exception as e:
            print(f"  Warning: Error creating node for {filepath}: {e}")
    
    def determine_node_type(self, filepath, relative_path):
        """Determine node type and color based on file characteristics."""
        name = filepath.name
        path_str = str(relative_path).lower()
        
        if name == '__init__.py':
            return 'package', '#9b59b6'
        elif name.startswith('test_') or '/test' in path_str or 'test/' in path_str:
            return 'test', '#f39c12'
        elif name == 'setup.py':
            return 'setup', '#e74c3c'
        elif 'example' in path_str or 'demo' in path_str:
            return 'example', '#95a5a6'
        elif len(relative_path.parts) > 1:
            return 'module', '#3498db'
        else:
            return 'script', '#2ecc71'
    
    def analyze_file_imports(self, filepath):
        """Analyze all imports in a Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            relative_path = filepath.relative_to(self.root_path)
            source_id = self.file_map[str(relative_path)]
            source_dir = filepath.parent
            
            # Walk the AST to find all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # Handle: import module1, module2
                    for alias in node.names:
                        self.process_import(source_id, source_dir, alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    # Handle: from module import name1, name2
                    if node.level == 0 and node.module:
                        # Absolute import
                        self.process_from_import(source_id, source_dir, node.module, 
                                                [n.name for n in node.names])
                    elif node.level > 0:
                        # Relative import
                        self.process_relative_import(source_id, source_dir, node.level,
                                                    node.module, [n.name for n in node.names])
        
        except Exception as e:
            print(f"  Warning: Error analyzing {filepath}: {e}")
    
    def process_import(self, source_id, source_dir, module_name):
        """Process: import module_name"""
        target_id = self.resolve_import(source_dir, module_name)
        if target_id and target_id != source_id:
            self.add_edge(source_id, target_id, 'import')
            self.stats['standard_imports'] += 1
    
    def process_from_import(self, source_id, source_dir, module_name, names):
        """Process: from module_name import names"""
        target_id = self.resolve_import(source_dir, module_name)
        if target_id and target_id != source_id:
            self.add_edge(source_id, target_id, 'from_import')
            self.stats['from_imports'] += 1
    
    def process_relative_import(self, source_id, source_dir, level, module, names):
        """Process: from . import something or from ..module import something"""
        # Calculate the base directory for the relative import
        base_dir = source_dir
        for _ in range(level - 1):
            base_dir = base_dir.parent
        
        if module:
            # from .module import something or from ..module import something
            target_path = base_dir / module.replace('.', os.sep)
            
            # Try as a Python file
            if (target_path.with_suffix('.py')).exists():
                target_file = target_path.with_suffix('.py')
            # Try as a package
            elif (target_path / '__init__.py').exists():
                target_file = target_path / '__init__.py'
            else:
                return
            
            relative_path = target_file.relative_to(self.root_path)
            target_id = self.file_map.get(str(relative_path))
            if target_id and target_id != source_id:
                self.add_edge(source_id, target_id, 'relative_import')
                self.stats['relative_imports'] += 1
        else:
            # from . import something
            for name in names:
                # Try as a file in the same directory
                target_file = base_dir / f"{name}.py"
                if not target_file.exists():
                    # Try as a package
                    target_file = base_dir / name / '__init__.py'
                
                if target_file.exists():
                    relative_path = target_file.relative_to(self.root_path)
                    target_id = self.file_map.get(str(relative_path))
                    if target_id and target_id != source_id:
                        self.add_edge(source_id, target_id, 'relative_import')
                        self.stats['relative_imports'] += 1
    
    def resolve_import(self, source_dir, module_name):
        """Resolve a module name to a node ID."""
        # Skip standard library and external packages
        if self.is_standard_or_external(module_name):
            return None
        
        # Check if it's a direct module in our map
        if module_name in self.module_map:
            return self.module_map[module_name]
        
        # Check if it starts with our package name
        if module_name.startswith(f"{self.package_name}."):
            # Try to find it in our module map
            if module_name in self.module_map:
                return self.module_map[module_name]
            
            # Try without package prefix
            without_prefix = module_name[len(self.package_name) + 1:]
            if without_prefix in self.module_map:
                return self.module_map[without_prefix]
        
        # Try to resolve as a file path
        parts = module_name.split('.')
        
        # Special handling for haam package patterns
        if self.package_name == 'haam' and module_name.startswith('haam'):
            # Try haam.haam_module pattern
            if len(parts) == 2 and parts[0] == 'haam':
                test_module = f"haam.haam_{parts[1]}"
                if test_module in self.module_map:
                    return self.module_map[test_module]
                
                # Also try just haam_module
                test_module = f"haam_{parts[1]}"
                if test_module in self.module_map:
                    return self.module_map[test_module]
        
        # Try to find the module in various forms
        for i in range(len(parts)):
            test_path = '.'.join(parts[:i+1])
            if test_path in self.module_map:
                return self.module_map[test_path]
        
        return None
    
    def is_standard_or_external(self, module_name):
        """Check if module is standard library or external package."""
        if not module_name:
            return True
        
        # Common standard library modules
        stdlib = {
            'os', 'sys', 'io', 'time', 'datetime', 'json', 'csv', 're', 'math',
            'random', 'collections', 'itertools', 'functools', 'typing', 'ast',
            'pathlib', 'subprocess', 'threading', 'multiprocessing', 'queue',
            'unittest', 'doctest', 'pdb', 'warnings', 'copy', 'pickle', 'shelve',
            'sqlite3', 'hashlib', 'hmac', 'secrets', 'urllib', 'http', 'email',
            'argparse', 'logging', 'configparser', 'tempfile', 'shutil', 'glob',
            'fnmatch', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'base64'
        }
        
        # Common external packages
        external = {
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'tensorflow',
            'torch', 'keras', 'pytest', 'requests', 'flask', 'django', 'fastapi',
            'sqlalchemy', 'celery', 'redis', 'boto3', 'pillow', 'opencv', 'nltk',
            'spacy', 'transformers', 'datasets', 'accelerate', 'tqdm', 'rich',
            'click', 'typer', 'pydantic', 'jinja2', 'yaml', 'toml', 'dotenv',
            'seaborn', 'plotly', 'bokeh', 'streamlit', 'gradio', 'jupyterlab'
        }
        
        first_part = module_name.split('.')[0]
        return first_part in stdlib or first_part in external
    
    def add_edge(self, source_id, target_id, edge_type):
        """Add an edge between two nodes."""
        edge_key = (source_id, target_id, edge_type)
        if edge_key not in self.edge_set:
            self.edge_set.add(edge_key)
            
            edge = {
                'data': {
                    'id': f"{source_id}_to_{target_id}_{edge_type}",
                    'source': source_id,
                    'target': target_id,
                    'type': edge_type
                }
            }
            
            self.edges.append(edge)
    
    def calculate_metrics(self):
        """Calculate graph-level metrics."""
        # Calculate degree for each node
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for edge in self.edges:
            out_degree[edge['data']['source']] += 1
            in_degree[edge['data']['target']] += 1
        
        # Update nodes with degree information
        for node in self.nodes:
            node_id = node['data']['id']
            node['data']['in_degree'] = in_degree.get(node_id, 0)
            node['data']['out_degree'] = out_degree.get(node_id, 0)
            node['data']['total_degree'] = node['data']['in_degree'] + node['data']['out_degree']
        
        # Calculate statistics
        self.stats['total_nodes'] = len(self.nodes)
        self.stats['total_edges'] = len(self.edges)
        self.stats['orphan_nodes'] = sum(1 for n in self.nodes if n['data']['total_degree'] == 0)
        self.stats['hub_nodes'] = sum(1 for n in self.nodes if n['data']['total_degree'] >= 5)
        
        if self.nodes:
            self.stats['avg_degree'] = sum(n['data']['total_degree'] for n in self.nodes) / len(self.nodes)
        else:
            self.stats['avg_degree'] = 0


def generate_html_visualization(graph_data, output_file='codebase_viz.html'):
    """Generate interactive HTML visualization."""
    
    # Use a template string without format placeholders to avoid issues
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Codebase Structure Analysis</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #cy {{
            flex: 1;
            background: white;
            border: 1px solid #ddd;
        }}
        #sidebar {{
            width: 380px;
            padding: 20px;
            background: #fff;
            border-left: 1px solid #ddd;
            overflow-y: auto;
            box-shadow: -2px 0 5px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            font-size: 24px;
            margin-bottom: 10px;
        }}
        .stats {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }}
        .stat-value {{
            font-weight: bold;
            color: #0066cc;
        }}
        button {{
            display: inline-block;
            padding: 8px 16px;
            margin: 2px;
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background: #0052a3;
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        .node-info {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        h3 {{
            margin-top: 20px;
            margin-bottom: 10px;
            color: #333;
        }}
        #search-box {{
            width: 100%;
            padding: 8px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 3px;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
</head>
<body>
    <div id="container">
        <div id="cy"></div>
        <div id="sidebar">
            <h1>Codebase Structure</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <input type="text" id="search-box" placeholder="Search files..." />
            
            <div class="stats">
                <h3>Statistics</h3>
                <div class="stat-row">
                    <span>Total Files:</span>
                    <span class="stat-value">{graph_data['stats'].get('total_nodes', 0)}</span>
                </div>
                <div class="stat-row">
                    <span>Total Lines:</span>
                    <span class="stat-value">{graph_data['stats'].get('total_lines', 0):,}</span>
                </div>
                <div class="stat-row">
                    <span>Import Links:</span>
                    <span class="stat-value">{graph_data['stats'].get('total_edges', 0)}</span>
                </div>
                <div class="stat-row">
                    <span>Orphan Files:</span>
                    <span class="stat-value">{graph_data['stats'].get('orphan_nodes', 0)}</span>
                </div>
                <div class="stat-row">
                    <span>Hub Files:</span>
                    <span class="stat-value">{graph_data['stats'].get('hub_nodes', 0)}</span>
                </div>
                <div class="stat-row">
                    <span>Avg Connections:</span>
                    <span class="stat-value">{graph_data['stats'].get('avg_degree', 0):.1f}</span>
                </div>
            </div>
            
            <div class="controls">
                <h3>Layout</h3>
                <button onclick="applyLayout('cose')">Force-Directed</button>
                <button onclick="applyLayout('breadthfirst')">Hierarchy</button>
                <button onclick="applyLayout('circle')">Circle</button>
                <button onclick="applyLayout('grid')">Grid</button>
                
                <h3>Highlight</h3>
                <button onclick="highlightType('module')">Modules</button>
                <button onclick="highlightType('test')">Tests</button>
                <button onclick="highlightType('package')">Packages</button>
                <button onclick="highlightOrphans()">Orphans</button>
                <button onclick="highlightHubs()">Hubs</button>
                <button onclick="resetHighlight()">Reset</button>
            </div>
            
            <div class="node-info" id="node-info">
                <h3>Node Details</h3>
                <p>Click on a node to see details</p>
            </div>
            
            <div class="legend">
                <h3>Legend</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background:#3498db;"></div>
                    <span>Module</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#2ecc71;"></div>
                    <span>Script</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#f39c12;"></div>
                    <span>Test</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#9b59b6;"></div>
                    <span>Package</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const graphData = {json.dumps(graph_data)};
        
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
                        'width': ele => Math.max(30, Math.min(80, 30 + Math.sqrt(ele.data('lines') || 50))),
                        'height': ele => Math.max(30, Math.min(80, 30 + Math.sqrt(ele.data('lines') || 50))),
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '10px',
                        'text-outline-width': 2,
                        'text-outline-color': '#fff'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#999',
                        'target-arrow-color': '#999',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.7
                    }}
                }},
                {{
                    selector: '.highlighted',
                    style: {{
                        'background-color': '#ff6b6b',
                        'line-color': '#ff6b6b',
                        'target-arrow-color': '#ff6b6b',
                        'z-index': 999
                    }}
                }},
                {{
                    selector: '.dimmed',
                    style: {{
                        'opacity': 0.2
                    }}
                }}
            ],
            layout: {{
                name: 'cose',
                animate: true,
                animationDuration: 1000,
                nodeRepulsion: 5000,
                idealEdgeLength: 100,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            }}
        }});
        
        // Node click handler
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            
            const imports = cy.edges(`[source = "${{data.id}}"]`).length;
            const importedBy = cy.edges(`[target = "${{data.id}}"]`).length;
            
            document.getElementById('node-info').innerHTML = `
                <h3>${{data.label}}</h3>
                <p><strong>Type:</strong> ${{data.type}}</p>
                <p><strong>Path:</strong> ${{data.path}}</p>
                <p><strong>Lines:</strong> ${{data.lines || 0}}</p>
                <p><strong>Functions:</strong> ${{data.functions || 0}}</p>
                <p><strong>Classes:</strong> ${{data.classes || 0}}</p>
                <p><strong>Complexity:</strong> ${{data.complexity || 0}}</p>
                <p><strong>Imports:</strong> ${{imports}} modules</p>
                <p><strong>Imported by:</strong> ${{importedBy}} modules</p>
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
                    nodeRepulsion: 5000,
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
    </script>
</body>
</html>'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Python codebase structure and visualize imports'
    )
    parser.add_argument('path', nargs='?', default='.',
                       help='Path to analyze (default: current directory)')
    parser.add_argument('-p', '--package', 
                       help='Package name (default: directory name)')
    parser.add_argument('-o', '--output', default='codebase_analysis.html',
                       help='Output HTML file')
    parser.add_argument('--json', action='store_true',
                       help='Also save JSON data')
    parser.add_argument('--open', action='store_true',
                       help='Open in browser after generation')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = CompleteCodebaseAnalyzer(args.path, args.package)
    graph_data = analyzer.analyze()
    
    # Generate visualization
    output_file = generate_html_visualization(graph_data, args.output)
    print(f"\nVisualization saved to: {output_file}")
    
    # Save JSON if requested
    if args.json:
        json_file = args.output.replace('.html', '_data.json')
        with open(json_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"JSON data saved to: {json_file}")
    
    # Open in browser if requested
    if args.open:
        webbrowser.open(f'file://{Path(output_file).absolute()}')


if __name__ == '__main__':
    main()