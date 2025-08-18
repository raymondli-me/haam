#!/usr/bin/env python3
"""
Final HAAM Codebase Analyzer
============================
Properly detects all import patterns and generates accurate visualization.
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

class HAAMCodebaseAnalyzer:
    """Analyzes HAAM codebase with comprehensive import detection."""
    
    def __init__(self, root_path="."):
        self.root_path = Path(root_path).resolve()
        self.nodes = []
        self.edges = []
        self.stats = defaultdict(int)
        self.node_map = {}  # Maps file paths to node IDs
        
    def analyze(self):
        """Analyze the codebase."""
        print(f"Analyzing HAAM codebase at: {self.root_path}")
        
        # Collect Python files
        python_files = self.collect_python_files()
        print(f"Found {len(python_files)} Python files")
        
        # First pass: create nodes
        for filepath in python_files:
            self.create_node(filepath)
        
        # Build node map for easier lookups
        for node in self.nodes:
            self.node_map[node['data']['path']] = node['data']['id']
        
        # Second pass: analyze imports
        for filepath in python_files:
            self.analyze_file_imports(filepath)
        
        # Calculate metrics
        self.calculate_metrics()
        
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'stats': dict(self.stats)
        }
    
    def collect_python_files(self):
        """Collect all Python files."""
        python_files = []
        ignore_dirs = {'.git', '__pycache__', 'venv', 'env', '.tox', 'build', 'dist', 'node_modules'}
        
        for root, dirs, files in os.walk(self.root_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    # Skip Colab-style files
                    if not self.is_colab_file(filepath):
                        python_files.append(filepath)
                    else:
                        self.stats['skipped_colab'] += 1
        
        return sorted(python_files)
    
    def is_colab_file(self, filepath):
        """Check if file is a Colab/Jupyter export."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(300)
                return any(marker in content for marker in [
                    '!pip install',
                    'get_ipython()',
                    'from google.colab import',
                    '%%'
                ])
        except:
            return False
    
    def create_node(self, filepath):
        """Create a node for a Python file."""
        relative_path = filepath.relative_to(self.root_path)
        node_id = str(relative_path).replace('/', '_').replace('\\', '_').replace('.py', '')
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = len(content.splitlines())
            
            # Parse AST for metrics
            try:
                tree = ast.parse(content)
                functions = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
                classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
            except:
                functions = 0
                classes = 0
            
            # Determine type and color
            if filepath.name == '__init__.py':
                node_type = 'package'
                color = '#9b59b6'
            elif 'test' in filepath.name:
                node_type = 'test'
                color = '#f39c12'
            elif relative_path.parts[0] == 'haam':  # Core package files
                node_type = 'core'
                color = '#3498db'
            elif relative_path.parts[0] == 'examples':
                node_type = 'example'
                color = '#95a5a6'
            elif filepath.name == 'setup.py':
                node_type = 'setup'
                color = '#e74c3c'
            else:
                node_type = 'script'
                color = '#2ecc71'
            
            node = {
                'data': {
                    'id': node_id,
                    'label': filepath.name,
                    'path': str(relative_path),
                    'type': node_type,
                    'lines': lines,
                    'functions': functions,
                    'classes': classes,
                    'color': color
                }
            }
            
            self.nodes.append(node)
            self.stats['total_lines'] += lines
            self.stats[f'{node_type}_count'] += 1
            
        except Exception as e:
            print(f"Error creating node for {filepath}: {e}")
    
    def analyze_file_imports(self, filepath):
        """Analyze imports in a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            relative_path = filepath.relative_to(self.root_path)
            source_id = str(relative_path).replace('/', '_').replace('\\', '_').replace('.py', '')
            
            # Walk through AST to find imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.process_import(source_id, alias.name, filepath)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.process_from_import(source_id, node.module, node.level, filepath)
                        # Debug relative imports
                        if node.level > 0:
                            print(f"  Relative import in {filepath.name}: from {'.' * node.level}{node.module} import ...")
                    elif node.level > 0:
                        # Relative import without module (from . import x)
                        self.process_relative_import(source_id, node.level, filepath, node.names)
        
        except Exception as e:
            if "invalid syntax" not in str(e):
                print(f"Error analyzing imports in {filepath}: {e}")
    
    def process_import(self, source_id, module_name, source_file):
        """Process 'import module' statements."""
        # Check if it's a HAAM module
        if module_name.startswith('haam'):
            target_id = self.find_module_node(module_name, source_file)
            if target_id and target_id != source_id:
                self.add_edge(source_id, target_id, f"import {module_name}")
    
    def process_from_import(self, source_id, module_name, level, source_file):
        """Process 'from module import ...' statements."""
        if level == 0:  # Absolute import
            if module_name.startswith('haam'):
                target_id = self.find_module_node(module_name, source_file)
                if target_id and target_id != source_id:
                    self.add_edge(source_id, target_id, f"from {module_name} import ...")
        elif level == 1 and module_name:  # from .module import (common in HAAM)
            # This is a relative import within the same package
            current_dir = source_file.parent
            
            # Try to find the module in the same directory
            candidates = [
                current_dir / f"{module_name}.py",
                current_dir / module_name / "__init__.py"
            ]
            
            for candidate in candidates:
                if candidate.exists():
                    rel_path = candidate.relative_to(self.root_path)
                    target_id = str(rel_path).replace('/', '_').replace('\\', '_').replace('.py', '')
                    if target_id in self.node_map.values() and target_id != source_id:
                        self.add_edge(source_id, target_id, f"from .{module_name} import ...")
                        break
        else:  # Relative import with higher levels
            # Calculate the target directory based on level
            current_dir = source_file.parent
            for _ in range(level):
                current_dir = current_dir.parent
            
            if module_name:
                # from ..module import x
                module_parts = module_name.split('.')
                target_path = current_dir
                for part in module_parts:
                    target_path = target_path / part
                
                # Try both .py file and __init__.py
                for candidate in [target_path.with_suffix('.py'), target_path / '__init__.py']:
                    if candidate.exists():
                        rel_path = candidate.relative_to(self.root_path)
                        target_id = str(rel_path).replace('/', '_').replace('\\', '_').replace('.py', '')
                        if target_id in self.node_map.values() and target_id != source_id:
                            self.add_edge(source_id, target_id, f"from {'.' * level}{module_name} import ...")
                            break
    
    def process_relative_import(self, source_id, level, source_file, names):
        """Process 'from . import x, y, z' statements."""
        current_dir = source_file.parent
        for _ in range(level - 1):
            current_dir = current_dir.parent
        
        for name_node in names:
            name = name_node.name
            # Try to find the target
            for candidate in [current_dir / f"{name}.py", current_dir / name / "__init__.py"]:
                if candidate.exists():
                    rel_path = candidate.relative_to(self.root_path)
                    target_id = str(rel_path).replace('/', '_').replace('\\', '_').replace('.py', '')
                    if target_id in self.node_map.values() and target_id != source_id:
                        self.add_edge(source_id, target_id, f"from {'.' * level} import {name}")
                        break
    
    def find_module_node(self, module_name, source_file):
        """Find the node ID for a module."""
        # Direct mapping attempts
        if module_name == 'haam':
            # Import of main package
            for path, node_id in self.node_map.items():
                if path == 'haam/__init__.py':
                    return node_id
        
        # Try to map module parts to file paths
        parts = module_name.split('.')
        
        # For haam.module_name, try haam/module_name.py and haam/haam_module_name.py
        if len(parts) >= 2 and parts[0] == 'haam':
            candidates = [
                Path('haam') / f"{parts[1]}.py",
                Path('haam') / f"haam_{parts[1]}.py",
                Path('haam') / parts[1] / "__init__.py"
            ]
            
            for candidate in candidates:
                rel_path = str(candidate)
                if rel_path in self.node_map:
                    return self.node_map[rel_path]
        
        return None
    
    def add_edge(self, source_id, target_id, label):
        """Add an edge between nodes."""
        edge = {
            'data': {
                'id': f"{source_id}_to_{target_id}",
                'source': source_id,
                'target': target_id,
                'label': label
            }
        }
        self.edges.append(edge)
        self.stats['total_imports'] += 1
    
    def calculate_metrics(self):
        """Calculate graph metrics."""
        # Calculate degrees
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
        
        # Stats
        self.stats['total_nodes'] = len(self.nodes)
        self.stats['total_edges'] = len(self.edges)
        self.stats['orphan_nodes'] = sum(1 for n in self.nodes if n['data']['total_degree'] == 0)
        self.stats['hub_nodes'] = sum(1 for n in self.nodes if n['data']['total_degree'] >= 5)


def generate_html_visualization(graph_data, output_file='haam_codebase.html'):
    """Generate the visualization HTML."""
    
    # Prepare the data
    nodes_json = json.dumps(graph_data['nodes'])
    edges_json = json.dumps(graph_data['edges'])
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>HAAM Codebase Structure</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        #cy {{
            width: 100vw;
            height: 100vh;
            background: #f5f5f5;
        }}
        #info {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 300px;
        }}
        button {{
            margin: 2px;
            padding: 5px 10px;
            cursor: pointer;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
        }}
        button:hover {{
            background: #2980b9;
        }}
        .stats {{
            margin-top: 10px;
            font-size: 14px;
        }}
        .node-info {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
</head>
<body>
    <div id="cy"></div>
    <div id="info">
        <h2>HAAM Codebase Map</h2>
        <div>
            <button onclick="runLayout('cose')">Force Layout</button>
            <button onclick="runLayout('breadthfirst')">Tree Layout</button>
            <button onclick="runLayout('circle')">Circle Layout</button>
            <button onclick="runLayout('grid')">Grid Layout</button>
        </div>
        <div class="stats">
            <strong>Files:</strong> {graph_data['stats']['total_nodes']}<br>
            <strong>Imports:</strong> {graph_data['stats']['total_edges']}<br>
            <strong>Lines:</strong> {graph_data['stats']['total_lines']:,}<br>
            <strong>Orphans:</strong> {graph_data['stats']['orphan_nodes']}<br>
            <strong>Hubs:</strong> {graph_data['stats']['hub_nodes']}
        </div>
        <div class="node-info" id="node-info">
            <strong>Click a node for details</strong>
        </div>
    </div>
    
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        
        console.log('Loading', nodes.length, 'nodes and', edges.length, 'edges');
        
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {{
                nodes: nodes,
                edges: edges
            }},
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'background-color': 'data(color)',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'font-size': '10px',
                        'width': ele => Math.max(30, Math.min(80, Math.sqrt(ele.data('lines') || 50) * 3)),
                        'height': ele => Math.max(30, Math.min(80, Math.sqrt(ele.data('lines') || 50) * 3))
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
                        'label': 'data(label)',
                        'font-size': '8px',
                        'text-rotation': 'autorotate'
                    }}
                }},
                {{
                    selector: ':selected',
                    style: {{
                        'background-color': '#ff0000',
                        'line-color': '#ff0000',
                        'target-arrow-color': '#ff0000'
                    }}
                }}
            ],
            layout: {{
                name: 'cose',
                animate: true
            }}
        }});
        
        // Event handlers
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            
            document.getElementById('node-info').innerHTML = `
                <strong>${{data.label}}</strong><br>
                Type: ${{data.type}}<br>
                Path: ${{data.path}}<br>
                Lines: ${{data.lines || 0}}<br>
                Imports: ${{data.out_degree || 0}}<br>
                Imported by: ${{data.in_degree || 0}}
            `;
        }});
        
        function runLayout(name) {{
            cy.layout({{ name: name, animate: true }}).run();
        }}
        
        // Fit on load
        cy.ready(() => {{
            cy.fit();
        }});
    </script>
</body>
</html>"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Analyze HAAM codebase structure')
    parser.add_argument('path', nargs='?', default='.', help='Path to analyze')
    parser.add_argument('-o', '--output', default='haam_structure.html', help='Output file')
    parser.add_argument('--open', action='store_true', help='Open in browser')
    
    args = parser.parse_args()
    
    # Analyze
    analyzer = HAAMCodebaseAnalyzer(args.path)
    graph_data = analyzer.analyze()
    
    print(f"\nAnalysis Summary:")
    print(f"  Total files: {graph_data['stats']['total_nodes']}")
    print(f"  Total imports: {graph_data['stats']['total_edges']}")
    print(f"  Core modules: {graph_data['stats'].get('core_count', 0)}")
    print(f"  Test files: {graph_data['stats'].get('test_count', 0)}")
    print(f"  Scripts: {graph_data['stats'].get('script_count', 0)}")
    
    # Generate visualization
    output_file = generate_html_visualization(graph_data, args.output)
    print(f"\nVisualization saved to: {output_file}")
    
    # Save data
    data_file = args.output.replace('.html', '_data.json')
    with open(data_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"Data saved to: {data_file}")
    
    if args.open:
        webbrowser.open(f'file://{Path(output_file).absolute()}')


if __name__ == '__main__':
    main()