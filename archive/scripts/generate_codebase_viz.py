#!/usr/bin/env python3
"""
Dynamic Codebase Visualization Generator
========================================
Analyzes Python codebase and generates interactive Cytoscape.js visualization.
This is the typical/professional way to create code visualizations.
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

class CodebaseAnalyzer:
    """Analyzes Python codebase structure for visualization."""
    
    def __init__(self, root_path=".", ignore_patterns=None):
        self.root_path = Path(root_path).resolve()
        self.ignore_patterns = ignore_patterns or [
            '__pycache__', '.git', '.pytest_cache', 'venv', 'env',
            'build', 'dist', '*.egg-info', '.tox', 'node_modules'
        ]
        self.nodes = []
        self.edges = []
        self.stats = defaultdict(int)
        
    def should_ignore(self, path):
        """Check if path should be ignored."""
        path_str = str(path)
        for pattern in self.ignore_patterns:
            if '*' in pattern:
                if re.match(pattern.replace('*', '.*'), path.name):
                    return True
            elif pattern in path_str:
                return True
        return False
    
    def is_python_file(self, filepath):
        """Check if file is a valid Python file."""
        if not filepath.suffix == '.py':
            return False
        
        # Skip Jupyter notebook exports and Colab scripts
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = f.read(500)
                if any(marker in first_lines for marker in [
                    '!pip install',
                    'get_ipython()',
                    '# coding: utf-8',
                    'from google.colab import'
                ]):
                    self.stats['skipped_colab'] += 1
                    return False
        except:
            return False
            
        return True
    
    def extract_imports(self, tree, module_path):
        """Extract import relationships from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'name': alias.asname or alias.name
                    })
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append({
                        'module': node.module,
                        'names': [n.name for n in node.names],
                        'level': node.level
                    })
        
        return imports
    
    def analyze_file(self, filepath):
        """Analyze a single Python file."""
        relative_path = filepath.relative_to(self.root_path)
        
        # Create node ID (replace separators with underscores)
        node_id = str(relative_path).replace('/', '_').replace('\\', '_').replace('.py', '')
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic metrics
            lines = len(content.splitlines())
            size = len(content.encode('utf-8'))
            
            # Parse AST
            tree = ast.parse(content, filename=str(filepath))
            
            # Extract detailed metrics
            classes = []
            functions = []
            imports = self.extract_imports(tree, relative_path)
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            # Calculate complexity (simplified McCabe)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            # Determine node type and color
            if filepath.name == '__init__.py':
                node_type = 'package'
                color = '#9b59b6'
            elif filepath.name.startswith('test_') or 'test' in filepath.parts:
                node_type = 'test'
                color = '#f39c12'
            elif filepath.parent.name == self.root_path.name or filepath.parent == self.root_path:
                if filepath.name in ['setup.py', 'manage.py']:
                    node_type = 'script'
                    color = '#e74c3c'
                else:
                    node_type = 'module'
                    color = '#2ecc71'
            elif any(part in ['examples', 'demo', 'samples'] for part in filepath.parts):
                node_type = 'example'
                color = '#95a5a6'
            else:
                # Determine by location
                parent_dir = filepath.parent.name
                if parent_dir == self.root_path.name:
                    node_type = 'module'
                    color = '#2ecc71'
                else:
                    node_type = 'core'
                    color = '#3498db'
            
            # Create node
            node = {
                'data': {
                    'id': node_id,
                    'label': filepath.name,
                    'path': str(relative_path),
                    'type': node_type,
                    'lines': lines,
                    'size': size,
                    'classes': len(classes),
                    'functions': len(functions),
                    'complexity': complexity,
                    'color': color
                }
            }
            self.nodes.append(node)
            
            # Update stats
            self.stats['total_lines'] += lines
            self.stats['total_functions'] += len(functions)
            self.stats['total_classes'] += len(classes)
            self.stats[f'{node_type}_count'] += 1
            
            # Process imports to create edges
            for imp in imports:
                # Try to resolve internal imports
                if imp['module'].startswith('.'):
                    # Relative import
                    continue  # Skip for now
                elif imp['module'].split('.')[0] in ['os', 'sys', 'json', 'ast']:
                    # Standard library
                    continue
                else:
                    # Check if it's an internal module
                    module_parts = imp['module'].split('.')
                    if module_parts[0] == self.root_path.name or module_parts[0] in ['haam']:
                        # Internal import
                        target_id = imp['module'].replace('.', '_')
                        edge = {
                            'data': {
                                'id': f"{node_id}_to_{target_id}",
                                'source': node_id,
                                'target': target_id,
                                'type': 'import'
                            }
                        }
                        self.edges.append(edge)
                        self.stats['total_imports'] += 1
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            # Add error node
            node = {
                'data': {
                    'id': node_id,
                    'label': filepath.name + ' (error)',
                    'path': str(relative_path),
                    'type': 'error',
                    'color': '#e74c3c',
                    'error': str(e)
                }
            }
            self.nodes.append(node)
    
    def analyze(self):
        """Analyze the entire codebase."""
        print(f"Analyzing codebase at: {self.root_path}")
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(self.root_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore(Path(root) / d)]
            
            for file in files:
                filepath = Path(root) / file
                if not self.should_ignore(filepath) and self.is_python_file(filepath):
                    python_files.append(filepath)
        
        print(f"Found {len(python_files)} Python files to analyze")
        
        # Analyze each file
        for filepath in python_files:
            self.analyze_file(filepath)
        
        # Filter edges to only include those where target exists
        valid_node_ids = {node['data']['id'] for node in self.nodes}
        self.edges = [edge for edge in self.edges 
                     if edge['data']['target'] in valid_node_ids]
        
        # Calculate additional metrics
        self.calculate_graph_metrics()
        
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'stats': dict(self.stats)
        }
    
    def calculate_graph_metrics(self):
        """Calculate graph-level metrics."""
        # Calculate in/out degrees
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for edge in self.edges:
            out_degree[edge['data']['source']] += 1
            in_degree[edge['data']['target']] += 1
        
        # Add degree information to nodes
        for node in self.nodes:
            node_id = node['data']['id']
            node['data']['in_degree'] = in_degree.get(node_id, 0)
            node['data']['out_degree'] = out_degree.get(node_id, 0)
            node['data']['total_degree'] = node['data']['in_degree'] + node['data']['out_degree']
        
        # Find interesting nodes
        orphan_nodes = [n for n in self.nodes if n['data']['total_degree'] == 0]
        hub_nodes = [n for n in self.nodes if n['data']['total_degree'] >= 5]
        
        self.stats['orphan_nodes'] = len(orphan_nodes)
        self.stats['hub_nodes'] = len(hub_nodes)
        self.stats['total_nodes'] = len(self.nodes)
        self.stats['total_edges'] = len(self.edges)


def generate_html_visualization(graph_data, output_file='codebase_visualization.html'):
    """Generate the HTML file with Cytoscape.js visualization."""
    
    template = '''<!DOCTYPE html>
<html>
<head>
    <title>Codebase Visualization - Generated {timestamp}</title>
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
            position: relative;
            min-height: 600px;
        }}
        #sidebar {{
            width: 350px;
            padding: 20px;
            background: #fff;
            border-left: 1px solid #ddd;
            overflow-y: auto;
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
            margin: 5px 0;
        }}
        button {{
            display: inline-block;
            padding: 8px 16px;
            margin: 2px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background: #0056b3;
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
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <script src="https://unpkg.com/layout-base/layout-base.js"></script>
    <script src="https://unpkg.com/cose-base/cose-base.js"></script>
    <script src="https://unpkg.com/cytoscape-cose-bilkent/cytoscape-cose-bilkent.js"></script>
</head>
<body>
    <div id="container">
        <div id="cy"></div>
        <div id="sidebar">
            <h1>Codebase Structure</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="stats">
                <h3>Statistics</h3>
                <div class="stat-row">
                    <span>Total Files:</span>
                    <strong>{total_nodes}</strong>
                </div>
                <div class="stat-row">
                    <span>Total Lines:</span>
                    <strong>{total_lines:,}</strong>
                </div>
                <div class="stat-row">
                    <span>Import Links:</span>
                    <strong>{total_edges}</strong>
                </div>
                <div class="stat-row">
                    <span>Orphan Files:</span>
                    <strong>{orphan_nodes}</strong>
                </div>
                <div class="stat-row">
                    <span>Hub Files:</span>
                    <strong>{hub_nodes}</strong>
                </div>
            </div>
            
            <div class="controls">
                <h3>Layout</h3>
                <button onclick="applyLayout('cose-bilkent')">Force-Directed</button>
                <button onclick="applyLayout('grid')">Grid</button>
                <button onclick="applyLayout('circle')">Circle</button>
                <button onclick="applyLayout('breadthfirst')">Tree</button>
                
                <h3>Highlight</h3>
                <button onclick="highlightType('core')">Core Modules</button>
                <button onclick="highlightType('test')">Tests</button>
                <button onclick="highlightOrphans()">Orphans</button>
                <button onclick="highlightHubs()">Hubs</button>
                <button onclick="resetHighlight()">Reset</button>
            </div>
            
            <div class="node-info" id="node-info">
                <h3>Node Details</h3>
                <p>Click on a node to see details</p>
            </div>
        </div>
    </div>
    
    <script>
        // Register the cose-bilkent layout if available
        if (typeof cytoscape !== 'undefined' && typeof cytoscapeCoseBilkent !== 'undefined') {{
            cytoscape.use(cytoscapeCoseBilkent);
        }}
        
        const fullData = {graph_data_json};
        
        // Extract just nodes and edges for Cytoscape
        const graphElements = {{
            nodes: fullData.nodes,
            edges: fullData.edges
        }};
        
        // Debug: Log what we're loading
        console.log('Loading graph with:', graphElements.nodes.length, 'nodes and', graphElements.edges.length, 'edges');
        console.log('First few nodes:', graphElements.nodes.slice(0, 3));
        
        // Initialize Cytoscape
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: graphElements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'background-color': 'data(color)',
                        'width': ele => Math.max(30, Math.min(100, Math.sqrt(ele.data('lines') || 100) * 3)),
                        'height': ele => Math.max(30, Math.min(100, Math.sqrt(ele.data('lines') || 100) * 3)),
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
                        'line-color': '#ccc',
                        'target-arrow-color': '#ccc',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }},
                {{
                    selector: '.highlighted',
                    style: {{
                        'background-color': '#ff0000',
                        'line-color': '#ff0000',
                        'target-arrow-color': '#ff0000',
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
                name: 'grid',  // Start with grid layout (no plugins needed)
                fit: true,
                padding: 50,
                animate: true,
                animationDuration: 500
            }}
        }});
        
        // Event handlers
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            
            document.getElementById('node-info').innerHTML = `
                <h3>${{data.label}}</h3>
                <p><strong>Type:</strong> ${{data.type}}</p>
                <p><strong>Path:</strong> ${{data.path}}</p>
                <p><strong>Lines:</strong> ${{data.lines || 'N/A'}}</p>
                <p><strong>Functions:</strong> ${{data.functions || 0}}</p>
                <p><strong>Classes:</strong> ${{data.classes || 0}}</p>
                <p><strong>Complexity:</strong> ${{data.complexity || 'N/A'}}</p>
                <p><strong>Connections:</strong> ${{data.total_degree || 0}}</p>
            `;
        }});
        
        // Layout functions
        function applyLayout(name) {{
            const layouts = {{
                'cose-bilkent': {{
                    name: 'cose-bilkent',
                    animate: true,
                    randomize: true
                }},
                'grid': {{
                    name: 'grid',
                    animate: true,
                    fit: true,
                    padding: 30
                }},
                'circle': {{
                    name: 'circle',
                    animate: true,
                    fit: true,
                    padding: 30
                }},
                'breadthfirst': {{
                    name: 'breadthfirst',
                    animate: true,
                    directed: true,
                    spacingFactor: 1.5,
                    fit: true
                }}
            }};
            
            const layoutConfig = layouts[name] || {{ name: name, animate: true, fit: true }};
            cy.layout(layoutConfig).run();
        }}
        
        // Highlight functions
        function highlightType(type) {{
            cy.elements().addClass('dimmed');
            cy.nodes(`[type = "${{type}}"]`).removeClass('dimmed').addClass('highlighted');
        }}
        
        function highlightOrphans() {{
            cy.elements().addClass('dimmed');
            cy.nodes().filter(n => n.degree() === 0).removeClass('dimmed').addClass('highlighted');
        }}
        
        function highlightHubs() {{
            cy.elements().addClass('dimmed');
            cy.nodes().filter(n => n.degree() >= 5).removeClass('dimmed').addClass('highlighted');
        }}
        
        function resetHighlight() {{
            cy.elements().removeClass('highlighted dimmed');
        }}
        
        // Ensure graph is visible on load
        cy.ready(function() {{
            console.log('Cytoscape ready with', cy.nodes().length, 'nodes');
            cy.fit();
            cy.center();
        }});
    </script>
</body>
</html>'''
    
    # Fill in the template
    html = template.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        graph_data_json=json.dumps(graph_data),
        total_nodes=graph_data['stats'].get('total_nodes', 0),
        total_lines=graph_data['stats'].get('total_lines', 0),
        total_edges=graph_data['stats'].get('total_edges', 0),
        orphan_nodes=graph_data['stats'].get('orphan_nodes', 0),
        hub_nodes=graph_data['stats'].get('hub_nodes', 0)
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nVisualization saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Generate interactive codebase visualization')
    parser.add_argument('path', nargs='?', default='.', help='Path to analyze (default: current directory)')
    parser.add_argument('-o', '--output', default='codebase_viz.html', help='Output HTML file')
    parser.add_argument('--open', action='store_true', help='Open in browser after generation')
    parser.add_argument('--ignore', nargs='*', help='Additional patterns to ignore')
    
    args = parser.parse_args()
    
    # Analyze codebase
    analyzer = CodebaseAnalyzer(args.path, ignore_patterns=args.ignore)
    graph_data = analyzer.analyze()
    
    # Print summary
    print(f"\nAnalysis Summary:")
    print(f"  Total files: {graph_data['stats']['total_nodes']}")
    print(f"  Total lines: {graph_data['stats']['total_lines']:,}")
    print(f"  Total imports: {graph_data['stats']['total_edges']}")
    print(f"  Skipped Colab files: {graph_data['stats'].get('skipped_colab', 0)}")
    
    # Generate visualization
    output_file = generate_html_visualization(graph_data, args.output)
    
    # Save graph data as JSON too
    json_file = args.output.replace('.html', '_data.json')
    with open(json_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"Graph data saved to: {json_file}")
    
    # Open in browser if requested
    if args.open:
        webbrowser.open(f'file://{Path(output_file).absolute()}')


if __name__ == '__main__':
    main()