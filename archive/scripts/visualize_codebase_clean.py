#!/usr/bin/env python3
"""
Clean HAAM Codebase Visualizer
==============================
Filters out Colab-style files and creates a clean visualization.
"""

import ast
import os
import json
from pathlib import Path
from collections import defaultdict

def is_colab_file(filepath):
    """Check if a file is a Colab-style script."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check for Colab-specific patterns
            if any(pattern in content[:500] for pattern in [
                '!pip install', 
                '#!/usr/bin/env python3\n"""',
                'from google.colab import',
                'drive.mount(',
                '%%',
                content.strip().startswith('!')
            ]):
                return True
        return False
    except:
        return False

def analyze_clean_codebase(root_path="."):
    """Analyze only clean Python files."""
    root_path = Path(root_path)
    nodes = []
    edges = []
    stats = {
        'total_files': 0,
        'skipped_files': 0,
        'total_lines': 0,
        'total_imports': 0
    }
    
    # Find all Python files
    for root, dirs, files in os.walk(root_path):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                
                # Skip Colab files
                if is_colab_file(filepath):
                    stats['skipped_files'] += 1
                    print(f"Skipping Colab file: {filepath.name}")
                    continue
                
                stats['total_files'] += 1
                relative_path = filepath.relative_to(root_path)
                node_id = str(relative_path).replace('/', '_').replace('.py', '')
                
                # Analyze the file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = len(content.splitlines())
                        stats['total_lines'] += lines
                    
                    # Parse AST
                    tree = ast.parse(content)
                    
                    # Extract functions and classes
                    functions = []
                    classes = []
                    imports = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            functions.append(node.name)
                        elif isinstance(node, ast.ClassDef):
                            classes.append(node.name)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                if 'haam' in alias.name:
                                    imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            if 'haam' in node.module:
                                imports.append(node.module)
                    
                    # Determine node type
                    if filepath.name == '__init__.py':
                        node_type = 'package'
                        color = '#9b59b6'
                    elif 'test' in filepath.name.lower():
                        node_type = 'test'
                        color = '#f39c12'
                    elif filepath.parent.name == 'haam':
                        node_type = 'core'
                        color = '#3498db'
                    elif filepath.name.startswith(('run_', 'generate_')):
                        node_type = 'script'
                        color = '#e74c3c'
                    else:
                        node_type = 'module'
                        color = '#2ecc71'
                    
                    # Add node
                    nodes.append({
                        'data': {
                            'id': node_id,
                            'label': filepath.name,
                            'path': str(relative_path),
                            'lines': lines,
                            'functions': len(functions),
                            'classes': len(classes),
                            'type': node_type,
                            'parent': str(filepath.parent) if filepath.parent != root_path else None
                        },
                        'style': {
                            'background-color': color,
                            'width': min(max(30, lines / 5), 100),
                            'height': min(max(30, lines / 5), 100)
                        }
                    })
                    
                    # Add edges for imports
                    for imp in imports:
                        target_id = imp.replace('.', '_')
                        if target_id.startswith('haam_'):
                            edges.append({
                                'data': {
                                    'source': node_id,
                                    'target': target_id
                                }
                            })
                            stats['total_imports'] += 1
                    
                except Exception as e:
                    print(f"Error analyzing {filepath}: {e}")
                    # Still add the node but mark it as error
                    nodes.append({
                        'data': {
                            'id': node_id,
                            'label': filepath.name + ' (error)',
                            'path': str(relative_path),
                            'type': 'error'
                        },
                        'style': {
                            'background-color': '#e74c3c',
                            'border-width': 3,
                            'border-color': '#c0392b'
                        }
                    })
    
    return nodes, edges, stats

def generate_clean_visualization(nodes, edges, stats, output_file='codebase_clean.html'):
    """Generate a clean, working visualization."""
    
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>HAAM Codebase Structure (Clean)</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: #f5f5f5;
        }
        #main-container {
            display: flex;
            height: 100vh;
        }
        #cy {
            flex: 1;
            background: white;
        }
        #sidebar {
            width: 300px;
            background: #2c3e50;
            color: white;
            padding: 20px;
            overflow-y: auto;
        }
        h2, h3 {
            margin-top: 0;
        }
        .stats {
            background: #34495e;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .stat-item {
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
        }
        .controls {
            margin-bottom: 20px;
        }
        button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: #2980b9;
        }
        #node-details {
            background: #34495e;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .legend {
            margin-top: 20px;
            font-size: 12px;
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
            border-radius: 3px;
        }
    </style>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/cytoscape-cola@2.5.1/cytoscape-cola.js"></script>
    <script src="https://unpkg.com/webcola@3.4.0/WebCola/cola.min.js"></script>
</head>
<body>
    <div id="main-container">
        <div id="cy"></div>
        <div id="sidebar">
            <h2>HAAM Codebase</h2>
            
            <div class="stats">
                <h3>Statistics</h3>
                <div class="stat-item">
                    <span>Total Files:</span>
                    <span>''' + str(stats['total_files']) + '''</span>
                </div>
                <div class="stat-item">
                    <span>Skipped (Colab):</span>
                    <span>''' + str(stats['skipped_files']) + '''</span>
                </div>
                <div class="stat-item">
                    <span>Total Lines:</span>
                    <span>''' + f"{stats['total_lines']:,}" + '''</span>
                </div>
                <div class="stat-item">
                    <span>Import Links:</span>
                    <span>''' + str(stats['total_imports']) + '''</span>
                </div>
            </div>
            
            <div class="controls">
                <h3>Layout</h3>
                <button onclick="applyLayout('cola')">Force-Directed</button>
                <button onclick="applyLayout('grid')">Grid</button>
                <button onclick="applyLayout('circle')">Circle</button>
                <button onclick="applyLayout('concentric')">Concentric</button>
                <button onclick="cy.fit()">Fit to Screen</button>
            </div>
            
            <div id="node-details">
                <h3>Node Details</h3>
                <p>Click on a node to see details</p>
            </div>
            
            <div class="legend">
                <h3>Legend</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background:#3498db;"></div>
                    <span>Core Package</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#e74c3c;"></div>
                    <span>Scripts</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#2ecc71;"></div>
                    <span>Modules</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#f39c12;"></div>
                    <span>Tests</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#9b59b6;"></div>
                    <span>Package Init</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Graph data
        const elements = {
            nodes: ''' + json.dumps(nodes) + ''',
            edges: ''' + json.dumps(edges) + '''
        };
        
        // Initialize Cytoscape
        const cy = cytoscape({
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '12px',
                        'color': '#fff',
                        'text-outline-width': 2,
                        'text-outline-color': '#2c3e50',
                        'border-width': 2,
                        'border-color': '#2c3e50'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#95a5a6',
                        'target-arrow-color': '#95a5a6',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }
                },
                {
                    selector: ':selected',
                    style: {
                        'border-width': 4,
                        'border-color': '#e74c3c'
                    }
                }
            ],
            layout: {
                name: 'cola',
                animate: true,
                randomize: false,
                avoidOverlap: true,
                edgeLength: 100,
                nodeSpacing: 20
            }
        });
        
        // Node click handler
        cy.on('tap', 'node', function(evt) {
            const node = evt.target;
            const data = node.data();
            
            const details = `
                <h3>Node Details</h3>
                <p><strong>${data.label}</strong></p>
                <p>Type: ${data.type}</p>
                <p>Path: ${data.path || 'N/A'}</p>
                <p>Lines: ${data.lines || 'N/A'}</p>
                <p>Functions: ${data.functions || 0}</p>
                <p>Classes: ${data.classes || 0}</p>
            `;
            
            document.getElementById('node-details').innerHTML = details;
        });
        
        // Layout function
        function applyLayout(name) {
            const layouts = {
                cola: {
                    name: 'cola',
                    animate: true,
                    randomize: false,
                    avoidOverlap: true,
                    edgeLength: 100,
                    nodeSpacing: 20
                },
                grid: {
                    name: 'grid',
                    animate: true,
                    padding: 30
                },
                circle: {
                    name: 'circle',
                    animate: true,
                    padding: 30
                },
                concentric: {
                    name: 'concentric',
                    animate: true,
                    concentric: function(node) {
                        return node.data('type') === 'core' ? 2 : 1;
                    },
                    levelWidth: function() { return 1; }
                }
            };
            
            cy.layout(layouts[name]).run();
        }
        
        // Initial fit
        cy.ready(function() {
            cy.fit();
        });
    </script>
</body>
</html>'''
    
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"\nClean visualization saved to: {output_file}")

if __name__ == "__main__":
    print("Analyzing clean HAAM codebase (excluding Colab files)...")
    
    nodes, edges, stats = analyze_clean_codebase(".")
    
    print(f"\nAnalysis complete:")
    print(f"  Total Python files found: {stats['total_files'] + stats['skipped_files']}")
    print(f"  Clean files analyzed: {stats['total_files']}")
    print(f"  Colab files skipped: {stats['skipped_files']}")
    print(f"  Total lines of code: {stats['total_lines']:,}")
    print(f"  Import relationships: {stats['total_imports']}")
    
    generate_clean_visualization(nodes, edges, stats)