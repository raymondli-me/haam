#!/usr/bin/env python3
"""
Simple HAAM Codebase Visualizer
================================
A simpler version that handles Colab scripts and creates a basic visualization.
"""

import os
import json
from pathlib import Path
import re

def analyze_codebase(root_path="."):
    """Simple analysis that just looks at files and basic imports."""
    root_path = Path(root_path)
    nodes = []
    edges = []
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(root_path):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    # Analyze each file
    for filepath in python_files:
        relative_path = filepath.relative_to(root_path)
        node_id = str(relative_path).replace('/', '_').replace('.py', '')
        
        # Get file stats
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.splitlines())
        except:
            lines = 0
            content = ""
        
        file_size = filepath.stat().st_size
        
        # Determine file type
        if 'test' in filepath.name.lower():
            node_type = 'test'
        elif filepath.name.startswith(('generate_', 'run_')):
            node_type = 'script'
        elif filepath.name == '__init__.py':
            node_type = 'package'
        elif filepath.parent.name == 'haam':
            node_type = 'core'
        else:
            node_type = 'module'
        
        # Simple import detection
        imports = []
        import_pattern = r'(?:from\s+(\S+)\s+import|import\s+(\S+))'
        for match in re.finditer(import_pattern, content):
            module = match.group(1) or match.group(2)
            if module and (module.startswith('haam') or module.startswith('.')):
                imports.append(module)
        
        # Add node
        nodes.append({
            'data': {
                'id': node_id,
                'label': filepath.name,
                'path': str(relative_path),
                'size': file_size,
                'lines': lines,
                'type': node_type,
                'parent': str(filepath.parent.name) if filepath.parent != root_path else None
            }
        })
        
        # Add edges for imports
        for imp in imports:
            if imp.startswith('haam.'):
                target = 'haam_' + imp.split('.')[1]
            elif imp == 'haam':
                target = 'haam___init__'
            else:
                continue
                
            edges.append({
                'data': {
                    'source': node_id,
                    'target': target,
                    'type': 'import'
                }
            })
    
    return {
        'elements': {
            'nodes': nodes,
            'edges': edges
        }
    }

def generate_simple_html(graph_data, output_file='codebase_simple.html'):
    """Generate a simple HTML visualization."""
    
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>HAAM Codebase Structure</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/cytoscape-fcose@2.1.0/cytoscape-fcose.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        #cy {
            width: 100vw;
            height: 100vh;
            background-color: #f5f5f5;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            max-width: 300px;
        }
        .controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        button {
            margin: 2px;
            padding: 5px 10px;
            cursor: pointer;
        }
        .stats {
            font-size: 14px;
            margin-top: 10px;
        }
        .legend {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            font-size: 12px;
        }
        .legend-item {
            display: inline-block;
            margin: 0 10px;
        }
        .legend-color {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div id="cy"></div>
    
    <div id="info">
        <h3>HAAM Codebase Map</h3>
        <div class="stats">
            <strong>Files:</strong> <span id="file-count">0</span><br>
            <strong>Imports:</strong> <span id="import-count">0</span><br>
            <strong>Selected:</strong> <span id="selected">Click a node</span>
        </div>
    </div>
    
    <div class="controls">
        <button onclick="runLayout('fcose')">Force Layout</button>
        <button onclick="runLayout('grid')">Grid Layout</button>
        <button onclick="runLayout('circle')">Circle Layout</button>
        <button onclick="cy.fit()">Fit</button>
    </div>
    
    <div class="legend">
        <span class="legend-item"><span class="legend-color" style="background:#e74c3c;"></span>Scripts</span>
        <span class="legend-item"><span class="legend-color" style="background:#3498db;"></span>Core</span>
        <span class="legend-item"><span class="legend-color" style="background:#2ecc71;"></span>Modules</span>
        <span class="legend-item"><span class="legend-color" style="background:#f39c12;"></span>Tests</span>
    </div>
    
    <script>
        const graphData = ''' + json.dumps(graph_data) + ''';
        
        // Initialize Cytoscape
        const cy = cytoscape({
            container: document.getElementById('cy'),
            elements: graphData.elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'label': 'data(label)',
                        'font-size': '10px',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'width': ele => Math.max(30, Math.min(100, ele.data('lines') / 10)),
                        'height': ele => Math.max(30, Math.min(100, ele.data('lines') / 10)),
                        'background-color': ele => {
                            switch(ele.data('type')) {
                                case 'script': return '#e74c3c';
                                case 'core': return '#3498db';
                                case 'test': return '#f39c12';
                                case 'package': return '#9b59b6';
                                default: return '#2ecc71';
                            }
                        }
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
                        'border-width': 3,
                        'border-color': '#e74c3c'
                    }
                }
            ],
            layout: {
                name: 'fcose',
                animate: true,
                animationDuration: 1000
            }
        });
        
        // Update stats
        document.getElementById('file-count').textContent = graphData.elements.nodes.length;
        document.getElementById('import-count').textContent = graphData.elements.edges.length;
        
        // Node click handler
        cy.on('tap', 'node', function(evt) {
            const node = evt.target;
            const data = node.data();
            document.getElementById('selected').innerHTML = 
                `<strong>${data.label}</strong><br>
                 Path: ${data.path}<br>
                 Lines: ${data.lines}<br>
                 Size: ${(data.size / 1024).toFixed(1)} KB`;
        });
        
        // Layout function
        function runLayout(name) {
            cy.layout({ name: name, animate: true }).run();
        }
        
        // Group by directory
        const groups = {};
        cy.nodes().forEach(node => {
            const parent = node.data('parent');
            if (parent) {
                if (!groups[parent]) groups[parent] = [];
                groups[parent].push(node);
            }
        });
        
        // Log structure
        console.log('Codebase structure:', groups);
        console.log('Total nodes:', cy.nodes().length);
        console.log('Total edges:', cy.edges().length);
    </script>
</body>
</html>'''
    
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"Simple visualization saved to: {output_file}")

if __name__ == "__main__":
    print("Analyzing HAAM codebase (simple version)...")
    graph_data = analyze_codebase(".")
    
    print(f"Found {len(graph_data['elements']['nodes'])} files")
    print(f"Found {len(graph_data['elements']['edges'])} imports")
    
    # Group files by type
    by_type = {}
    for node in graph_data['elements']['nodes']:
        node_type = node['data']['type']
        if node_type not in by_type:
            by_type[node_type] = []
        by_type[node_type].append(node['data']['label'])
    
    print("\nFiles by type:")
    for type_name, files in by_type.items():
        print(f"  {type_name}: {len(files)} files")
        if type_name == 'script':
            print(f"    Scripts: {', '.join(sorted(files)[:5])}...")
    
    generate_simple_html(graph_data)