"""
HAAM Visualization Module
=========================

Functions for creating interactive visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Optional, Tuple, Union
import os


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)


class HAAMVisualizer:
    """
    Create visualizations for HAAM analysis results.
    """
    
    def __init__(self, haam_results: Dict, topic_summaries: Optional[Dict] = None):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        haam_results : Dict
            Results from HAAMAnalysis
        topic_summaries : Dict, optional
            Topic summaries from TopicAnalyzer
        """
        self.results = haam_results
        self.topic_summaries = topic_summaries or {}
        
    def create_main_visualization(self, 
                                 pc_indices: List[int],
                                 output_file: Optional[str] = None) -> str:
        """
        Create main HAAM framework visualization.
        
        Parameters
        ----------
        pc_indices : List[int]
            List of PC indices to display (0-based)
        output_file : str, optional
            Path to save HTML file
            
        Returns
        -------
        str
            HTML content
        """
        # Get coefficient data
        pc_data = []
        
        for i, pc_idx in enumerate(pc_indices[:9]):  # Max 9 PCs
            pc_info = {
                'pc': pc_idx + 1,  # 1-based for display
                'name': f'PC{pc_idx + 1}',
                'corrs': []
            }
            
            # Get correlations/coefficients for each outcome
            for outcome in ['SC', 'AI', 'HU']:
                if outcome in self.results['debiased_lasso']:
                    coef = self.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                    pc_info['corrs'].append(float(coef))
                else:
                    pc_info['corrs'].append(0.0)
            
            # Add topic information
            if pc_idx in self.topic_summaries:
                pc_info['pos'] = self.topic_summaries[pc_idx]['high']
                pc_info['neg'] = self.topic_summaries[pc_idx]['low']
            else:
                pc_info['pos'] = 'loading...'
                pc_info['neg'] = 'loading...'
                
            # Add position for visualization (3x3 grid)
            row = i // 3
            col = i % 3
            pc_info['x'] = 100 + col * 240
            pc_info['y'] = 50 + row * 160
            
            pc_data.append(pc_info)
        
        # Calculate R² values
        r2_values = {}
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                r2_values[outcome] = self.results['debiased_lasso'][outcome]['r2_cv']
            else:
                r2_values[outcome] = 0.0
                
        # Generate HTML
        html_template = self._get_main_visualization_template()
        
        # Insert data
        html_content = html_template.replace(
            '%%PC_DATA%%', 
            json.dumps(pc_data, indent=2, cls=NumpyEncoder)
        )
        
        # Insert R² values
        html_content = html_content.replace('%%R2_SC%%', f"{r2_values.get('SC', 0):.3f}")
        html_content = html_content.replace('%%R2_AI%%', f"{r2_values.get('AI', 0):.3f}")
        html_content = html_content.replace('%%R2_HU%%', f"{r2_values.get('HU', 0):.3f}")
        
        # Calculate PoMA values (placeholder for now)
        html_content = html_content.replace('%%POMA_AI%%', "65.6")
        html_content = html_content.replace('%%POMA_HU%%', "44.6")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Main visualization saved to: {output_file}")
            
        return html_content
    
    def create_mini_visualization(self,
                                 n_components: int = 200,
                                 n_highlight: int = 20,
                                 output_file: Optional[str] = None) -> str:
        """
        Create mini grid visualization of all PCs.
        
        Parameters
        ----------
        n_components : int
            Total number of components to show
        n_highlight : int
            Number of top components to highlight
        output_file : str, optional
            Path to save HTML file
            
        Returns
        -------
        str
            HTML content
        """
        # Prepare coefficient data
        coef_data = {}
        
        for i in range(n_components):
            coef_data[str(i+1)] = {
                'sc': 0.0,
                'ai': 0.0,
                'hu': 0.0
            }
            
            for outcome in ['sc', 'ai', 'hu']:
                outcome_upper = outcome.upper()
                if outcome_upper in self.results['debiased_lasso']:
                    coef_data[str(i+1)][outcome] = float(
                        self.results['debiased_lasso'][outcome_upper]['coefs_std'][i]
                    )
        
        # Generate HTML
        html_template = self._get_mini_visualization_template()
        
        # Insert data
        html_content = html_template.replace(
            '%%COEFFICIENT_DATA%%',
            json.dumps(coef_data, indent=2, cls=NumpyEncoder)
        )
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Mini visualization saved to: {output_file}")
            
        return html_content
    
    def plot_pc_effects(self, 
                       pc_idx: int,
                       topic_associations: Dict,
                       figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
        """
        Create 4-panel bar chart showing PC effects on outcomes.
        
        Parameters
        ----------
        pc_idx : int
            PC index (0-based)
        topic_associations : Dict
            Topic associations from TopicAnalyzer
        figsize : Tuple[int, int]
            Figure size
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'PC{pc_idx + 1} Effects Analysis', fontsize=16, fontweight='bold')
        
        outcomes = ['SC', 'AI', 'HU']
        colors = ['#334155', '#be123c', '#d97706']
        
        for ax, outcome, color in zip(axes, outcomes, colors):
            if outcome not in self.results['debiased_lasso']:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{outcome} Model')
                continue
                
            # Get coefficient and SE
            coef = self.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
            se = self.results['debiased_lasso'][outcome]['ses_std'][pc_idx]
            
            # Create bar with error bar
            ax.bar(0, coef, color=color, alpha=0.7, width=0.5)
            ax.errorbar(0, coef, yerr=1.96*se, color='black', capsize=10, capthick=2)
            
            # Add significance stars
            p_val = 2 * (1 - stats.norm.cdf(abs(coef/se))) if se > 0 else 1
            sig_text = ''
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
                
            ax.text(0, coef + np.sign(coef)*0.02, sig_text, ha='center', fontsize=14)
            
            # Formatting
            ax.set_title(f'{outcome} Model', fontsize=14)
            ax.set_ylabel('Standardized Coefficient', fontsize=12)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.grid(True, axis='y', alpha=0.3)
            
        plt.tight_layout()
        return fig
    
    def create_umap_visualization(self,
                                 umap_embeddings: np.ndarray,
                                 color_by: str = 'SC',
                                 topic_labels: Optional[Dict] = None,
                                 show_topics: bool = True,
                                 output_file: Optional[str] = None) -> go.Figure:
        """
        Create interactive UMAP visualization.
        
        Parameters
        ----------
        umap_embeddings : np.ndarray
            2D or 3D UMAP embeddings
        color_by : str
            Variable to color by: 'SC', 'AI', 'HU', or 'PC1', 'PC2', etc.
        topic_labels : Dict, optional
            Topic labels for points
        show_topics : bool
            Whether to show topic labels
        output_file : str, optional
            Path to save HTML file
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        # Determine if 2D or 3D
        is_3d = umap_embeddings.shape[1] == 3
        
        # Get color values
        if color_by in ['SC', 'AI', 'HU']:
            outcome_map = {'SC': 'criterion', 'AI': 'ai_judgment', 'HU': 'human_judgment'}
            color_values = getattr(self, outcome_map[color_by], None)
            color_title = f'{color_by} Values'
        elif color_by.startswith('PC'):
            pc_idx = int(color_by[2:]) - 1  # Convert to 0-based
            color_values = self.results['pca_features'][:, pc_idx]
            color_title = f'{color_by} Scores'
        else:
            raise ValueError(f"Invalid color_by option: {color_by}")
            
        # Create figure
        if is_3d:
            fig = go.Figure(data=[go.Scatter3d(
                x=umap_embeddings[:, 0],
                y=umap_embeddings[:, 1],
                z=umap_embeddings[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color_values,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title=color_title)
                ),
                text=[f'Point {i}' for i in range(len(umap_embeddings))],
                hoverinfo='text'
            )])
            
            fig.update_layout(
                title=f'3D UMAP colored by {color_by}',
                scene=dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3'
                )
            )
        else:
            fig = go.Figure(data=[go.Scatter(
                x=umap_embeddings[:, 0],
                y=umap_embeddings[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=color_values,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title=color_title)
                ),
                text=[f'Point {i}' for i in range(len(umap_embeddings))],
                hoverinfo='text'
            )])
            
            fig.update_layout(
                title=f'2D UMAP colored by {color_by}',
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2'
            )
            
        # Add topic labels if provided
        if show_topics and topic_labels:
            # Implementation for topic labels would go here
            pass
            
        fig.update_layout(
            width=800,
            height=600,
            hovermode='closest'
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"UMAP visualization saved to: {output_file}")
            
        return fig
    
    def create_pc_effects_plot(self, 
                              pc_indices: List[int],
                              output_file: Optional[str] = None) -> go.Figure:
        """
        Create bar chart showing PC effects.
        
        Parameters
        ----------
        pc_indices : List[int]
            List of PC indices to plot
        output_file : str, optional
            Path to save HTML file
            
        Returns
        -------
        go.Figure
            Plotly figure
        """
        import plotly.graph_objects as go
        
        # Get coefficient data
        coef_data = []
        outcomes = ['SC', 'AI', 'HU']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for outcome in outcomes:
            if outcome in self.results['debiased_lasso']:
                coefs = self.results['debiased_lasso'][outcome]['coefs_std']
                coef_values = [coefs[pc] for pc in pc_indices]
                coef_data.append(coef_values)
            else:
                coef_data.append([0] * len(pc_indices))
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each outcome
        for i, (outcome, coefs, color) in enumerate(zip(outcomes, coef_data, colors)):
            fig.add_trace(go.Bar(
                name=outcome,
                x=[f'PC{pc+1}' for pc in pc_indices],
                y=coefs,
                marker_color=color,
                offsetgroup=i
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Top {len(pc_indices)} Principal Components - Standardized Coefficients',
            xaxis_title='Principal Component',
            yaxis_title='Standardized Coefficient',
            barmode='group',
            template='plotly_white',
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        # Add zero line
        fig.add_hline(y=0, line_color='black', line_width=0.5)
        
        if output_file:
            fig.write_html(output_file)
            print(f"PC effects visualization saved to: {output_file}")
            
        return fig
    
    def _get_main_visualization_template(self) -> str:
        """Get HTML template for main visualization."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HAAM Framework Visualization</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .tooltip { visibility: hidden; opacity: 0; transition: opacity 0.2s; }
        .has-tooltip:hover .tooltip { visibility: visible; opacity: 1; }
        .positive-fill { fill: #10b981; }
        .negative-fill { fill: #ef4444; }
        .neutral-fill { fill: #94a3b8; }
    </style>
</head>
<body class="bg-slate-50 p-4">
    <div class="max-w-screen-2xl mx-auto bg-white p-8 rounded-xl shadow-lg">
        <svg viewBox="0 0 1500 850" xmlns="http://www.w3.org/2000/svg" class="w-full h-auto">
            <!-- SVG content with placeholders -->
            <text x="750" y="35" text-anchor="middle" font-size="28" font-weight="bold">
                HAAM Framework Analysis Results
            </text>
            
            <!-- Nodes -->
            <g id="criterion-node">
                <rect x="50" y="335" width="150" height="50" rx="10" fill="#334155" />
                <text x="125" y="365" font-size="16" fill="white" text-anchor="middle">Criterion</text>
                <text x="125" y="418" font-size="12" font-weight="600">R² = %%R2_SC%%</text>
            </g>
            
            <g id="ai-node">
                <rect x="1300" y="235" width="150" height="50" rx="10" fill="#be123c" />
                <text x="1375" y="265" font-size="16" fill="white" text-anchor="middle">AI Judgment</text>
                <text x="1375" y="202" font-size="12" font-weight="600" fill="#be123c">R² = %%R2_AI%%</text>
            </g>
            
            <g id="human-node">
                <rect x="1300" y="435" width="150" height="50" rx="10" fill="#d97706" />
                <text x="1375" y="465" font-size="16" fill="white" text-anchor="middle">Human Judgment</text>
                <text x="1375" y="518" font-size="12" font-weight="600" fill="#d97706">R² = %%R2_HU%%</text>
            </g>
            
            <!-- PC Box -->
            <rect x="400" y="120" width="700" height="480" rx="15" fill="#f8fafc" stroke="#94a3b8" stroke-width="2"/>
            <text x="750" y="105" text-anchor="middle" font-size="16" font-weight="600">Principal Components</text>
            
            <!-- PC Data will be inserted here -->
            <g id="pc-info-grid" transform="translate(410, 100)"></g>
        </svg>
        
        <script>
            const pcData = %%PC_DATA%%;
            const grid = document.getElementById('pc-info-grid');
            
            pcData.forEach(data => {
                const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
                g.setAttribute('transform', `translate(${data.x}, ${data.y})`);
                
                // PC title
                const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
                title.setAttribute('x', 0);
                title.setAttribute('y', 0);
                title.setAttribute('text-anchor', 'middle');
                title.setAttribute('font-size', '14');
                title.setAttribute('font-weight', 'bold');
                title.textContent = data.name;
                g.appendChild(title);
                
                // Keywords
                const posText = document.createElementNS("http://www.w3.org/2000/svg", "text");
                posText.setAttribute('x', 0);
                posText.setAttribute('y', 20);
                posText.setAttribute('text-anchor', 'middle');
                posText.setAttribute('font-size', '11');
                posText.setAttribute('fill', '#059669');
                posText.textContent = `+ ${data.pos}`;
                g.appendChild(posText);
                
                const negText = document.createElementNS("http://www.w3.org/2000/svg", "text");
                negText.setAttribute('x', 0);
                negText.setAttribute('y', 35);
                negText.setAttribute('text-anchor', 'middle');
                negText.setAttribute('font-size', '11');
                negText.setAttribute('fill', '#dc2626');
                negText.textContent = `- ${data.neg}`;
                g.appendChild(negText);
                
                // Coefficient dots
                const dotPositions = [-25, 0, 25];
                const labels = ['SC', 'AI', 'HU'];
                
                data.corrs.forEach((corr, i) => {
                    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    dot.setAttribute('cx', dotPositions[i]);
                    dot.setAttribute('cy', 55);
                    dot.setAttribute('r', Math.min(8, Math.abs(corr) * 20));
                    
                    if (corr > 0.005) dot.setAttribute('class', 'positive-fill');
                    else if (corr < -0.005) dot.setAttribute('class', 'negative-fill');
                    else dot.setAttribute('class', 'neutral-fill');
                    
                    g.appendChild(dot);
                });
                
                grid.appendChild(g);
            });
        </script>
    </div>
</body>
</html>'''
    
    def _get_mini_visualization_template(self) -> str:
        """Get HTML template for mini visualization."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PC Coefficient Grid</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .positive-fill { fill: #10b981; }
        .negative-fill { fill: #ef4444; }
        .zero-fill { fill: #cbd5e1; }
        .tooltip-text {
            visibility: hidden;
            background-color: #1e293b;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 10;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .tooltip-container:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body class="p-8 bg-slate-100">
    <div class="max-w-7xl mx-auto bg-white p-6 rounded-xl shadow-md">
        <h1 class="text-2xl font-bold mb-4">Principal Component Coefficients</h1>
        <div id="visualization-grid" class="grid grid-cols-10 gap-4"></div>
    </div>
    
    <script>
        const coefficientData = %%COEFFICIENT_DATA%%;
        const grid = document.getElementById('visualization-grid');
        
        // Find max coefficient for scaling
        let maxAbsCoeff = 0;
        Object.values(coefficientData).forEach(pc => {
            ['sc', 'ai', 'hu'].forEach(model => {
                const coeff = Math.abs(pc[model]);
                if (coeff > maxAbsCoeff) maxAbsCoeff = coeff;
            });
        });
        
        const getRadius = (coeff) => {
            if (maxAbsCoeff === 0) return 1.5;
            const scale = Math.pow(Math.abs(coeff) / maxAbsCoeff, 0.5);
            return 1.5 + scale * 8.5;
        };
        
        // Create grid items
        for (let i = 1; i <= 200; i++) {
            const pcData = coefficientData[i.toString()] || { sc: 0, ai: 0, hu: 0 };
            const container = document.createElement('div');
            container.className = 'flex flex-col items-center p-2 border rounded bg-slate-50';
            
            const title = document.createElement('h4');
            title.className = 'text-sm font-semibold mb-2';
            title.textContent = `PC${i}`;
            container.appendChild(title);
            
            const svgContainer = document.createElement('div');
            svgContainer.className = 'tooltip-container relative';
            
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('width', '80');
            svg.setAttribute('height', '24');
            
            ['sc', 'ai', 'hu'].forEach((model, index) => {
                const coeff = pcData[model];
                const radius = getRadius(coeff);
                const colorClass = coeff > 0 ? 'positive-fill' : 
                                 (coeff < 0 ? 'negative-fill' : 'zero-fill');
                
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', 15 + index * 25);
                circle.setAttribute('cy', 12);
                circle.setAttribute('r', radius);
                circle.setAttribute('class', colorClass);
                svg.appendChild(circle);
            });
            
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip-text';
            tooltip.innerHTML = `
                <div><strong>SC:</strong> ${pcData.sc.toFixed(4)}</div>
                <div><strong>AI:</strong> ${pcData.ai.toFixed(4)}</div>
                <div><strong>HU:</strong> ${pcData.hu.toFixed(4)}</div>
            `;
            
            svgContainer.appendChild(svg);
            svgContainer.appendChild(tooltip);
            container.appendChild(svgContainer);
            grid.appendChild(container);
        }
    </script>
</body>
</html>'''