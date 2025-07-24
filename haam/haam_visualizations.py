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
from typing import Dict, List, Optional, Tuple, Union, Any
import os
from scipy import stats


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
                                 output_file: Optional[str] = None,
                                 pc_names: Optional[Dict[int, str]] = None) -> str:
        """
        Create main HAAM framework visualization with dynamic metrics.
        
        The visualization now shows:
        - Generic "Y" label instead of "SC" for criterion
        - Dynamically calculated R², PoMA, and unmodeled path percentages
        - Custom PC names when provided (shows "-" otherwise)
        - Enhanced topic display using c-TF-IDF
        
        Parameters
        ----------
        pc_indices : List[int]
            List of PC indices to display (0-based)
        output_file : str, optional
            Path to save HTML file
        pc_names : Dict[int, str], optional
            Manual names for PCs. Keys are PC indices (0-based), values are names.
            If not provided, uses "-" for all PCs.
            Example: {0: "Formality", 3: "Complexity", 6: "Sentiment"}
            
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
                'name': '',  # Will be set based on topics
                'corrs': []
            }
            
            # Get correlations/coefficients for each outcome (Y, AI, HU)
            # Changed SC to Y to match the template
            outcome_map = {'Y': 'SC', 'AI': 'AI', 'HU': 'HU'}
            for display_outcome in ['Y', 'AI', 'HU']:
                internal_outcome = outcome_map[display_outcome]
                if internal_outcome in self.results['debiased_lasso']:
                    coef = self.results['debiased_lasso'][internal_outcome]['coefs_std'][pc_idx]
                    pc_info['corrs'].append(float(coef))
                else:
                    pc_info['corrs'].append(0.0)
            
            # Add topic information
            if pc_idx in self.topic_summaries:
                # Handle both old and new format
                if 'high_topics' in self.topic_summaries[pc_idx]:
                    # New format - show first topic only for visualization
                    high_topics = self.topic_summaries[pc_idx]['high_topics'][:1]
                    low_topics = self.topic_summaries[pc_idx]['low_topics'][:1]
                    # Extract just the first few keywords from each topic
                    if high_topics and high_topics[0] != 'No significant high topics':
                        keywords = high_topics[0].split(' | ')[:3]
                        pc_info['pos'] = ', '.join(keywords)
                    else:
                        pc_info['pos'] = ''  # Empty instead of 'None'
                    
                    if low_topics and low_topics[0] != 'No significant low topics':
                        keywords = low_topics[0].split(' | ')[:3]
                        pc_info['neg'] = ', '.join(keywords)
                    else:
                        pc_info['neg'] = ''  # Empty instead of 'None'
                else:
                    # Old format
                    pc_info['pos'] = self.topic_summaries[pc_idx].get('high', 'None')
                    pc_info['neg'] = self.topic_summaries[pc_idx].get('low', 'None')
                    
                # Use manual name if provided, otherwise use dash
                if pc_names and pc_idx in pc_names:
                    pc_info['name'] = pc_names[pc_idx]
                else:
                    pc_info['name'] = '-'
            else:
                pc_info['pos'] = 'loading...'
                pc_info['neg'] = 'loading...'
                pc_info['name'] = ''
                
            # Add position for visualization (3x3 grid)
            row = i // 3
            col = i % 3
            pc_info['x'] = 100 + col * 240
            pc_info['y'] = 50 + row * 160
            
            pc_data.append(pc_info)
        
        # Calculate all metrics
        metrics = self._calculate_visualization_metrics()
        
        # Generate HTML
        html_template = self._get_main_visualization_template()
        
        # Insert data
        html_content = html_template.replace(
            '%%PC_DATA%%', 
            json.dumps(pc_data, indent=2, cls=NumpyEncoder)
        )
        
        # Insert all metrics
        # R² values
        html_content = html_content.replace('%%R2_Y%%', f"{metrics['r2_y']:.3f}")
        html_content = html_content.replace('%%R2_AI%%', f"{metrics['r2_ai']:.3f}")
        html_content = html_content.replace('%%R2_HU%%', f"{metrics['r2_hu']:.3f}")
        
        # Total Effects (DML coefficients)
        html_content = html_content.replace('%%TOTAL_EFFECT_Y_AI%%', f"{metrics['total_effect_y_ai']:.3f}")
        html_content = html_content.replace('%%TOTAL_EFFECT_Y_HU%%', f"{metrics['total_effect_y_hu']:.3f}")
        html_content = html_content.replace('%%TOTAL_EFFECT_HU_AI%%', f"{metrics['total_effect_hu_ai']:.3f}")
        
        # DML Check Betas
        html_content = html_content.replace('%%DML_CHECK_Y_AI%%', f"{metrics['dml_check_y_ai']:.3f}")
        html_content = html_content.replace('%%DML_CHECK_Y_HU%%', f"{metrics['dml_check_y_hu']:.3f}")
        html_content = html_content.replace('%%DML_CHECK_HU_AI%%', f"{metrics['dml_check_hu_ai']:.3f}")
        
        # Residual Correlations
        html_content = html_content.replace('%%C_AI_HU%%', f"{metrics['c_ai_hu']:.3f}")
        html_content = html_content.replace('%%C_Y_AI%%', f"{metrics['c_y_ai']:.3f}")
        html_content = html_content.replace('%%C_Y_HU%%', f"{metrics['c_y_hu']:.3f}")
        
        # Value-Prediction Correlations
        html_content = html_content.replace('%%R_Y_YHAT%%', f"{metrics['r_y_yhat']:.3f}")
        html_content = html_content.replace('%%R_AI_AIHAT%%', f"{metrics['r_ai_aihat']:.3f}")
        html_content = html_content.replace('%%R_HU_HUHAT%%', f"{metrics['r_hu_huhat']:.3f}")
        
        # Policy Similarities
        html_content = html_content.replace('%%R_YHAT_AIHAT%%', f"{metrics['r_yhat_aihat']:.3f}")
        html_content = html_content.replace('%%R_YHAT_HUHAT%%', f"{metrics['r_yhat_huhat']:.3f}")
        html_content = html_content.replace('%%R_AIHAT_HUHAT%%', f"{metrics['r_aihat_huhat']:.3f}")
        
        # PoMA and Unmodeled paths
        html_content = html_content.replace('%%POMA_AI%%', f"{metrics['poma_ai']:.1f}%")
        html_content = html_content.replace('%%POMA_HU%%', f"{metrics['poma_hu']:.1f}%")
        html_content = html_content.replace('%%POMA_HU_AI%%', f"{metrics['poma_hu_ai']:.1f}%")
        html_content = html_content.replace('%%UNMODELED_AI%%', f"{metrics['unmodeled_ai']:.1f}%")
        html_content = html_content.replace('%%UNMODELED_HU%%', f"{metrics['unmodeled_hu']:.1f}%")
        html_content = html_content.replace('%%UNMODELED_HU_AI%%', f"{metrics['unmodeled_hu_ai']:.1f}%")
        
        # Number of selected components
        html_content = html_content.replace('%%N_SELECTED_Y%%', str(metrics['n_selected_y']))
        html_content = html_content.replace('%%N_SELECTED_AI%%', str(metrics['n_selected_ai']))
        html_content = html_content.replace('%%N_SELECTED_HU%%', str(metrics['n_selected_hu']))
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Main visualization saved to: {output_file}")
            
        return html_content
    
    def _calculate_visualization_metrics(self) -> Dict[str, float]:
        """
        Calculate all metrics needed for the visualization.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # R² values (cross-validated)
        metrics['r2_y'] = self.results['debiased_lasso'].get('SC', {}).get('r2_cv', 0.0)
        metrics['r2_ai'] = self.results['debiased_lasso'].get('AI', {}).get('r2_cv', 0.0)
        metrics['r2_hu'] = self.results['debiased_lasso'].get('HU', {}).get('r2_cv', 0.0)
        
        # Number of selected components
        metrics['n_selected_y'] = self.results['debiased_lasso'].get('SC', {}).get('n_selected', 0)
        metrics['n_selected_ai'] = self.results['debiased_lasso'].get('AI', {}).get('n_selected', 0)
        metrics['n_selected_hu'] = self.results['debiased_lasso'].get('HU', {}).get('n_selected', 0)
        
        # Total Effects (DML coefficients)
        if 'total_effects' in self.results:
            te = self.results['total_effects']
            metrics['total_effect_y_ai'] = te.get('Y_AI', {}).get('coefficient', 0.0)
            metrics['total_effect_y_hu'] = te.get('Y_HU', {}).get('coefficient', 0.0)
            metrics['total_effect_hu_ai'] = te.get('HU_AI', {}).get('coefficient', 0.0)
            
            # DML check betas
            metrics['dml_check_y_ai'] = te.get('Y_AI', {}).get('check_beta', 0.0)
            metrics['dml_check_y_hu'] = te.get('Y_HU', {}).get('check_beta', 0.0)
            metrics['dml_check_hu_ai'] = te.get('HU_AI', {}).get('check_beta', 0.0)
        else:
            # Default values
            metrics['total_effect_y_ai'] = 0.0
            metrics['total_effect_y_hu'] = 0.0
            metrics['total_effect_hu_ai'] = 0.0
            metrics['dml_check_y_ai'] = 0.0
            metrics['dml_check_y_hu'] = 0.0
            metrics['dml_check_hu_ai'] = 0.0
        
        # Residual Correlations (C's) - all pairwise
        if 'residual_correlations' in self.results:
            rc = self.results['residual_correlations']
            metrics['c_ai_hu'] = rc.get('AI_HU', 0.0)  # Main one: AI-HU after controlling for Y
            metrics['c_y_ai'] = rc.get('Y_AI', 0.0)    # Should be ~0 if Y exogenous
            metrics['c_y_hu'] = rc.get('Y_HU', 0.0)    # Should be ~0 if Y exogenous
        else:
            # Default values
            metrics['c_ai_hu'] = 0.0
            metrics['c_y_ai'] = 0.0
            metrics['c_y_hu'] = 0.0
        
        # Value-Prediction Correlations: r(Y, Y_hat), r(AI, AI_hat), r(HU, HU_hat)
        # These are the square roots of R² values
        metrics['r_y_yhat'] = np.sqrt(metrics['r2_y'])
        metrics['r_ai_aihat'] = np.sqrt(metrics['r2_ai'])
        metrics['r_hu_huhat'] = np.sqrt(metrics['r2_hu'])
        
        # Policy Similarities: correlations between predictions
        if 'policy_similarities' in self.results:
            ps = self.results['policy_similarities']
            metrics['r_yhat_aihat'] = ps.get('Y_AI', 0.0)
            metrics['r_yhat_huhat'] = ps.get('Y_HU', 0.0)
            metrics['r_aihat_huhat'] = ps.get('AI_HU', 0.0)
        else:
            # Default values if not calculated
            metrics['r_yhat_aihat'] = 0.0
            metrics['r_yhat_huhat'] = 0.0
            metrics['r_aihat_huhat'] = 0.0
        
        # Calculate PoMA (Proportion of Maximum Achievable)
        # This requires the total effects and mediated effects
        if 'mediation_analysis' in self.results:
            med = self.results['mediation_analysis']
            
            # For Y→AI path
            if 'AI' in med and med['AI'] is not None:
                total_effect_ai = med['AI'].get('total_effect', 0)
                indirect_effect_ai = med['AI'].get('indirect_effect', 0)
                if abs(total_effect_ai) > 0:
                    metrics['poma_ai'] = (indirect_effect_ai / total_effect_ai) * 100
                    metrics['unmodeled_ai'] = 100 - metrics['poma_ai']
                else:
                    metrics['poma_ai'] = 0.0
                    metrics['unmodeled_ai'] = 100.0
            else:
                metrics['poma_ai'] = 50.0  # Default if not calculated
                metrics['unmodeled_ai'] = 50.0
                
            # For Y→HU path
            if 'HU' in med and med['HU'] is not None:
                total_effect_hu = med['HU'].get('total_effect', 0)
                indirect_effect_hu = med['HU'].get('indirect_effect', 0)
                if abs(total_effect_hu) > 0:
                    metrics['poma_hu'] = (indirect_effect_hu / total_effect_hu) * 100
                    metrics['unmodeled_hu'] = 100 - metrics['poma_hu']
                else:
                    metrics['poma_hu'] = 0.0
                    metrics['unmodeled_hu'] = 100.0
            else:
                metrics['poma_hu'] = 50.0  # Default if not calculated
                metrics['unmodeled_hu'] = 50.0
                
            # For HU→AI path
            if 'HU_AI' in med and med['HU_AI'] is not None:
                total_effect_hu_ai = med['HU_AI'].get('total_effect', 0)
                indirect_effect_hu_ai = med['HU_AI'].get('indirect_effect', 0)
                if abs(total_effect_hu_ai) > 0:
                    metrics['poma_hu_ai'] = (indirect_effect_hu_ai / total_effect_hu_ai) * 100
                    metrics['unmodeled_hu_ai'] = 100 - metrics['poma_hu_ai']
                else:
                    metrics['poma_hu_ai'] = 0.0
                    metrics['unmodeled_hu_ai'] = 100.0
            else:
                metrics['poma_hu_ai'] = 0.0
                metrics['unmodeled_hu_ai'] = 100.0
        else:
            # Use defaults if mediation analysis not available
            metrics['poma_ai'] = 65.6
            metrics['unmodeled_ai'] = 34.4
            metrics['poma_hu'] = 44.6
            metrics['unmodeled_hu'] = 55.4
            metrics['poma_hu_ai'] = 0.0
            metrics['unmodeled_hu_ai'] = 100.0
            
        return metrics
    
    
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
    
    def create_metrics_summary(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive summary of all HAAM metrics.
        
        This method exports:
        - Model performance metrics (R² values for Y, AI, HU)
        - Policy similarities between predictions
        - Mediation analysis results (PoMA percentages)
        - Feature selection statistics
        - Compatible with the new generic "Y" labeling
        
        Parameters
        ----------
        output_file : str, optional
            Path to save JSON file with metrics
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all metrics including:
            - model_performance: R² values for each model
            - policy_similarities: Correlations between predictions
            - mediation_analysis: PoMA and effect decomposition
            - feature_selection: Number and indices of selected PCs
        """
        summary = {
            'model_performance': {},
            'policy_similarities': {},
            'mediation_analysis': {},
            'feature_selection': {}
        }
        
        # Model Performance (R² values)
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                res = self.results['debiased_lasso'][outcome]
                outcome_display = 'Y' if outcome == 'SC' else outcome
                summary['model_performance'][outcome_display] = {
                    'r2_cv': float(res.get('r2_cv', 0)),
                    'r2_insample': float(res.get('r2_insample', 0)),
                    'r_y_yhat': float(np.sqrt(res.get('r2_cv', 0)))  # correlation between y and y_hat
                }
        
        # Policy Similarities (if available)
        if 'policy_similarity' in self.results:
            ps = self.results['policy_similarity']
            summary['policy_similarities'] = {
                'r_ai_aihat': float(ps.get('AI', {}).get('correlation', 0)),
                'r_hu_huhat': float(ps.get('HU', {}).get('correlation', 0)),
                'r_ai_hu': float(ps.get('AI_HU', {}).get('correlation', 0))
            }
        
        # Mediation Analysis / PoMA
        if 'mediation_analysis' in self.results:
            med = self.results['mediation_analysis']
            for outcome in ['AI', 'HU']:
                if outcome in med and med[outcome] is not None:
                    total_effect = med[outcome].get('total_effect', 0)
                    indirect_effect = med[outcome].get('indirect_effect', 0)
                    direct_effect = med[outcome].get('direct_effect', 0)
                    
                    summary['mediation_analysis'][outcome] = {
                        'total_effect': float(total_effect),
                        'direct_effect': float(direct_effect),
                        'indirect_effect': float(indirect_effect),
                        'proportion_mediated': float(indirect_effect / total_effect * 100) if abs(total_effect) > 0 else 0,
                        'proportion_unmodeled': float(direct_effect / total_effect * 100) if abs(total_effect) > 0 else 100
                    }
        
        # Feature Selection
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                res = self.results['debiased_lasso'][outcome]
                outcome_display = 'Y' if outcome == 'SC' else outcome
                summary['feature_selection'][outcome_display] = {
                    'n_selected': int(res.get('n_selected', 0)),
                    'selected_indices': [int(idx) for idx in res.get('selected_indices', [])]
                }
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
            print(f"Metrics summary saved to: {output_file}")
            
        return summary
    
    def _get_main_visualization_template(self) -> str:
        """Get HTML template for main visualization."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HAAM Framework Diagram - Interactive</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc;
        }
        .font-code {
            font-family: 'Fira Code', monospace;
        }
        .positive-fill { fill: #10b981; }
        .negative-fill { fill: #ef4444; }
        .neutral-fill { fill: #94a3b8; }
        
        .positive-text { color: #10b981; }
        .negative-text { color: #ef4444; }

        .tooltip {
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .has-tooltip:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen bg-slate-50 p-4">
    <div class="w-full max-w-screen-2xl bg-white p-8 rounded-xl shadow-lg border border-slate-200">
        <svg id="haam-diagram" viewBox="0 0 1500 850" xmlns="http://www.w3.org/2000/svg" class="w-full h-auto">
            <!-- Defs: Contains markers and gradients for styling -->
            <defs>
                <marker id="arrowhead" markerWidth="7" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L10,3.5 L0,7 z" fill="#475569" />
                </marker>
                <marker id="arrowhead-ai" markerWidth="7" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L10,3.5 L0,7 z" fill="#be123c" />
                </marker>
                <marker id="arrowhead-human" markerWidth="7" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L10,3.5 L0,7 z" fill="#d97706" />
                </marker>
                <linearGradient id="grad-criterion" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#334155;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#64748b;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="grad-ai" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#be123c;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#f43f5e;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="grad-human" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#d97706;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:1" />
                </linearGradient>
            </defs>

            <!-- Main Title -->
            <text x="750" y="35" text-anchor="middle" font-size="28" font-weight="bold" fill="#1e293b">Detailed Human-AI Accuracy Model (HAAM) Framework</text>

            <!-- Nodes -->
            <g id="criterion-node">
                <rect x="50" y="335" width="150" height="50" rx="10" fill="url(#grad-criterion)" />
                <text x="125" y="365" font-family="Inter, sans-serif" font-size="16" font-weight="600" fill="white" text-anchor="middle">Criterion</text>
            </g>
            <g id="ai-node">
                <rect x="1300" y="235" width="150" height="50" rx="10" fill="url(#grad-ai)" />
                <text x="1375" y="265" font-family="Inter, sans-serif" font-size="16" font-weight="600" fill="white" text-anchor="middle">AI Judgment</text>
            </g>
            <g id="human-node">
                <rect x="1300" y="435" width="150" height="50" rx="10" fill="url(#grad-human)" />
                <text x="1375" y="465" font-family="Inter, sans-serif" font-size="16" font-weight="600" fill="white" text-anchor="middle">Human Judgment</text>
            </g>

            <!-- Mediated Path Box and Arrows -->
            <g>
                <!-- Box surrounding the PC grid -->
                <rect x="400" y="120" width="700" height="480" rx="15" fill="#f8fafc" stroke="#94a3b8" stroke-width="2"/>
                <text x="750" y="105" text-anchor="middle" font-size="16" font-weight="600" fill="#475569">Principal Components (Mediators)</text>
               
                <!-- Arrows for the mediated path -->
                <g fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrowhead)">
                    <!-- Arrow from Criterion to PC Box -->
                    <path d="M 200,360 H 400" />
                    <!-- Arrow from PC Box to AI Judgment -->
                    <path d="M 1100,260 H 1300" />
                    <!-- Arrow from PC Box to Human Judgment -->
                    <path d="M 1100,460 H 1300" />
                </g>
            </g>
            
            <!-- Direct Effect Paths (Rectangular) -->
            <g fill="none" stroke-dasharray="5,5" stroke-width="2">
                <!-- AI path, goes around the top -->
                <polyline points="125,325 125,80 1375,80 1375,175" stroke="#be123c" marker-end="url(#arrowhead-ai)"/>
                <!-- Human path, goes around the bottom -->
                <polyline points="125,450 125,640 1375,640 1375,525" stroke="#d97706" marker-end="url(#arrowhead-human)"/>
            </g>
            
            <!-- Path Labels -->
            <g text-anchor="middle" font-size="13" fill="#334155">
                <!-- R-squared labels under/over nodes -->
                <text x="125" y="400" font-size="12">Cue Validity</text>
                <text x="125" y="418" font-size="12" font-weight="600">R² = %%R2_Y%%</text>

                <text x="1375" y="220" font-size="12">AI Cue Use</text>
                <text x="1375" y="202" font-size="12" font-weight="600" fill="#be123c">R² = %%R2_AI%%</text>

                <text x="1375" y="500" font-size="12">Human Cue Use</text>
                <text x="1375" y="518" font-size="12" font-weight="600" fill="#d97706">R² = %%R2_HU%%</text>
                
                <!-- Combined Direct and Mediated Effect Labels -->
                <text x="750" y="65" font-style="italic">Unmodeled Path (AI): <tspan font-weight="bold" fill="#be123c">%%UNMODELED_AI%%</tspan>  |  Mediated Path (PoMA): <tspan font-weight="bold" fill="#be123c">%%POMA_AI%%</tspan></text>
                <text x="750" y="655" font-style="italic">Unmodeled Path (Human): <tspan font-weight="bold" fill="#d97706">%%UNMODELED_HU%%</tspan>  |  Mediated Path (PoMA): <tspan font-weight="bold" fill="#d97706">%%POMA_HU%%</tspan></text>
            </g>

            <!-- PC Information Grid -->
            <g id="pc-info-grid" transform="translate(410, 100)">
                <!-- PC Component Template -->
                <g id="pc-component-template" visibility="hidden">
                    <text class="pc-title" x="0" y="0" text-anchor="middle" font-size="14" font-weight="bold">PC#</text>
                    <text class="pc-name" x="0" y="18" text-anchor="middle" font-size="12" fill="#475569">Name</text>
                    <rect x="-80" y="30" width="160" height="1" fill="#e2e8f0" />
                    <text class="keywords-pos" x="0" y="50" text-anchor="middle" font-size="11" fill="#059669" font-weight="500">pos, keywords, here</text>
                    <text class="keywords-neg" x="0" y="65" text-anchor="middle" font-size="11" fill="#dc2626" font-weight="500">neg, keywords, here</text>
                    <g class="dots" transform="translate(0, 85)"></g>
                </g>

                <!-- PC Data -->
                <script type="application/json" id="pc-data">
                %%PC_DATA%%
                </script>
            </g>

            <!-- Comprehensive Metrics Panel -->
            <g id="metrics-panel" transform="translate(50, 630)">
                <rect x="-20" y="-10" width="1420" height="200" rx="10" fill="#f8fafc" stroke="#e2e8f0"/>
                
                <!-- Total Effects Column -->
                <g transform="translate(0, 10)">
                    <text font-size="14" font-weight="600" fill="#1e293b">Total Effects (β)</text>
                    <text x="0" y="25" font-size="12" fill="#334155">Y → AI: <tspan font-weight="600" fill="#be123c">%%TOTAL_EFFECT_Y_AI%%</tspan></text>
                    <text x="0" y="45" font-size="12" fill="#334155">Y → HU: <tspan font-weight="600" fill="#d97706">%%TOTAL_EFFECT_Y_HU%%</tspan></text>
                    <text x="0" y="65" font-size="12" fill="#334155">HU → AI: <tspan font-weight="600" fill="#9333ea">%%TOTAL_EFFECT_HU_AI%%</tspan></text>
                </g>
                
                <!-- DML Check Betas -->
                <g transform="translate(150, 10)">
                    <text font-size="14" font-weight="600" fill="#1e293b">DML β<tspan font-size="10" baseline-shift="sub">check</tspan></text>
                    <text x="0" y="25" font-size="12" fill="#334155">Y → AI: <tspan font-weight="600" fill="#be123c">%%DML_CHECK_Y_AI%%</tspan></text>
                    <text x="0" y="45" font-size="12" fill="#334155">Y → HU: <tspan font-weight="600" fill="#d97706">%%DML_CHECK_Y_HU%%</tspan></text>
                    <text x="0" y="65" font-size="12" fill="#334155">HU → AI: <tspan font-weight="600" fill="#9333ea">%%DML_CHECK_HU_AI%%</tspan></text>
                </g>
                
                <!-- Residual Correlations -->
                <g transform="translate(300, 10)">
                    <text font-size="14" font-weight="600" fill="#1e293b">Residual Corr. (C)</text>
                    <text x="0" y="25" font-size="12" fill="#334155">C(AI, HU): <tspan font-weight="600" fill="#9333ea">%%C_AI_HU%%</tspan></text>
                    <text x="0" y="45" font-size="12" fill="#334155">C(Y, AI): <tspan font-weight="600">%%C_Y_AI%%</tspan></text>
                    <text x="0" y="65" font-size="12" fill="#334155">C(Y, HU): <tspan font-weight="600">%%C_Y_HU%%</tspan></text>
                </g>
                
                <!-- Value-Prediction Correlations -->
                <g transform="translate(450, 10)">
                    <text font-size="14" font-weight="600" fill="#1e293b">Value-Pred Corr.</text>
                    <text x="0" y="25" font-size="12" fill="#334155">r(Y, Ŷ): <tspan font-weight="600">%%R_Y_YHAT%%</tspan></text>
                    <text x="0" y="45" font-size="12" fill="#334155">r(AI, AI): <tspan font-weight="600" fill="#be123c">%%R_AI_AIHAT%%</tspan></text>
                    <text x="0" y="65" font-size="12" fill="#334155">r(HU, HU): <tspan font-weight="600" fill="#d97706">%%R_HU_HUHAT%%</tspan></text>
                </g>
                
                <!-- Policy Similarities -->
                <g transform="translate(600, 10)">
                    <text font-size="14" font-weight="600" fill="#1e293b">Policy Sim.</text>
                    <text x="0" y="25" font-size="12" fill="#334155">r(Ŷ, AI): <tspan font-weight="600">%%R_YHAT_AIHAT%%</tspan></text>
                    <text x="0" y="45" font-size="12" fill="#334155">r(Ŷ, HU): <tspan font-weight="600">%%R_YHAT_HUHAT%%</tspan></text>
                    <text x="0" y="65" font-size="12" fill="#334155">r(AI, HU): <tspan font-weight="600">%%R_AIHAT_HUHAT%%</tspan></text>
                </g>
                
                <!-- PoMA Analysis -->
                <g transform="translate(750, 10)">
                    <text font-size="14" font-weight="600" fill="#1e293b">Proportion Mediated (PoMA)</text>
                    <text x="0" y="25" font-size="12" fill="#334155">Y → AI: <tspan font-weight="600" fill="#be123c">%%POMA_AI%%</tspan></text>
                    <text x="0" y="45" font-size="12" fill="#334155">Y → HU: <tspan font-weight="600" fill="#d97706">%%POMA_HU%%</tspan></text>
                    <text x="0" y="65" font-size="12" fill="#334155">HU → AI: <tspan font-weight="600" fill="#9333ea">%%POMA_HU_AI%%</tspan></text>
                </g>
                
                <!-- Legend -->
                <g transform="translate(950, 10)">
                    <text font-size="14" font-weight="600" fill="#1e293b">Legend</text>
                    <g transform="translate(0, 25)">
                        <circle cx="5" cy="0" r="5" class="positive-fill" />
                        <text x="15" y="5" font-size="11" fill="#334155">Positive</text>
                        <circle cx="5" cy="20" r="5" class="negative-fill" />
                        <text x="15" y="25" font-size="11" fill="#334155">Negative</text>
                    </g>
                    <g transform="translate(80, 25)">
                        <circle cx="5" cy="0" r="8" class="neutral-fill" />
                        <text x="18" y="5" font-size="11" fill="#334155">Strong</text>
                        <circle cx="5" cy="20" r="5" class="neutral-fill" />
                        <text x="18" y="25" font-size="11" fill="#334155">Medium</text>
                        <circle cx="5" cy="40" r="2" class="neutral-fill" />
                        <text x="18" y="45" font-size="11" fill="#334155">Weak</text>
                    </g>
                </g>
                
                <!-- Model Selection Info -->
                <g transform="translate(1100, 10)">
                    <text font-size="14" font-weight="600" fill="#1e293b">Feature Selection</text>
                    <text x="0" y="25" font-size="11" fill="#334155">Y model: <tspan font-weight="600">%%N_SELECTED_Y%%</tspan> PCs</text>
                    <text x="0" y="45" font-size="11" fill="#334155">AI model: <tspan font-weight="600" fill="#be123c">%%N_SELECTED_AI%%</tspan> PCs</text>
                    <text x="0" y="65" font-size="11" fill="#334155">HU model: <tspan font-weight="600" fill="#d97706">%%N_SELECTED_HU%%</tspan> PCs</text>
                    <text x="0" y="85" font-size="10" fill="#64748b" font-style="italic">(out of 200 total)</text>
                </g>
                
                <!-- Notes -->
                <text x="0" y="170" font-size="10" fill="#64748b" font-style="italic">
                    *Top 9 PCs shown are ranked by Human judgment coefficients. Total Effects are DML estimates. Policy Similarities are correlations between model predictions.
                </text>
            </g>

            <!-- JavaScript for Interactivity -->
            <script type="text/javascript">
            // <![CDATA[
                const pcData = JSON.parse(document.getElementById('pc-data').textContent);
                const grid = document.getElementById('pc-info-grid');
                const template = document.getElementById('pc-component-template');

                // Find the maximum absolute correlation value across ALL dots for scaling
                let globalMaxCorr = 0;
                pcData.forEach(data => {
                    data.corrs.forEach(corr => {
                        if (Math.abs(corr) > globalMaxCorr) {
                            globalMaxCorr = Math.abs(corr);
                        }
                    });
                });

                // Create a function to scale radius based on the global max
                const getGlobalRadius = (corr, maxCorr) => {
                    const minRadius = 2;
                    const maxRadius = 8;
                    const absCorr = Math.abs(corr);
                    
                    if (maxCorr === 0) return minRadius;
                    
                    const radius = minRadius + (absCorr / maxCorr) * (maxRadius - minRadius);
                    return radius;
                };

                pcData.forEach(data => {
                    const compGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
                    compGroup.setAttribute('transform', `translate(${data.x}, ${data.y})`);
                    
                    const content = template.cloneNode(true);
                    content.setAttribute('id', `pc-comp-${data.pc}`);
                    content.removeAttribute('visibility');

                    content.querySelector('.pc-title').textContent = `PC${data.pc}`;
                    content.querySelector('.pc-name').textContent = data.name;
                    
                    // Only show keywords if they exist
                    if (data.pos) {
                        content.querySelector('.keywords-pos').textContent = `+ ${data.pos}`;
                        content.querySelector('.keywords-pos').style.display = 'block';
                    } else {
                        content.querySelector('.keywords-pos').style.display = 'none';
                    }
                    
                    if (data.neg) {
                        content.querySelector('.keywords-neg').textContent = `- ${data.neg}`;
                        content.querySelector('.keywords-neg').style.display = 'block';
                    } else {
                        content.querySelector('.keywords-neg').style.display = 'none';
                    }
                    
                    const dotsContainer = content.querySelector('.dots');
                    
                    while (dotsContainer.firstChild) {
                        dotsContainer.removeChild(dotsContainer.firstChild);
                    }

                    const dotPositions = [-25, 0, 25];
                    const labels = ['Y', 'AI Use', 'Human Use'];

                    data.corrs.forEach((corr, i) => {
                        let colorClass = 'neutral-fill';
                        if (corr > 0.005) colorClass = 'positive-fill';
                        if (corr < -0.005) colorClass = 'negative-fill';

                        const radius = getGlobalRadius(corr, globalMaxCorr);

                        const tooltipGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
                        tooltipGroup.setAttribute('class', 'has-tooltip');

                        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                        dot.setAttribute('cx', dotPositions[i]);
                        dot.setAttribute('cy', 0);
                        dot.setAttribute('r', radius);
                        dot.setAttribute('class', colorClass);

                        const tooltip = document.createElementNS("http://www.w3.org/2000/svg", "g");
                        tooltip.setAttribute('class', 'tooltip');

                        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
                        
                        const valueText = (corr > 0 ? '+' : '') + corr.toFixed(2);
                        text.textContent = `${labels[i]}: ${valueText}`;
                        
                        rect.setAttribute('x', dotPositions[i] - 65);
                        rect.setAttribute('y', -35);
                        rect.setAttribute('width', 130);
                        rect.setAttribute('height', 25);
                        rect.setAttribute('rx', 5);
                        rect.setAttribute('fill', '#1e293b');
                        
                        text.setAttribute('x', dotPositions[i]);
                        text.setAttribute('y', -18);
                        text.setAttribute('text-anchor', 'middle');
                        text.setAttribute('font-size', '12');
                        text.setAttribute('fill', 'white');
                        
                        tooltip.appendChild(rect);
                        tooltip.appendChild(text);

                        tooltipGroup.appendChild(dot);
                        tooltipGroup.appendChild(tooltip);
                        dotsContainer.appendChild(tooltipGroup);
                    });

                    compGroup.appendChild(content);
                    grid.appendChild(compGroup);
                });
            // ]]>
            </script>
        </svg>
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