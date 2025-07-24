"""
HAAM Package - Simplified API
=============================

Main module providing a simplified interface for HAAM analysis.
"""

from .haam_package import HAAMAnalysis
from .haam_topics import TopicAnalyzer
from .haam_visualizations import HAAMVisualizer
import numpy as np
import pandas as pd
import umap
import os
from typing import Dict, List, Optional, Tuple, Union, Any


class HAAM:
    """
    Simplified interface for HAAM analysis.
    
    This class provides an easy-to-use API for performing the full
    HAAM analysis pipeline including statistical analysis, topic modeling,
    and visualization generation.
    """
    
    def __init__(self, 
                 criterion: Union[np.ndarray, pd.Series, list],
                 ai_judgment: Union[np.ndarray, pd.Series, list],
                 human_judgment: Union[np.ndarray, pd.Series, list],
                 embeddings: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 texts: Optional[List[str]] = None,
                 n_components: int = 200,
                 auto_run: bool = True):
        """
        Initialize HAAM analysis.
        
        Parameters
        ----------
        criterion : array-like
            Criterion variable (e.g., social class)
        ai_judgment : array-like
            AI predictions/ratings
        human_judgment : array-like
            Human ratings
        embeddings : array-like, optional
            Pre-computed embeddings. If None, will be generated from texts
        texts : List[str], optional
            Text data for generating embeddings if not provided
        n_components : int, default=200
            Number of PCA components
        auto_run : bool, default=True
            Whether to automatically run the full analysis
        """
        # Convert inputs to numpy arrays
        self.criterion = self._to_numpy(criterion)
        self.ai_judgment = self._to_numpy(ai_judgment)
        self.human_judgment = self._to_numpy(human_judgment)
        
        if embeddings is not None:
            self.embeddings = self._to_numpy(embeddings)
        else:
            self.embeddings = None
            
        self.texts = texts
        self.n_components = n_components
        
        # Initialize components
        self.analysis = None
        self.topic_analyzer = None
        self.visualizer = None
        self.results = {}
        
        if auto_run:
            self.run_full_analysis()
    
    @staticmethod
    def _to_numpy(data: Union[np.ndarray, pd.Series, pd.DataFrame, list]) -> np.ndarray:
        """Convert various input types to numpy array."""
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        else:
            return data
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the complete HAAM analysis pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all results
        """
        print("="*60)
        print("Running HAAM Analysis")
        print("="*60)
        
        # Step 1: Core analysis
        print("\n1. Initializing core analysis...")
        self.analysis = HAAMAnalysis(
            criterion=self.criterion,
            ai_judgment=self.ai_judgment,
            human_judgment=self.human_judgment,
            embeddings=self.embeddings,
            texts=self.texts,
            n_components=self.n_components
        )
        
        # Step 2: Fit debiased lasso
        print("\n2. Fitting debiased lasso models...")
        self.analysis.fit_debiased_lasso(use_sample_splitting=True)
        
        # Step 3: Topic analysis (if texts provided)
        if self.texts is not None:
            print("\n3. Performing topic analysis...")
            self.topic_analyzer = TopicAnalyzer(
                texts=self.texts,
                embeddings=self.analysis.embeddings,
                pca_features=self.analysis.results['pca_features']
            )
            
            # Get top PCs
            top_pcs = self.analysis.get_top_pcs(n_top=9, ranking_method='triple')
            
            # Get topic summaries
            self.topic_summaries = self.topic_analyzer.create_topic_summary_for_pcs(top_pcs)
        else:
            print("\n3. Skipping topic analysis (no texts provided)")
            self.topic_summaries = {}
        
        # Step 4: Initialize visualizer
        print("\n4. Preparing visualizations...")
        self.visualizer = HAAMVisualizer(
            haam_results=self.analysis.results,
            topic_summaries=self.topic_summaries
        )
        
        # Compile results
        self.results = {
            'coefficients': self._get_coefficient_df(),
            'model_summary': self._get_model_summary(),
            'top_pcs': self.analysis.get_top_pcs(n_top=9),
            'analysis': self.analysis,
            'topic_analyzer': self.topic_analyzer,
            'visualizer': self.visualizer
        }
        
        print("\n✓ Analysis complete!")
        print("="*60)
        
        return self.results
    
    def _get_coefficient_df(self) -> pd.DataFrame:
        """Get coefficient dataframe."""
        data = []
        for i in range(self.n_components):
            row = {'PC': i + 1}
            for outcome in ['SC', 'AI', 'HU']:
                if outcome in self.analysis.results['debiased_lasso']:
                    res = self.analysis.results['debiased_lasso'][outcome]
                    row[f'{outcome}_coef'] = res['coefs_std'][i]
                    row[f'{outcome}_se'] = res['ses_std'][i]
                else:
                    row[f'{outcome}_coef'] = 0
                    row[f'{outcome}_se'] = 0
            data.append(row)
        return pd.DataFrame(data)
    
    def _get_model_summary(self) -> pd.DataFrame:
        """Get model summary dataframe."""
        data = []
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.analysis.results['debiased_lasso']:
                res = self.analysis.results['debiased_lasso'][outcome]
                data.append({
                    'Outcome': outcome,
                    'N_selected': res['n_selected'],
                    'R2_insample': res['r2_insample'],
                    'R2_cv': res['r2_cv']
                })
        return pd.DataFrame(data)
    
    def create_main_visualization(self, output_dir: Optional[str] = None) -> str:
        """
        Create and save main HAAM visualization.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save output. If None, uses current directory
            
        Returns
        -------
        str
            Path to saved HTML file
        """
        if self.visualizer is None:
            raise RuntimeError("Must run analysis first")
            
        if output_dir is None:
            output_dir = os.getcwd()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'haam_main_visualization.html')
        
        top_pcs = self.analysis.get_top_pcs(n_top=9)
        self.visualizer.create_main_visualization(top_pcs, output_file)
        
        return output_file
    
    def create_mini_grid(self, output_dir: Optional[str] = None) -> str:
        """
        Create and save mini grid visualization.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save output. If None, uses current directory
            
        Returns
        -------
        str
            Path to saved HTML file
        """
        if self.visualizer is None:
            raise RuntimeError("Must run analysis first")
            
        if output_dir is None:
            output_dir = os.getcwd()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'haam_mini_grid.html')
        
        self.visualizer.create_mini_visualization(
            n_components=self.n_components,
            output_file=output_file
        )
        
        return output_file
    
    def create_pc_effects_visualization(self, 
                                      output_dir: Optional[str] = None,
                                      n_top: int = 20) -> str:
        """
        Create PC effects bar chart visualization.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save output. If None, uses current directory
        n_top : int
            Number of top PCs to display
            
        Returns
        -------
        str
            Path to saved HTML file
        """
        if self.visualizer is None:
            raise RuntimeError("Must run analysis first")
            
        if output_dir is None:
            output_dir = os.getcwd()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'haam_pc_effects.html')
        
        # Get top PCs
        top_pcs = self.analysis.get_top_pcs(n_top=n_top, ranking_method='triple')
        
        # Create the visualization
        self.visualizer.create_pc_effects_plot(
            pc_indices=top_pcs,
            output_file=output_file
        )
        
        return output_file
    
    def explore_pc_topics(self, 
                         pc_indices: Optional[List[int]] = None,
                         n_topics: int = 10) -> pd.DataFrame:
        """
        Explore topic associations for specified PCs.
        
        Parameters
        ----------
        pc_indices : List[int], optional
            PC indices to explore (0-based). If None, uses top PCs
        n_topics : int
            Number of topics to show per PC
            
        Returns
        -------
        pd.DataFrame
            DataFrame with topic information
        """
        if self.topic_analyzer is None:
            raise RuntimeError("Topic analysis not available (no texts provided)")
            
        if pc_indices is None:
            pc_indices = self.analysis.get_top_pcs(n_top=9)
            
        associations = self.topic_analyzer.get_pc_topic_associations(pc_indices)
        
        # Convert to dataframe
        data = []
        for pc_idx in pc_indices:
            pc_assocs = associations[pc_idx]
            
            # Get top and bottom topics
            high_topics = [a for a in pc_assocs if a['avg_percentile'] > 75][:n_topics//2]
            low_topics = [a for a in pc_assocs if a['avg_percentile'] < 25][-n_topics//2:]
            
            for assoc in high_topics + low_topics:
                data.append({
                    'PC': pc_idx + 1,
                    'Direction': 'HIGH' if assoc['avg_percentile'] > 50 else 'LOW',
                    'Topic_ID': assoc['topic_id'],
                    'Keywords': assoc['keywords'],
                    'Size': assoc['size'],
                    'Percentile': assoc['avg_percentile'],
                    'Effect_Size': assoc['effect_size'],
                    'P_Value': assoc['p_value']
                })
                
        return pd.DataFrame(data)
    
    def plot_pc_effects(self, pc_idx: int, save_path: Optional[str] = None):
        """
        Create bar plot showing PC effects on outcomes.
        
        Parameters
        ----------
        pc_idx : int
            PC index (0-based)
        save_path : str, optional
            Path to save figure
        """
        if self.visualizer is None:
            raise RuntimeError("Must run analysis first")
            
        if self.topic_analyzer:
            associations = self.topic_analyzer.get_pc_topic_associations([pc_idx])
        else:
            associations = {pc_idx: []}
            
        fig = self.visualizer.plot_pc_effects(pc_idx, associations)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_umap_visualization(self,
                                 n_components: int = 3,
                                 color_by: str = 'SC',
                                 output_dir: Optional[str] = None) -> str:
        """
        Create UMAP visualization.
        
        Parameters
        ----------
        n_components : int
            Number of UMAP components (2 or 3)
        color_by : str
            Variable to color by: 'SC', 'AI', 'HU', or 'PC1', 'PC2', etc.
        output_dir : str, optional
            Directory to save output
            
        Returns
        -------
        str
            Path to saved HTML file
        """
        print(f"\nComputing {n_components}D UMAP...")
        
        # Compute UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        
        umap_embeddings = reducer.fit_transform(self.analysis.embeddings)
        
        # Create visualization
        if output_dir is None:
            output_dir = os.getcwd()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'haam_umap_{n_components}d_{color_by}.html')
        
        # Need to pass the color values to the visualizer
        self.visualizer.criterion = self.criterion
        self.visualizer.ai_judgment = self.ai_judgment
        self.visualizer.human_judgment = self.human_judgment
        
        fig = self.visualizer.create_umap_visualization(
            umap_embeddings=umap_embeddings,
            color_by=color_by,
            topic_labels=self.topic_analyzer.topic_keywords if self.topic_analyzer else None,
            output_file=output_file
        )
        
        return output_file
    
    def export_all_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export all results and create all visualizations.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save all outputs
            
        Returns
        -------
        Dict[str, str]
            Dictionary of all output file paths
        """
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'haam_output')
            
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExporting all results to: {output_dir}")
        
        outputs = {}
        
        # Export CSVs
        csv_paths = self.analysis.export_results(output_dir)
        outputs.update(csv_paths)
        
        # Export topic exploration if available
        if self.topic_analyzer:
            topic_df = self.explore_pc_topics()
            topic_path = os.path.join(output_dir, 'pc_topic_exploration.csv')
            topic_df.to_csv(topic_path, index=False)
            outputs['topics'] = topic_path
        
        # Create visualizations
        outputs['main_viz'] = self.create_main_visualization(output_dir)
        outputs['mini_grid'] = self.create_mini_grid(output_dir)
        
        # Create UMAP visualizations
        for color_by in ['SC', 'AI', 'HU']:
            outputs[f'umap_3d_{color_by}'] = self.create_umap_visualization(
                n_components=3, 
                color_by=color_by, 
                output_dir=output_dir
            )
        
        print("\n✓ All results exported successfully!")
        
        return outputs


# ===============================================
# EXAMPLE USAGE
# ===============================================

def example_usage():
    """
    Example usage of the HAAM package.
    """
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate some data
    criterion = np.random.normal(3, 1, n_samples)  # Social class 1-5
    ai_judgment = criterion + np.random.normal(0, 0.5, n_samples)  # AI correlates with SC
    human_judgment = criterion + np.random.normal(0, 0.7, n_samples)  # Human also correlates
    
    # Generate example texts
    texts = [f"This is example text {i} about various topics." for i in range(n_samples)]
    
    # Option 1: Simple usage with auto-run
    print("Example 1: Simple usage")
    print("-" * 40)
    
    haam = HAAM(
        criterion=criterion,
        ai_judgment=ai_judgment,
        human_judgment=human_judgment,
        texts=texts,
        n_components=50,  # Fewer components for example
        auto_run=True
    )
    
    # Access results
    print("\nModel Summary:")
    print(haam.results['model_summary'])
    
    print("\nTop PCs:")
    print(haam.results['top_pcs'])
    
    # Export everything
    output_paths = haam.export_all_results('./haam_example_output')
    
    print("\nOutput files created:")
    for key, path in output_paths.items():
        print(f"  {key}: {path}")
    
    # Option 2: Using pre-computed embeddings
    print("\n\nExample 2: Using pre-computed embeddings")
    print("-" * 40)
    
    # Generate embeddings separately
    embeddings = HAAMAnalysis.generate_embeddings(texts[:100])  # Just first 100 for speed
    
    haam2 = HAAM(
        criterion=criterion[:100],
        ai_judgment=ai_judgment[:100],
        human_judgment=human_judgment[:100],
        embeddings=embeddings,
        n_components=50,
        auto_run=True
    )
    
    # Explore specific PC
    pc_topics = haam2.explore_pc_topics(pc_indices=[0, 1, 2])
    print("\nPC Topic Exploration:")
    print(pc_topics.head())
    
    # Create PC effects plot
    fig = haam2.plot_pc_effects(pc_idx=0)
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    example_usage()