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
                 auto_run: bool = True,
                 min_cluster_size: int = 10,
                 min_samples: int = 2,
                 umap_n_components: int = 3,
                 standardize: bool = False,
                 sample_split_post_lasso: bool = True):
        """
        Initialize HAAM analysis with enhanced parameters.
        
        Recent updates:
        - UMAP: n_neighbors=5, min_dist=0.0, metric='cosine'
        - HDBSCAN: min_cluster_size=10, min_samples=2
        - c-TF-IDF implementation with BERTopic formula
        - Generic "Y" labeling instead of "SC" in visualizations
        
        Parameters
        ----------
        criterion : array-like
            Criterion variable (any ground truth variable, not limited to social class)
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
        min_cluster_size : int, default=10
            Minimum cluster size for HDBSCAN (matches BERTopic-style clustering)
        min_samples : int, default=2
            Minimum samples for core points in HDBSCAN
        umap_n_components : int, default=3
            Number of UMAP components for clustering (3D by default)
        standardize : bool, default=False
            Whether to standardize X and Y variables for both total effects and DML calculations.
            When True, all coefficients will be in standardized units.
        sample_split_post_lasso : bool, default=True
            Whether to use sample splitting for post-LASSO inference.
            True: Conservative inference with valid p-values (original behavior)
            False: Maximum statistical power using full sample
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
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_components = umap_n_components
        self.standardize = standardize
        self.sample_split_post_lasso = sample_split_post_lasso
        
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
            n_components=self.n_components,
            standardize=self.standardize
        )
        
        # Step 2: Fit debiased lasso
        print("\n2. Fitting debiased lasso models...")
        self.analysis.fit_debiased_lasso(use_sample_splitting=self.sample_split_post_lasso)
        
        # Step 3: Topic analysis (if texts provided)
        if self.texts is not None:
            print("\n3. Performing topic analysis...")
            self.topic_analyzer = TopicAnalyzer(
                texts=self.texts,
                embeddings=self.analysis.embeddings,
                pca_features=self.analysis.results['pca_features'],
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                umap_n_components=self.umap_n_components
            )
            
            # Get ALL PCs (not just top ones)
            all_pcs = list(range(self.n_components))
            
            # Get topic summaries for ALL PCs with many topics
            self.topic_summaries = self.topic_analyzer.create_topic_summary_for_pcs(
                all_pcs,  # All 200 PCs
                n_keywords=10,  # More keywords per topic
                n_topics_per_side=30  # Top 30 and bottom 30
            )
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
    
    def create_main_visualization(self, 
                                    output_dir: Optional[str] = None,
                                    pc_names: Optional[Dict[int, str]] = None) -> str:
        """
        Create and save main HAAM visualization.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save output. If None, uses current directory
        pc_names : Dict[int, str], optional
            Manual names for PCs. Keys are PC indices (0-based), values are names.
            Example: {4: "Lifestyle & Work", 7: "Professions", 1: "Narrative Style"}
            
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
        
        # Default to Human ranking (HU) for visualization
        top_pcs = self.analysis.get_top_pcs(n_top=9, ranking_method='HU')
        self.visualizer.create_main_visualization(top_pcs, output_file, pc_names)
        
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
    
    def create_metrics_summary(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create and save comprehensive metrics summary.
        
        Exports all key metrics including:
        - Model performance (R² values for Y, AI, HU)
        - Policy similarities between predictions
        - Mediation analysis (PoMA percentages)
        - Feature selection statistics
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save output. If None, uses current directory
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all metrics (also saved to JSON file)
        """
        if self.visualizer is None:
            raise RuntimeError("Must run analysis first")
            
        if output_dir is None:
            output_dir = os.getcwd()
            
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'haam_metrics_summary.json')
        
        metrics = self.visualizer.create_metrics_summary(output_file)
        
        return metrics
    
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
        
        # Compute UMAP (matching my_colab.py parameters)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=5,  # Changed from 15 to match my_colab.py
            min_dist=0.0,  # Changed from 0.1 to match my_colab.py
            metric='cosine',  # Changed from euclidean to match my_colab.py
            random_state=42
        )
        
        # Use embeddings directly for UMAP (not PCA features)
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
    
    def visualize_pc_umap_with_topics(self,
                                     pc_idx: int,
                                     output_dir: Optional[str] = None,
                                     show_top_n: int = 5,
                                     show_bottom_n: int = 5,
                                     display: bool = True) -> str:
        """
        Create UMAP visualization for a specific PC with topic labels.
        
        Parameters
        ----------
        pc_idx : int
            PC index (0-based). E.g., 0 for PC1, 4 for PC5
        output_dir : str, optional
            Directory to save output. If None, saves to current directory
        show_top_n : int
            Number of high-scoring topics to label
        show_bottom_n : int
            Number of low-scoring topics to label
        display : bool
            Whether to display in notebook/colab
            
        Returns
        -------
        str
            Path to saved HTML file
        """
        if not hasattr(self, 'topic_analyzer') or self.topic_analyzer is None:
            raise ValueError("Topic analysis not performed. Initialize with texts to enable topic analysis.")
        
        # Get PC associations for this specific PC
        pc_associations = self.topic_analyzer.get_pc_topic_associations([pc_idx])
        
        # Set up output path
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'pc{pc_idx + 1}_umap_with_topics.html')
        
        # Create visualization
        fig = self.visualizer.create_pc_umap_with_topics(
            pc_idx=pc_idx,
            pc_scores=self.analysis.results['pca_features'][:, pc_idx],
            umap_embeddings=self.topic_analyzer.umap_embeddings,
            cluster_labels=self.topic_analyzer.cluster_labels,
            topic_keywords=self.topic_analyzer.topic_keywords,
            pc_associations=pc_associations,
            output_file=output_file,
            show_top_n=show_top_n,
            show_bottom_n=show_bottom_n,
            display=display
        )
        
        return output_file
    
    def create_all_pc_umap_visualizations(self,
                                         pc_indices: Optional[List[int]] = None,
                                         output_dir: Optional[str] = None,
                                         show_top_n: int = 5,
                                         show_bottom_n: int = 5,
                                         display: bool = False) -> Dict[int, str]:
        """
        Create UMAP visualizations for multiple PCs with topic labels.
        
        Parameters
        ----------
        pc_indices : List[int], optional
            List of PC indices to visualize. If None, uses top 10 PCs
        output_dir : str, optional
            Directory to save visualizations
        show_top_n : int
            Number of high topics to show per PC
        show_bottom_n : int
            Number of low topics to show per PC
        display : bool
            Whether to display each plot (set False for batch processing)
            
        Returns
        -------
        Dict[int, str]
            Mapping of PC index to output file path
        """
        if not hasattr(self, 'topic_analyzer') or self.topic_analyzer is None:
            raise ValueError("Topic analysis not performed. Initialize with texts to enable topic analysis.")
        
        # Default to top 10 PCs if not specified
        if pc_indices is None:
            pc_indices = self.analysis.get_top_pcs(n=10, ranking_method='triple')
        
        # Get associations for all requested PCs
        pc_associations = self.topic_analyzer.get_pc_topic_associations(pc_indices)
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'pc_umap_visualizations')
        
        # Create all visualizations
        output_files = self.visualizer.create_all_pc_umap_visualizations(
            pc_indices=pc_indices,
            pc_scores_all=self.analysis.results['pca_features'],
            umap_embeddings=self.topic_analyzer.umap_embeddings,
            cluster_labels=self.topic_analyzer.cluster_labels,
            topic_keywords=self.topic_analyzer.topic_keywords,
            pc_associations=pc_associations,
            output_dir=output_dir,
            show_top_n=show_top_n,
            show_bottom_n=show_bottom_n,
            display=display
        )
        
        return output_files
    
    def create_3d_umap_with_pc_arrows(self,
                                     pc_indices: Optional[Union[int, List[int]]] = None,
                                     top_k: int = 1,
                                     percentile_threshold: float = 90.0,
                                     arrow_mode: str = 'all',
                                     color_by_usage: bool = True,
                                     show_topic_labels: Union[bool, int] = 10,
                                     output_dir: Optional[str] = None,
                                     display: bool = True) -> str:
        """
        Create 3D UMAP visualization with PC directional arrows.
        
        This creates an interactive 3D scatter plot where:
        - Topics are positioned in 3D UMAP space based on their semantic similarity
        - Arrows show PC directions from average of bottom-k to top-k topics
        - Topics are colored based on HU/AI usage patterns (quartiles)
        
        The key insight: In UMAP space, PC gradients often form linear patterns,
        allowing us to visualize how principal components map to topic space.
        
        Parameters
        ----------
        pc_indices : int or List[int], optional
            PC indices to show arrows for (0-based).
            - If None and arrow_mode='all': shows arrows for PC1, PC2, PC3
            - If int: shows arrow for that single PC
            - If list: shows arrows for all PCs in the list
        top_k : int, default=1
            Number of top/bottom scoring topics to average for arrow endpoints.
            Default=1 for cleaner single-topic arrows.
            If fewer topics meet the threshold, uses all available.
        percentile_threshold : float, default=90.0
            Percentile threshold for selecting top/bottom topics.
            90.0 means top 10% and bottom 10% of topics.
        arrow_mode : str, default='all'
            Controls which arrows to display:
            - 'single': Show arrow for single PC
            - 'list': Show arrows for specified list of PCs
            - 'all': Show arrows for first 3 PCs
        color_by_usage : bool, default=True
            Whether to color topics by HU/AI usage patterns:
            - Dark red: All three (HU, AI, Y) in top quartile
            - Red: HU & AI both in top quartile
            - Dark blue: All three in bottom quartile
            - Blue: HU & AI both in bottom quartile
            - Gray: Mixed patterns
        show_topic_labels : bool or int, default=10
            Controls topic label display:
            - True: Show all topic labels
            - False: Hide all labels (hover still works)
            - int: Show only N closest topics to camera
        output_dir : str, optional
            Directory to save output. If None, uses current directory
        display : bool, default=True
            Whether to display in notebook/colab
            
        Returns
        -------
        str
            Path to saved HTML file
            
        Examples
        --------
        # Show arrows for first 3 PCs with default settings
        haam.create_3d_umap_with_pc_arrows()
        
        # Show arrow only for PC5, using top/bottom 3 topics
        haam.create_3d_umap_with_pc_arrows(pc_indices=4, top_k=3, arrow_mode='single')
        
        # Show arrows for specific PCs with stricter threshold
        haam.create_3d_umap_with_pc_arrows(
            pc_indices=[0, 3, 7], 
            percentile_threshold=95.0,  # Top/bottom 5%
            arrow_mode='list'
        )
        
        # Use more topics for more stable arrow directions
        haam.create_3d_umap_with_pc_arrows(top_k=10, percentile_threshold=80.0)
        """
        if not hasattr(self, 'topic_analyzer') or self.topic_analyzer is None:
            raise ValueError("Topic analysis not performed. Initialize with texts to enable topic analysis.")
        
        if self.visualizer is None:
            raise RuntimeError("Must run analysis first")
        
        # Set up output path
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on options
        if isinstance(pc_indices, int):
            filename = f'3d_umap_pc{pc_indices+1}_arrow.html'
        elif isinstance(pc_indices, list):
            pc_str = '_'.join([str(idx+1) for idx in pc_indices[:3]])  # Limit filename length
            filename = f'3d_umap_pc{pc_str}_arrows.html'
        else:
            filename = '3d_umap_with_pc_arrows.html'
        
        output_file = os.path.join(output_dir, filename)
        
        # Create the visualization
        fig = self.visualizer.create_3d_umap_with_pc_arrows(
            umap_embeddings=self.topic_analyzer.umap_embeddings,
            cluster_labels=self.topic_analyzer.cluster_labels,
            topic_keywords=self.topic_analyzer.topic_keywords,
            pc_scores_all=self.analysis.results['pca_features'],
            pc_indices=pc_indices,
            top_k=top_k,
            percentile_threshold=percentile_threshold,
            arrow_mode=arrow_mode,
            color_by_usage=color_by_usage,
            show_topic_labels=show_topic_labels,
            output_file=output_file,
            display=display
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