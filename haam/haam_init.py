"""
HAAM Package - Simplified API
=============================

Main module providing a simplified interface for HAAM analysis.
"""

from .haam_package import HAAMAnalysis
from .haam_topics import TopicAnalyzer
from .haam_visualizations import HAAMVisualizer
from .haam_wordcloud import PCWordCloudGenerator
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
        - Generic "X" labeling instead of "SC" in visualizations
        
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
            Whether to standardize X and outcome variables for both total effects and DML calculations.
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
            for outcome in ['X', 'AI', 'HU']:
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
        for outcome in ['X', 'AI', 'HU']:
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
        - Model performance (R² values for X, AI, HU)
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
                                 color_by: str = 'X',
                                 output_dir: Optional[str] = None) -> str:
        """
        Create UMAP visualization.
        
        Parameters
        ----------
        n_components : int
            Number of UMAP components (2 or 3)
        color_by : str
            Variable to color by: 'X', 'AI', 'HU', or 'PC1', 'PC2', etc.
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
                                     color_mode: str = 'legacy',
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
            Whether to color topics by HU/AI usage patterns
        color_mode : str, default='legacy'
            Coloring mode when color_by_usage=True:
            - 'legacy': Use PC coefficient-based inference (original behavior)
            - 'validity': Use direct X/HU/AI measurement (consistent with word clouds)
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
        # Show arrows with new validity coloring (consistent with word clouds)
        haam.create_3d_umap_with_pc_arrows(color_mode='validity')
        
        # Use legacy PC-based coloring (default)
        haam.create_3d_umap_with_pc_arrows(color_mode='legacy')
        
        # Show arrow only for PC5 with validity coloring
        haam.create_3d_umap_with_pc_arrows(
            pc_indices=4, 
            arrow_mode='single',
            color_mode='validity'
        )
        
        # Show arrows for specific PCs with stricter threshold
        haam.create_3d_umap_with_pc_arrows(
            pc_indices=[0, 3, 7], 
            percentile_threshold=95.0,  # Top/bottom 5%
            arrow_mode='list',
            color_mode='validity'
        )
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
        mode_suffix = '_validity' if color_mode == 'validity' else ''
        if isinstance(pc_indices, int):
            filename = f'3d_umap_pc{pc_indices+1}_arrow{mode_suffix}.html'
        elif isinstance(pc_indices, list):
            pc_str = '_'.join([str(idx+1) for idx in pc_indices[:3]])  # Limit filename length
            filename = f'3d_umap_pc{pc_str}_arrows{mode_suffix}.html'
        else:
            filename = f'3d_umap_with_pc_arrows{mode_suffix}.html'
        
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
            color_mode=color_mode,
            criterion=self.criterion if color_mode == 'validity' else None,
            human_judgment=self.human_judgment if color_mode == 'validity' else None,
            ai_judgment=self.ai_judgment if color_mode == 'validity' else None,
            show_topic_labels=show_topic_labels,
            output_file=output_file,
            display=display
        )
        
        return output_file
    
    def create_pc_wordclouds(self,
                           pc_idx: int,
                           k: int = 10,
                           max_words: int = 100,
                           figsize: Tuple[int, int] = (10, 5),
                           output_dir: Optional[str] = None,
                           display: bool = True,
                           color_mode: str = 'pole') -> Tuple[Any, str, str]:
        """
        Create word clouds for high and low poles of a specific PC.
        
        Parameters
        ----------
        pc_idx : int
            PC index (0-based)
        k : int
            Number of topics to include from each pole
        max_words : int
            Maximum words to display in word cloud
        figsize : Tuple[int, int]
            Figure size (width, height) for each subplot
        output_dir : str, optional
            Directory to save output files
        display : bool
            Whether to display the plots
        color_mode : str, optional
            'pole' (default): Red for high pole, blue for low pole
            'validity': Color based on X/HU/AI agreement:
                - Dark red: top quartile for all (X, HU, AI)
                - Light red: top quartile for HU & AI only
                - Dark blue: bottom quartile for all
                - Light blue: bottom quartile for HU & AI only
                - Grey: mixed signals
            
        Returns
        -------
        Tuple[Figure, str, str]
            Figure object, high pole output path, low pole output path
        """
        if not hasattr(self, 'topic_analyzer') or self.topic_analyzer is None:
            raise ValueError("Topic analysis not performed. Initialize with texts to enable topic analysis.")
            
        if not hasattr(self, 'wordcloud_generator'):
            self.wordcloud_generator = PCWordCloudGenerator(
                self.topic_analyzer, 
                self.analysis.results,
                criterion=self.criterion,
                human_judgment=self.human_judgment,
                ai_judgment=self.ai_judgment
            )
            
        return self.wordcloud_generator.create_pc_wordclouds(
            pc_idx=pc_idx,
            k=k,
            max_words=max_words,
            figsize=figsize,
            output_dir=output_dir,
            display=display,
            color_mode=color_mode
        )
    
    def create_all_pc_wordclouds(self,
                               pc_indices: Optional[List[int]] = None,
                               k: int = 10,
                               max_words: int = 100,
                               figsize: Tuple[int, int] = (10, 5),
                               output_dir: str = './wordclouds',
                               display: bool = False,
                               color_mode: str = 'pole') -> Dict[int, Tuple[str, str]]:
        """
        Create word clouds for all specified PCs.
        
        Parameters
        ----------
        pc_indices : List[int], optional
            List of PC indices. If None, uses top 9 PCs by 'triple' ranking
        k : int
            Number of topics to include from each pole
        max_words : int
            Maximum words to display in word cloud
        figsize : Tuple[int, int]
            Figure size for each subplot
        output_dir : str
            Directory to save output files
        display : bool
            Whether to display each plot
        color_mode : str
            'pole' or 'validity' coloring mode
            
        Returns
        -------
        Dict[int, Tuple[str, str]]
            Dictionary mapping PC index to (high_path, low_path)
        """
        if not hasattr(self, 'topic_analyzer') or self.topic_analyzer is None:
            raise ValueError("Topic analysis not performed. Initialize with texts to enable topic analysis.")
            
        if not hasattr(self, 'wordcloud_generator'):
            self.wordcloud_generator = PCWordCloudGenerator(
                self.topic_analyzer, 
                self.analysis.results,
                criterion=self.criterion,
                human_judgment=self.human_judgment,
                ai_judgment=self.ai_judgment
            )
            
        # If no indices specified, use top 9 PCs
        if pc_indices is None:
            pc_indices = self.analysis.get_top_pcs(n=9, ranking_method='triple')
            
        return self.wordcloud_generator.create_all_pc_wordclouds(
            pc_indices=pc_indices,
            k=k,
            max_words=max_words,
            figsize=figsize,
            output_dir=output_dir,
            display=display,
            color_mode=color_mode
        )
    
    def create_top_pcs_wordcloud_grid(self,
                                    pc_indices: Optional[List[int]] = None,
                                    ranking_method: str = 'triple',
                                    n_pcs: int = 9,
                                    k: int = 10,
                                    max_words: int = 50,
                                    output_file: Optional[str] = None,
                                    display: bool = True,
                                    color_mode: str = 'pole') -> Any:
        """
        Create a grid visualization of word clouds for top PCs.
        
        Parameters
        ----------
        pc_indices : List[int], optional
            List of PC indices. If None, uses top PCs by ranking_method
        ranking_method : str
            Method to rank PCs if pc_indices not provided: 'triple', 'HU', 'AI', 'X'
        n_pcs : int
            Number of top PCs to include if pc_indices not provided
        k : int
            Number of topics to include from each pole
        max_words : int
            Maximum words per word cloud
        output_file : str, optional
            Path to save the grid visualization
        display : bool
            Whether to display the plot
        color_mode : str
            'pole' (default): Red for high pole, blue for low pole
            'validity': Color based on X/HU/AI agreement
            
        Returns
        -------
        Figure
            Figure object
        """
        if not hasattr(self, 'topic_analyzer') or self.topic_analyzer is None:
            raise ValueError("Topic analysis not performed. Initialize with texts to enable topic analysis.")
            
        if not hasattr(self, 'wordcloud_generator'):
            self.wordcloud_generator = PCWordCloudGenerator(
                self.topic_analyzer, 
                self.analysis.results,
                criterion=self.criterion,
                human_judgment=self.human_judgment,
                ai_judgment=self.ai_judgment
            )
            
        # If no indices specified, use top PCs
        if pc_indices is None:
            pc_indices = self.analysis.get_top_pcs(n=n_pcs, ranking_method=ranking_method)
            
        return self.wordcloud_generator.create_top_pcs_wordcloud_grid(
            pc_indices=pc_indices,
            k=k,
            max_words=max_words,
            output_file=output_file,
            display=display,
            color_mode=color_mode
        )
    
    def create_comprehensive_pc_analysis(self,
                                       pc_indices: Optional[List[int]] = None,
                                       n_pcs: int = 15,
                                       k_topics: int = 3,
                                       max_words: int = 100,
                                       generate_wordclouds: bool = True,
                                       generate_3d_umap: bool = True,
                                       umap_arrow_k: int = 1,
                                       show_data_counts: bool = True,
                                       output_dir: Optional[str] = None,
                                       display: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive PC analysis with word cloud table and 3D UMAP visualization.
        
        This method generates a complete analysis similar to the Colab scripts, including:
        - Individual word clouds for each PC's high and low poles
        - A comprehensive table showing all PCs with X/HU/AI quartile labels
        - 3D UMAP visualization with PC arrows (optional)
        - Summary report with data availability statistics
        
        Parameters
        ----------
        pc_indices : List[int], optional
            List of PC indices (0-based) to analyze. If None, uses first n_pcs PCs.
            Example: [2, 1, 3, 4, 5] for PC3, PC2, PC4, PC5, PC6
        n_pcs : int, default=15
            Number of PCs to analyze if pc_indices not provided
        k_topics : int, default=3
            Number of topics to include from each pole in word clouds
        max_words : int, default=100
            Maximum words to display in each word cloud
        generate_wordclouds : bool, default=True
            Whether to generate word cloud table
        generate_3d_umap : bool, default=True
            Whether to generate 3D UMAP visualization with PC arrows
        umap_arrow_k : int, default=1
            Number of topics for arrow endpoints in UMAP (1 = single topic endpoints)
        show_data_counts : bool, default=True
            Whether to show data availability counts (e.g., "HU: n=3" for sparse data)
        output_dir : str, optional
            Directory to save all outputs. If None, creates 'haam_comprehensive_analysis'
        display : bool, default=True
            Whether to display visualizations
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'wordcloud_paths': Dict mapping PC index to (high_path, low_path)
            - 'table_path': Path to comprehensive PC table image
            - 'umap_path': Path to 3D UMAP HTML (if generated)
            - 'report_path': Path to text report with statistics
            - 'summary': Dict with analysis summary statistics
            
        Examples
        --------
        # Analyze first 15 PCs with all defaults
        results = haam.create_comprehensive_pc_analysis()
        
        # Analyze specific PCs
        specific_pcs = [2, 1, 3, 4, 5, 14, 13, 11, 12, 46, 9, 17, 16, 20, 105]
        results = haam.create_comprehensive_pc_analysis(pc_indices=specific_pcs)
        
        # Only generate word clouds, skip 3D UMAP
        results = haam.create_comprehensive_pc_analysis(
            n_pcs=10,
            generate_3d_umap=False
        )
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch
        
        if not hasattr(self, 'topic_analyzer') or self.topic_analyzer is None:
            raise ValueError("Topic analysis not performed. Initialize with texts to enable topic analysis.")
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'haam_comprehensive_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which PCs to analyze
        if pc_indices is None:
            pc_indices = list(range(min(n_pcs, self.n_components)))
        
        print("="*80)
        print("COMPREHENSIVE PC ANALYSIS WITH VALIDITY COLORING")
        print("="*80)
        print(f"\nAnalyzing {len(pc_indices)} PCs: {[pc+1 for pc in pc_indices]}")
        print(f"Output directory: {output_dir}")
        
        results = {
            'wordcloud_paths': {},
            'table_path': None,
            'umap_path': None,
            'report_path': None,
            'summary': {}
        }
        
        # Initialize word cloud generator if needed
        if not hasattr(self, 'wordcloud_generator'):
            self.wordcloud_generator = PCWordCloudGenerator(
                self.topic_analyzer,
                self.analysis.results,
                criterion=self.criterion,
                human_judgment=self.human_judgment,
                ai_judgment=self.ai_judgment
            )
        
        # ==============================================================================
        # STEP 1: GENERATE INDIVIDUAL WORD CLOUDS
        # ==============================================================================
        
        if generate_wordclouds:
            print("\n1. GENERATING INDIVIDUAL WORD CLOUDS...")
            print("-"*60)
            
            wordcloud_dir = os.path.join(output_dir, 'wordclouds')
            os.makedirs(wordcloud_dir, exist_ok=True)
            
            for i, pc_idx in enumerate(pc_indices):
                print(f"\n[{i+1}/{len(pc_indices)}] PC{pc_idx + 1}:")
                try:
                    fig, high_path, low_path = self.create_pc_wordclouds(
                        pc_idx=pc_idx,
                        k=k_topics,
                        max_words=max_words,
                        figsize=(16, 8),
                        output_dir=wordcloud_dir,
                        display=False,  # Don't display individual clouds
                        color_mode='validity'  # Use validity coloring
                    )
                    results['wordcloud_paths'][pc_idx] = (high_path, low_path)
                    print(f"  ✓ Generated word clouds")
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
            
            # ==============================================================================
            # STEP 2: CREATE COMPREHENSIVE TABLE
            # ==============================================================================
            
            print("\n2. CREATING COMPREHENSIVE PC TABLE...")
            print("-"*60)
            
            # Create a large figure for the table
            fig = plt.figure(figsize=(24, 4 * len(pc_indices)))
            
            # Create grid
            gs = gridspec.GridSpec(len(pc_indices), 3, width_ratios=[1, 2, 2], 
                                  wspace=0.2, hspace=0.3)
            
            for i, pc_idx in enumerate(pc_indices):
                print(f"  Processing PC{pc_idx + 1}...")
                
                # Get topics
                pc_topics = self.topic_analyzer.get_pc_high_low_topics(
                    pc_idx=pc_idx, n_high=k_topics, n_low=k_topics, p_threshold=0.05
                )
                
                # Column 1: PC Label
                ax_label = plt.subplot(gs[i, 0])
                ax_label.axis('off')
                
                # Get PC associations
                pc_assoc = self._get_pc_associations(pc_idx)
                
                label_text = f'PC{pc_idx + 1}\n\n'
                label_text += f"PC coefficients:\n"
                label_text += f"X: {pc_assoc.get('X', '?')} | "
                label_text += f"HU: {pc_assoc.get('HU', '?')} | "
                label_text += f"AI: {pc_assoc.get('AI', '?')}"
                
                ax_label.text(0.5, 0.5, label_text, ha='center', va='center',
                             fontsize=14, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
                
                # Column 2: High Pole
                self._add_wordcloud_to_table(gs[i, 1], pc_idx, pc_topics.get('high', []),
                                           'high', k_topics, max_words, show_data_counts)
                
                # Column 3: Low Pole  
                self._add_wordcloud_to_table(gs[i, 2], pc_idx, pc_topics.get('low', []),
                                           'low', k_topics, max_words, show_data_counts)
            
            # Add main title and legend
            plt.suptitle(f'Principal Components Analysis - Validity Colored Word Clouds',
                        fontsize=24, fontweight='bold', y=0.995)
            
            # Add color legend
            legend_elements = [
                Patch(facecolor='#8B0000', label='Consensus high'),
                Patch(facecolor='#FF6B6B', label='Any high'),
                Patch(facecolor='#00008B', label='Consensus low'),
                Patch(facecolor='#6B9AFF', label='Any low'),
                Patch(facecolor='#4A4A4A', label='Opposing'),
                Patch(facecolor='#B0B0B0', label='All middle')
            ]
            plt.figlegend(handles=legend_elements, loc='lower center',
                         bbox_to_anchor=(0.5, -0.005), ncol=6, fontsize=14)
            
            # Save table
            table_path = os.path.join(output_dir, 'pc_table_comprehensive.png')
            plt.savefig(table_path, dpi=200, bbox_inches='tight', facecolor='white')
            results['table_path'] = table_path
            print(f"✓ Saved comprehensive table: {table_path}")
            
            if display:
                plt.show()
            else:
                plt.close()
        
        # ==============================================================================
        # STEP 3: CREATE 3D UMAP VISUALIZATION
        # ==============================================================================
        
        if generate_3d_umap:
            print("\n3. CREATING 3D UMAP VISUALIZATION...")
            print("-"*60)
            
            try:
                umap_path = self.create_3d_umap_with_pc_arrows(
                    pc_indices=pc_indices,
                    arrow_mode='list',
                    top_k=umap_arrow_k,
                    percentile_threshold=90.0,
                    color_by_usage=True,
                    color_mode='validity',
                    show_topic_labels=False,
                    output_dir=output_dir,
                    display=display
                )
                results['umap_path'] = umap_path
                print(f"✓ Created 3D UMAP visualization: {umap_path}")
            except Exception as e:
                print(f"✗ Error creating 3D UMAP: {str(e)}")
        
        # ==============================================================================
        # STEP 4: CREATE SUMMARY REPORT
        # ==============================================================================
        
        print("\n4. CREATING SUMMARY REPORT...")
        print("-"*60)
        
        report_lines = ["PC Validity Analysis Report"]
        report_lines.append("="*120)
        report_lines.append("PC\tPC Coefs\tHigh Pole Quartiles\tHU Data\tLow Pole Quartiles\tHU Data\tSamples")
        report_lines.append("-"*120)
        
        total_high_samples = 0
        total_low_samples = 0
        
        for pc_idx in pc_indices:
            try:
                # Get PC associations
                pc_assoc = self._get_pc_associations(pc_idx)
                pc_pattern = f"X:{pc_assoc.get('X', '?')} HU:{pc_assoc.get('HU', '?')} AI:{pc_assoc.get('AI', '?')}"
                
                # Get topics
                pc_topics = self.topic_analyzer.get_pc_high_low_topics(
                    pc_idx=pc_idx, n_high=k_topics, n_low=k_topics, p_threshold=0.05
                )
                
                # Get quartiles for each pole
                high_topic_ids = [t['topic_id'] for t in pc_topics.get('high', [])]
                low_topic_ids = [t['topic_id'] for t in pc_topics.get('low', [])]
                
                high_quartiles = self._get_topic_quartile_positions(high_topic_ids)
                low_quartiles = self._get_topic_quartile_positions(low_topic_ids)
                
                high_pattern = f"X:{high_quartiles['X']} HU:{high_quartiles['HU']} AI:{high_quartiles['AI']}"
                low_pattern = f"X:{low_quartiles['X']} HU:{low_quartiles['HU']} AI:{low_quartiles['AI']}"
                
                # Get HU data counts
                high_hu_n = high_quartiles.get('_counts', {}).get('HU', 0)
                low_hu_n = low_quartiles.get('_counts', {}).get('HU', 0)
                
                # Get sample sizes
                high_samples = sum(t.get('size', 0) for t in pc_topics.get('high', []))
                low_samples = sum(t.get('size', 0) for t in pc_topics.get('low', []))
                
                total_high_samples += high_samples
                total_low_samples += low_samples
                
                report_lines.append(
                    f"PC{pc_idx+1}\t{pc_pattern}\t{high_pattern}\tn={high_hu_n}\t"
                    f"{low_pattern}\tn={low_hu_n}\t"
                    f"H:{high_samples:,} L:{low_samples:,}"
                )
            except:
                report_lines.append(f"PC{pc_idx+1}\tError\tError\t-\tError\t-\tError")
        
        # Add summary statistics
        report_lines.append("-"*120)
        report_lines.append(f"TOTAL SAMPLES: High poles: {total_high_samples:,}, Low poles: {total_low_samples:,}")
        
        # Calculate data availability
        x_avail = (~np.isnan(self.criterion)).sum()
        hu_avail = (~np.isnan(self.human_judgment)).sum()
        ai_avail = (~np.isnan(self.ai_judgment)).sum()
        total_n = len(self.criterion)
        
        report_lines.append(f"\nDATA AVAILABILITY:")
        report_lines.append(f"  X (criterion): {x_avail:,}/{total_n:,} ({x_avail/total_n*100:.1f}%)")
        report_lines.append(f"  HU (human): {hu_avail:,}/{total_n:,} ({hu_avail/total_n*100:.1f}%)")
        report_lines.append(f"  AI: {ai_avail:,}/{total_n:,} ({ai_avail/total_n*100:.1f}%)")
        
        # Save report
        report_path = os.path.join(output_dir, 'validity_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        results['report_path'] = report_path
        
        # Display summary
        print("\nValidity Analysis Summary:")
        print("-"*120)
        for line in report_lines[:min(20, len(report_lines))]:
            print(line)
        if len(report_lines) > 20:
            print("...")
        print(f"\n✓ Full report saved to: {report_path}")
        
        # Store summary statistics
        results['summary'] = {
            'n_pcs_analyzed': len(pc_indices),
            'total_high_samples': total_high_samples,
            'total_low_samples': total_low_samples,
            'data_availability': {
                'X': f"{x_avail/total_n*100:.1f}%",
                'HU': f"{hu_avail/total_n*100:.1f}%", 
                'AI': f"{ai_avail/total_n*100:.1f}%"
            }
        }
        
        # ==============================================================================
        # FINAL SUMMARY
        # ==============================================================================
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll outputs saved to: {output_dir}/")
        print("\n✅ Generated:")
        if generate_wordclouds:
            print(f"  - {len(results['wordcloud_paths'])} PC word cloud pairs")
            print(f"  - 1 comprehensive PC table")
        if generate_3d_umap:
            print(f"  - 1 3D UMAP visualization with PC arrows")
        print(f"  - 1 summary report")
        
        return results
    
    def _get_pc_associations(self, pc_idx: int) -> Dict[str, str]:
        """Get X/HU/AI associations for a PC based on coefficients."""
        associations = {}
        try:
            for outcome in ['X', 'AI', 'HU']:
                if outcome in self.analysis.results['debiased_lasso']:
                    coef = self.analysis.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                    associations[outcome] = 'H' if coef > 0 else 'L'
        except:
            pass
        return associations
    
    def _get_topic_quartile_positions(self, topic_ids: List[int]) -> Dict[str, Any]:
        """
        Calculate actual quartile positions for topics using available data.
        Handles sparse data by using nanmean and showing actual values when available.
        """
        if not topic_ids:
            return {'X': '?', 'HU': '?', 'AI': '?'}
        
        # Get the cluster labels
        cluster_labels = self.topic_analyzer.cluster_labels
        
        # Calculate means for the specified topics using available data
        topic_means = {'X': [], 'HU': [], 'AI': []}
        topic_counts = {'X': 0, 'HU': 0, 'AI': 0}
        
        for topic_id in topic_ids:
            # Get documents in this topic
            topic_mask = cluster_labels == topic_id
            
            if np.any(topic_mask):
                # For X (criterion)
                x_values = self.criterion[topic_mask]
                x_valid = x_values[~np.isnan(x_values)]
                if len(x_valid) > 0:
                    topic_means['X'].append(np.mean(x_valid))
                    topic_counts['X'] += len(x_valid)
                
                # For HU (human judgment)
                hu_values = self.human_judgment[topic_mask]
                hu_valid = hu_values[~np.isnan(hu_values)]
                if len(hu_valid) > 0:
                    topic_means['HU'].append(np.mean(hu_valid))
                    topic_counts['HU'] += len(hu_valid)
                
                # For AI
                ai_values = self.ai_judgment[topic_mask]
                ai_valid = ai_values[~np.isnan(ai_values)]
                if len(ai_valid) > 0:
                    topic_means['AI'].append(np.mean(ai_valid))
                    topic_counts['AI'] += len(ai_valid)
        
        # Calculate average across topics (only for topics with data)
        avg_means = {}
        for measure in ['X', 'HU', 'AI']:
            if topic_means[measure]:
                avg_means[measure] = np.mean(topic_means[measure])
            else:
                avg_means[measure] = np.nan
        
        # Calculate global quartiles using only non-NaN values
        quartiles = {}
        for measure, values in [('X', self.criterion),
                               ('HU', self.human_judgment),
                               ('AI', self.ai_judgment)]:
            # Get non-NaN values for quartile calculation
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                q25 = np.percentile(valid_values, 25)
                q75 = np.percentile(valid_values, 75)
                
                if np.isnan(avg_means[measure]):
                    # No data for this measure in these topics
                    quartiles[measure] = '?'
                elif avg_means[measure] >= q75:
                    quartiles[measure] = 'H'
                elif avg_means[measure] <= q25:
                    quartiles[measure] = 'L'
                else:
                    quartiles[measure] = 'M'
            else:
                # No valid data for this measure at all
                quartiles[measure] = '?'
        
        # Add data counts for transparency
        quartiles['_counts'] = topic_counts
        
        return quartiles
    
    def _add_wordcloud_to_table(self, ax, pc_idx: int, topics: List[Dict], 
                               pole: str, k: int, max_words: int, 
                               show_data_counts: bool):
        """Add word cloud to table subplot."""
        ax_subplot = plt.subplot(ax)
        n_topics = len(topics)
        samples = sum(t.get('size', 0) for t in topics)
        
        try:
            if n_topics > 0:
                # Generate word cloud with validity coloring
                temp_fig, _, _ = self.create_pc_wordclouds(
                    pc_idx=pc_idx, k=k, max_words=max_words,
                    output_dir=None, display=False, color_mode='validity'
                )
                
                # Extract the appropriate pole image
                temp_axes = temp_fig.get_axes()
                pole_idx = 0 if pole == 'high' else 1
                if temp_axes and len(temp_axes) > pole_idx:
                    for child in temp_axes[pole_idx].get_children():
                        if hasattr(child, 'get_array') and child.get_array() is not None:
                            ax_subplot.imshow(child.get_array())
                            break
                plt.close(temp_fig)
                
                # Get quartiles
                topic_ids = [t['topic_id'] for t in topics]
                quartiles = self._get_topic_quartile_positions(topic_ids)
                
                title = f'{pole.capitalize()} ({n_topics} topics, n={samples:,})\n'
                title += f"X:{quartiles['X']} HU:{quartiles['HU']} AI:{quartiles['AI']}"
                
                # Add HU count if sparse and requested
                if show_data_counts and '_counts' in quartiles and quartiles['_counts']['HU'] < 10:
                    title += f" (HU:n={quartiles['_counts']['HU']})"
                
                color = 'darkred' if pole == 'high' else 'darkblue'
                ax_subplot.set_title(title, fontsize=12, color=color)
            else:
                ax_subplot.text(0.5, 0.5, f'No {pole} topics', ha='center', va='center')
                color = 'darkred' if pole == 'high' else 'darkblue'
                ax_subplot.set_title(f'{pole.capitalize()} (0 topics)', fontsize=12, color=color)
        except:
            ax_subplot.text(0.5, 0.5, 'Error', ha='center', va='center')
            color = 'darkred' if pole == 'high' else 'darkblue'
            ax_subplot.set_title(f'{pole.capitalize()} (error)', fontsize=12, color=color)
        
        ax_subplot.axis('off')
    
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
        for color_by in ['X', 'AI', 'HU']:
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
    ai_judgment = criterion + np.random.normal(0, 0.5, n_samples)  # AI correlates with X
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