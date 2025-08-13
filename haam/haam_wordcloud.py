"""
HAAM Word Cloud Module
=====================

Generate word clouds for PC poles (high and low) with customized coloring.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Union
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class PCWordCloudGenerator:
    """
    Generate word clouds for principal component poles.
    """
    
    def __init__(self, topic_analyzer):
        """
        Initialize word cloud generator.
        
        Parameters
        ----------
        topic_analyzer : TopicAnalyzer
            TopicAnalyzer instance with computed topics and PC associations
        """
        self.topic_analyzer = topic_analyzer
        
    def create_pc_wordclouds(self, 
                           pc_idx: int,
                           k: int = 10,
                           max_words: int = 100,
                           figsize: Tuple[int, int] = (10, 5),
                           output_dir: Optional[str] = None,
                           display: bool = True) -> Tuple[plt.Figure, str, str]:
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
            
        Returns
        -------
        Tuple[plt.Figure, str, str]
            Figure object, high pole output path, low pole output path
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            raise ImportError("wordcloud package not installed. Install with: pip install wordcloud")
            
        # Get high and low topics for this PC
        pc_topics = self.topic_analyzer.get_pc_high_low_topics(
            pc_idx=pc_idx,
            n_high=k,
            n_low=k,
            p_threshold=0.05
        )
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
        
        # Process high pole topics
        high_word_freq = self._aggregate_topic_keywords(pc_topics['high'])
        if high_word_freq:
            # Create red colormap for high pole
            colors_red = ["#ffcccc", "#ff9999", "#ff6666", "#ff3333", "#cc0000", "#990000"]
            cmap_red = LinearSegmentedColormap.from_list("red_gradient", colors_red)
            
            wc_high = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=cmap_red,
                max_words=max_words,
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(high_word_freq)  # Use frequency dict directly
            
            ax1.imshow(wc_high, interpolation='bilinear')
            ax1.set_title(f'PC{pc_idx + 1} - High Pole Topics', fontsize=16, fontweight='bold', color='darkred')
            ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, 'No significant high topics', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title(f'PC{pc_idx + 1} - High Pole Topics', fontsize=16, fontweight='bold', color='darkred')
            ax1.axis('off')
        
        # Process low pole topics
        low_word_freq = self._aggregate_topic_keywords(pc_topics['low'])
        if low_word_freq:
            # Create blue colormap for low pole
            colors_blue = ["#ccccff", "#9999ff", "#6666ff", "#3333ff", "#0000cc", "#000099"]
            cmap_blue = LinearSegmentedColormap.from_list("blue_gradient", colors_blue)
            
            wc_low = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=cmap_blue,
                max_words=max_words,
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(low_word_freq)  # Use frequency dict directly
            
            ax2.imshow(wc_low, interpolation='bilinear')
            ax2.set_title(f'PC{pc_idx + 1} - Low Pole Topics', fontsize=16, fontweight='bold', color='darkblue')
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'No significant low topics', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title(f'PC{pc_idx + 1} - Low Pole Topics', fontsize=16, fontweight='bold', color='darkblue')
            ax2.axis('off')
        
        plt.tight_layout()
        
        # Save if output directory provided
        high_path = None
        low_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save combined figure
            combined_path = os.path.join(output_dir, f'pc{pc_idx + 1}_wordclouds.png')
            fig.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            # Also save individual word clouds
            if high_word_freq:
                fig_high, ax_high = plt.subplots(1, 1, figsize=figsize)
                ax_high.imshow(wc_high, interpolation='bilinear')
                ax_high.set_title(f'PC{pc_idx + 1} - High Pole Topics', fontsize=16, fontweight='bold', color='darkred')
                ax_high.axis('off')
                high_path = os.path.join(output_dir, f'pc{pc_idx + 1}_high_wordcloud.png')
                fig_high.savefig(high_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig_high)
                
            if low_word_freq:
                fig_low, ax_low = plt.subplots(1, 1, figsize=figsize)
                ax_low.imshow(wc_low, interpolation='bilinear')
                ax_low.set_title(f'PC{pc_idx + 1} - Low Pole Topics', fontsize=16, fontweight='bold', color='darkblue')
                ax_low.axis('off')
                low_path = os.path.join(output_dir, f'pc{pc_idx + 1}_low_wordcloud.png')
                fig_low.savefig(low_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig_low)
        
        if display:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, high_path, low_path
    
    def _aggregate_topic_keywords(self, topics: List[Dict]) -> str:
        """
        Aggregate keywords from multiple topics with frequency weighting.
        
        Parameters
        ----------
        topics : List[Dict]
            List of topic dictionaries with 'keywords' field
            
        Returns
        -------
        str
            Aggregated text with frequency dictionary for word cloud
        """
        if not topics:
            return ""
            
        word_freq = Counter()
        
        for i, topic in enumerate(topics):
            # Weight keywords by topic rank (higher rank = more frequent)
            weight = len(topics) - i
            keywords = topic['keywords'].split(' | ')
            
            for keyword in keywords:
                # Clean keyword
                keyword = keyword.strip()
                if keyword and len(keyword) > 1:  # Skip single characters
                    word_freq[keyword] += weight
        
        # Return the word frequency dictionary directly
        return word_freq
    
    def create_all_pc_wordclouds(self,
                               pc_indices: Optional[List[int]] = None,
                               k: int = 10,
                               max_words: int = 100,
                               figsize: Tuple[int, int] = (10, 5),
                               output_dir: str = './wordclouds',
                               display: bool = False) -> Dict[int, Tuple[str, str]]:
        """
        Create word clouds for all specified PCs.
        
        Parameters
        ----------
        pc_indices : List[int], optional
            List of PC indices. If None, uses all available PCs
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
            
        Returns
        -------
        Dict[int, Tuple[str, str]]
            Dictionary mapping PC index to (high_path, low_path)
        """
        if pc_indices is None:
            pc_indices = list(range(self.topic_analyzer.pca_features.shape[1]))
            
        os.makedirs(output_dir, exist_ok=True)
        output_paths = {}
        
        print(f"Generating word clouds for {len(pc_indices)} PCs...")
        
        for i, pc_idx in enumerate(pc_indices):
            print(f"  Processing PC{pc_idx + 1} ({i + 1}/{len(pc_indices)})...")
            
            try:
                _, high_path, low_path = self.create_pc_wordclouds(
                    pc_idx=pc_idx,
                    k=k,
                    max_words=max_words,
                    figsize=figsize,
                    output_dir=output_dir,
                    display=display
                )
                output_paths[pc_idx] = (high_path, low_path)
                
            except Exception as e:
                print(f"    Warning: Failed to create word cloud for PC{pc_idx + 1}: {e}")
                
        print(f"\nGenerated word clouds saved to: {output_dir}")
        return output_paths
    
    def create_top_pcs_wordcloud_grid(self,
                                    pc_indices: List[int],
                                    k: int = 10,
                                    max_words: int = 50,
                                    output_file: Optional[str] = None,
                                    display: bool = True) -> plt.Figure:
        """
        Create a grid visualization of word clouds for top PCs.
        
        Parameters
        ----------
        pc_indices : List[int]
            List of PC indices to visualize (max 9 for 3x3 grid)
        k : int
            Number of topics to include from each pole
        max_words : int
            Maximum words per word cloud
        output_file : str, optional
            Path to save the grid visualization
        display : bool
            Whether to display the plot
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            raise ImportError("wordcloud package not installed. Install with: pip install wordcloud")
            
        # Limit to 9 PCs for 3x3 grid
        pc_indices = pc_indices[:9]
        n_pcs = len(pc_indices)
        
        # Calculate grid dimensions
        n_cols = min(3, n_pcs)
        n_rows = (n_pcs + n_cols - 1) // n_cols
        
        # Create figure
        fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
        
        # Create colormaps
        colors_red = ["#ffcccc", "#ff9999", "#ff6666", "#ff3333", "#cc0000", "#990000"]
        cmap_red = LinearSegmentedColormap.from_list("red_gradient", colors_red)
        
        colors_blue = ["#ccccff", "#9999ff", "#6666ff", "#3333ff", "#0000cc", "#000099"]
        cmap_blue = LinearSegmentedColormap.from_list("blue_gradient", colors_blue)
        
        for i, pc_idx in enumerate(pc_indices):
            # Get topics
            pc_topics = self.topic_analyzer.get_pc_high_low_topics(
                pc_idx=pc_idx,
                n_high=k,
                n_low=k,
                p_threshold=0.05
            )
            
            # High pole subplot
            ax_high = plt.subplot(n_rows, n_cols * 2, i * 2 + 1)
            high_word_freq = self._aggregate_topic_keywords(pc_topics['high'])
            
            if high_word_freq:
                wc_high = WordCloud(
                    width=400,
                    height=200,
                    background_color='white',
                    colormap=cmap_red,
                    max_words=max_words,
                    relative_scaling=0.5,
                    min_font_size=8
                ).generate_from_frequencies(high_word_freq)
                ax_high.imshow(wc_high, interpolation='bilinear')
            else:
                ax_high.text(0.5, 0.5, 'No topics', ha='center', va='center', transform=ax_high.transAxes)
                
            ax_high.set_title(f'PC{pc_idx + 1} High', fontsize=12, color='darkred')
            ax_high.axis('off')
            
            # Low pole subplot
            ax_low = plt.subplot(n_rows, n_cols * 2, i * 2 + 2)
            low_word_freq = self._aggregate_topic_keywords(pc_topics['low'])
            
            if low_word_freq:
                wc_low = WordCloud(
                    width=400,
                    height=200,
                    background_color='white',
                    colormap=cmap_blue,
                    max_words=max_words,
                    relative_scaling=0.5,
                    min_font_size=8
                ).generate_from_frequencies(low_word_freq)
                ax_low.imshow(wc_low, interpolation='bilinear')
            else:
                ax_low.text(0.5, 0.5, 'No topics', ha='center', va='center', transform=ax_low.transAxes)
                
            ax_low.set_title(f'PC{pc_idx + 1} Low', fontsize=12, color='darkblue')
            ax_low.axis('off')
        
        plt.suptitle('Principal Component Word Clouds - High (Red) vs Low (Blue) Poles', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Grid visualization saved to: {output_file}")
            
        if display:
            plt.show()
        else:
            plt.close(fig)
            
        return fig