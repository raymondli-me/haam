"""
HAAM Word Cloud Module
=====================

Generate word clouds for PC poles (high and low) with customized coloring.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from typing import Dict, List, Optional, Tuple, Union
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class PCWordCloudGenerator:
    """
    Generate word clouds for principal component poles.
    """
    
    def __init__(self, topic_analyzer, analysis_results=None):
        """
        Initialize word cloud generator.
        
        Parameters
        ----------
        topic_analyzer : TopicAnalyzer
            TopicAnalyzer instance with computed topics and PC associations
        analysis_results : dict, optional
            HAAM analysis results containing model coefficients for validity coloring
        """
        self.topic_analyzer = topic_analyzer
        self.analysis_results = analysis_results
        
    def create_pc_wordclouds(self, 
                           pc_idx: int,
                           k: int = 10,
                           max_words: int = 100,
                           figsize: Tuple[int, int] = (10, 5),
                           output_dir: Optional[str] = None,
                           display: bool = True,
                           color_mode: str = 'pole') -> Tuple[plt.Figure, str, str]:
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
            'validity': Color based on Y/HU/AI agreement:
                - Dark red: top quartile for all (Y, HU, AI)
                - Light red: top quartile for HU & AI only
                - Dark blue: bottom quartile for all
                - Light blue: bottom quartile for HU & AI only
                - Grey: mixed signals
            
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
            if color_mode == 'validity':
                # Get validity-based colors
                word_colors = self._calculate_topic_validity_colors(pc_topics['high'], pc_idx)
                
                # Create custom color function
                def color_func_high(word, **kwargs):
                    return word_colors.get(word, '#B0B0B0')  # Default to light grey
                
                wc_high = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=max_words,
                    relative_scaling=0.5,
                    min_font_size=10,
                    color_func=color_func_high
                ).generate_from_frequencies(high_word_freq)
            else:
                # Original pole-based coloring
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
                ).generate_from_frequencies(high_word_freq)
            
            ax1.imshow(wc_high, interpolation='bilinear')
            title_suffix = " (Validity Colors)" if color_mode == 'validity' else ""
            ax1.set_title(f'PC{pc_idx + 1} - High Pole Topics{title_suffix}', fontsize=16, fontweight='bold', color='darkred')
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
            if color_mode == 'validity':
                # Get validity-based colors
                word_colors = self._calculate_topic_validity_colors(pc_topics['low'], pc_idx)
                
                # Create custom color function
                def color_func_low(word, **kwargs):
                    return word_colors.get(word, '#B0B0B0')  # Default to light grey
                
                wc_low = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=max_words,
                    relative_scaling=0.5,
                    min_font_size=10,
                    color_func=color_func_low
                ).generate_from_frequencies(low_word_freq)
            else:
                # Original pole-based coloring
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
                ).generate_from_frequencies(low_word_freq)
            
            ax2.imshow(wc_low, interpolation='bilinear')
            title_suffix = " (Validity Colors)" if color_mode == 'validity' else ""
            ax2.set_title(f'PC{pc_idx + 1} - Low Pole Topics{title_suffix}', fontsize=16, fontweight='bold', color='darkblue')
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'No significant low topics', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title(f'PC{pc_idx + 1} - Low Pole Topics', fontsize=16, fontweight='bold', color='darkblue')
            ax2.axis('off')
        
        plt.tight_layout()
        
        # Add legend for validity mode
        if color_mode == 'validity':
            # Create a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#8B0000', label='Valid high SC (Y+HU+AI top quartile)'),
                Patch(facecolor='#FF6B6B', label='Perceived high SC (HU+AI only)'),
                Patch(facecolor='#00008B', label='Valid low SC (Y+HU+AI bottom quartile)'),
                Patch(facecolor='#6B9AFF', label='Perceived low SC (HU+AI only)'),
                Patch(facecolor='#4A4A4A', label='Mixed strong signal'),
                Patch(facecolor='#B0B0B0', label='Mixed weak signal')
            ]
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), 
                      ncol=3, frameon=False, fontsize=10)
            plt.subplots_adjust(bottom=0.15)
        
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
    
    def _calculate_topic_validity_colors(self, topics: List[Dict], pc_idx: int) -> Dict[str, str]:
        """
        Calculate color for each word based on topic's validity across Y/HU/AI.
        
        This method determines colors based on whether topics are associated with
        valid vs perceived social class markers. It looks at topic averages across
        different outcomes to identify:
        - Valid markers: High/low in all three (Y, HU, AI)
        - Perceived markers: High/low in HU & AI but not Y
        - Mixed signals: Inconsistent patterns
        
        Parameters
        ----------
        topics : List[Dict]
            List of topic dictionaries from get_pc_high_low_topics
        pc_idx : int
            PC index to analyze
            
        Returns
        -------
        Dict[str, str]
            Mapping of word to color hex code
        """
        if not hasattr(self.topic_analyzer, 'cluster_labels'):
            return {}
            
        # Color definitions
        colors = {
            'dark_red': '#8B0000',     # All top quartile
            'light_red': '#FF6B6B',    # HU & AI top only
            'dark_blue': '#00008B',    # All bottom quartile
            'light_blue': '#6B9AFF',   # HU & AI bottom only
            'dark_grey': '#4A4A4A',    # Mixed strong
            'light_grey': '#B0B0B0'    # Mixed weak
        }
        
        # Get all topics for this PC to establish quartiles
        all_topics_result = self.topic_analyzer.get_pc_high_low_topics(
            pc_idx=pc_idx,
            n_high=999,  # Get all topics
            n_low=999,
            p_threshold=1.0  # Include all topics for quartile calculation
        )
        all_topics = all_topics_result.get('all', [])
        
        if not all_topics:
            return {}
            
        # Calculate quartiles based on avg_percentile
        # Topics with high avg_percentile are associated with high PC values
        # Topics with low avg_percentile are associated with low PC values
        percentiles = [t['avg_percentile'] for t in all_topics]
        q75 = np.percentile(percentiles, 75)
        q25 = np.percentile(percentiles, 25)
        
        # For validity checking, we need to know if this PC is associated with
        # high or low values in each outcome (Y/SC, HU, AI)
        # This requires checking the PC's relationship with each outcome
        pc_associations = {}
        
        if self.analysis_results and 'debiased_lasso' in self.analysis_results:
            # Check if this PC is positively or negatively associated with each outcome
            for outcome in ['SC', 'AI', 'HU']:
                if outcome in self.analysis_results['debiased_lasso']:
                    coef = self.analysis_results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                    # Positive coefficient means PC is associated with high outcome values
                    # Negative coefficient means PC is associated with low outcome values
                    pc_associations[outcome] = 'positive' if coef > 0 else 'negative'
        
        if not pc_associations:
            # Fallback to simple percentile-based coloring
            word_colors = {}
            for topic in topics:
                if topic['avg_percentile'] > q75:
                    color = colors['dark_red']
                elif topic['avg_percentile'] < q25:
                    color = colors['dark_blue']
                else:
                    color = colors['light_grey']
                    
                keywords = topic['keywords'].split(' | ')
                for keyword in keywords:
                    keyword = keyword.strip()
                    if keyword and len(keyword) > 1:
                        word_colors[keyword] = color
            return word_colors
        
        word_colors = {}
        
        for topic in topics:
            percentile = topic['avg_percentile']
            
            # Determine if topic is in high or low quartile
            is_high_topic = percentile > q75
            is_low_topic = percentile < q25
            
            if is_high_topic:
                # This topic is associated with high PC values
                # Check if all outcomes view high PC as high class
                all_positive = all(pc_associations.get(outcome) == 'positive' 
                                 for outcome in ['SC', 'AI', 'HU'])
                hu_ai_positive = (pc_associations.get('HU') == 'positive' and 
                                pc_associations.get('AI') == 'positive')
                
                if all_positive:
                    color = colors['dark_red']  # Valid high marker
                elif hu_ai_positive and pc_associations.get('SC') != 'positive':
                    color = colors['light_red']  # Perceived high marker
                else:
                    color = colors['dark_grey']  # Mixed signal
                    
            elif is_low_topic:
                # This topic is associated with low PC values
                # Check if all outcomes view low PC as low class
                all_negative = all(pc_associations.get(outcome) == 'negative' 
                                 for outcome in ['SC', 'AI', 'HU'])
                hu_ai_negative = (pc_associations.get('HU') == 'negative' and 
                                pc_associations.get('AI') == 'negative')
                
                if all_negative:
                    color = colors['dark_blue']  # Valid low marker
                elif hu_ai_negative and pc_associations.get('SC') != 'negative':
                    color = colors['light_blue']  # Perceived low marker
                else:
                    color = colors['dark_grey']  # Mixed signal
                    
            else:
                # Topic is in middle quartiles
                color = colors['light_grey']  # Weak signal
            
            # Apply color to all words in this topic
            keywords = topic['keywords'].split(' | ')
            for keyword in keywords:
                keyword = keyword.strip()
                if keyword and len(keyword) > 1:
                    word_colors[keyword] = color
                    
        return word_colors
    
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
        color_mode : str
            'pole' or 'validity' coloring mode
            
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
                    display=display,
                    color_mode=color_mode
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
                                    display: bool = True,
                                    color_mode: str = 'pole') -> plt.Figure:
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
        color_mode : str
            'pole' (default): Red for high pole, blue for low pole
            'validity': Color based on Y/HU/AI agreement
            
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
        
        # Create colormaps for pole mode
        if color_mode == 'pole':
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
                if color_mode == 'validity':
                    # Get validity-based colors
                    word_colors = self._calculate_topic_validity_colors(pc_topics['high'], pc_idx)
                    
                    # Create custom color function
                    def color_func_high(word, **kwargs):
                        return word_colors.get(word, '#B0B0B0')
                    
                    wc_high = WordCloud(
                        width=400,
                        height=200,
                        background_color='white',
                        max_words=max_words,
                        relative_scaling=0.5,
                        min_font_size=8,
                        color_func=color_func_high
                    ).generate_from_frequencies(high_word_freq)
                else:
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
                if color_mode == 'validity':
                    # Get validity-based colors
                    word_colors = self._calculate_topic_validity_colors(pc_topics['low'], pc_idx)
                    
                    # Create custom color function
                    def color_func_low(word, **kwargs):
                        return word_colors.get(word, '#B0B0B0')
                    
                    wc_low = WordCloud(
                        width=400,
                        height=200,
                        background_color='white',
                        max_words=max_words,
                        relative_scaling=0.5,
                        min_font_size=8,
                        color_func=color_func_low
                    ).generate_from_frequencies(low_word_freq)
                else:
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
        
        # Update title based on color mode
        if color_mode == 'validity':
            plt.suptitle('Principal Component Word Clouds - Validity-Based Coloring', 
                         fontsize=16, fontweight='bold')
        else:
            plt.suptitle('Principal Component Word Clouds - High (Red) vs Low (Blue) Poles', 
                         fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Add legend for validity mode
        if color_mode == 'validity':
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#8B0000', label='Valid high (Y+HU+AI)'),
                Patch(facecolor='#FF6B6B', label='Perceived high (HU+AI only)'),
                Patch(facecolor='#00008B', label='Valid low (Y+HU+AI)'),
                Patch(facecolor='#6B9AFF', label='Perceived low (HU+AI only)'),
                Patch(facecolor='#4A4A4A', label='Mixed strong'),
                Patch(facecolor='#B0B0B0', label='Mixed weak')
            ]
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.02), 
                      ncol=3, frameon=False, fontsize=9)
            plt.subplots_adjust(bottom=0.08)
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Grid visualization saved to: {output_file}")
            
        if display:
            plt.show()
        else:
            plt.close(fig)
            
        return fig