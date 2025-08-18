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
    
    def __init__(self, topic_analyzer, analysis_results=None, 
                 criterion=None, human_judgment=None, ai_judgment=None):
        """
        Initialize word cloud generator.
        
        Parameters
        ----------
        topic_analyzer : TopicAnalyzer
            TopicAnalyzer instance with computed topics and PC associations
        analysis_results : dict, optional
            HAAM analysis results containing model coefficients for validity coloring
        criterion : array-like, optional
            Ground truth values (X) for direct validity measurement
        human_judgment : array-like, optional
            Human judgment values (HU) for direct validity measurement
        ai_judgment : array-like, optional
            AI judgment values (AI) for direct validity measurement
        """
        self.topic_analyzer = topic_analyzer
        self.analysis_results = analysis_results
        self.criterion = criterion
        self.human_judgment = human_judgment
        self.ai_judgment = ai_judgment
        
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
            'validity': Color based on X/HU/AI agreement:
                - Dark red: consensus high (all in top quartile)
                - Light red: any high signal (at least one in top quartile)
                - Dark blue: consensus low (all in bottom quartile)
                - Light blue: any low signal (at least one in bottom quartile)
                - Dark grey: opposing signals (mix of high and low)
                - Light grey: all in middle quartiles
            
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
                Patch(facecolor='#8B0000', label='Consensus high (all top quartile)'),
                Patch(facecolor='#FF6B6B', label='Any high signal'),
                Patch(facecolor='#00008B', label='Consensus low (all bottom quartile)'),
                Patch(facecolor='#6B9AFF', label='Any low signal'),
                Patch(facecolor='#4A4A4A', label='Opposing signals (high & low)'),
                Patch(facecolor='#B0B0B0', label='All middle quartiles')
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
        Calculate color for each word based on topic's validity across X/HU/AI.
        
        This method uses direct measurement of topic X/HU/AI values to determine colors.
        It calculates the actual mean values for each topic and compares them to
        global quartiles to identify:
        - Valid markers: High/low in all three (X, HU, AI)
        - Perceived markers: High/low in HU & AI but not X
        - Mixed signals: Inconsistent patterns
        
        Parameters
        ----------
        topics : List[Dict]
            List of topic dictionaries from get_pc_high_low_topics
        pc_idx : int
            PC index to analyze (kept for compatibility)
            
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
            'light_red': '#FF6B6B',    # At least one in top quartile
            'dark_blue': '#00008B',    # All bottom quartile
            'light_blue': '#6B9AFF',   # At least one in bottom quartile
            'dark_grey': '#4A4A4A',    # Mixed (some high, some low)
            'light_grey': '#B0B0B0'    # All middle quartiles
        }
        
        # Check if we have X/HU/AI data for direct measurement
        if self.criterion is None or self.human_judgment is None or self.ai_judgment is None:
            # Fall back to old method if data not available
            print(f"[DEBUG] Falling back to legacy color method. criterion={self.criterion is not None}, "
                  f"human_judgment={self.human_judgment is not None}, ai_judgment={self.ai_judgment is not None}")
            return self._calculate_topic_validity_colors_legacy(topics, pc_idx)
        
        # First, calculate means for ALL topics to establish topic-level quartiles
        # This ensures topics can actually reach the extreme quartiles
        all_cluster_ids = np.unique(self.topic_analyzer.cluster_labels[self.topic_analyzer.cluster_labels != -1])
        all_topic_means = {'X': [], 'HU': [], 'AI': []}
        
        for cluster_id in all_cluster_ids:
            topic_mask = self.topic_analyzer.cluster_labels == cluster_id
            if np.any(topic_mask):
                # Y (criterion)
                y_values = self.criterion[topic_mask]
                y_valid = y_values[~np.isnan(y_values)]
                if len(y_valid) > 0:
                    all_topic_means['X'].append(np.mean(y_valid))
                
                # HU (human judgment) - handle sparse data
                hu_values = self.human_judgment[topic_mask]
                hu_valid = hu_values[~np.isnan(hu_values)]
                if len(hu_valid) > 0:
                    all_topic_means['HU'].append(np.mean(hu_valid))
                
                # AI (ai judgment)
                ai_values = self.ai_judgment[topic_mask]
                ai_valid = ai_values[~np.isnan(ai_values)]
                if len(ai_valid) > 0:
                    all_topic_means['AI'].append(np.mean(ai_valid))
        
        # Calculate quartiles based on TOPIC MEANS, not document values
        y_q25 = np.percentile(all_topic_means['X'], 25) if all_topic_means['X'] else np.nan
        y_q75 = np.percentile(all_topic_means['X'], 75) if all_topic_means['X'] else np.nan
        hu_q25 = np.percentile(all_topic_means['HU'], 25) if all_topic_means['HU'] else np.nan
        hu_q75 = np.percentile(all_topic_means['HU'], 75) if all_topic_means['HU'] else np.nan
        ai_q25 = np.percentile(all_topic_means['AI'], 25) if all_topic_means['AI'] else np.nan
        ai_q75 = np.percentile(all_topic_means['AI'], 75) if all_topic_means['AI'] else np.nan
        
        word_colors = {}
        cluster_labels = self.topic_analyzer.cluster_labels
        
        for topic in topics:
            topic_id = topic['topic_id']
            
            # Get documents in this topic
            topic_mask = cluster_labels == topic_id
            
            if not np.any(topic_mask):
                # No documents in topic, use default color
                color = colors['light_grey']
            else:
                # Calculate mean X/HU/AI values for this topic (handling NaN)
                y_values = self.criterion[topic_mask]
                y_valid = y_values[~np.isnan(y_values)]
                y_mean = np.mean(y_valid) if len(y_valid) > 0 else np.nan
                
                hu_values = self.human_judgment[topic_mask]
                hu_valid = hu_values[~np.isnan(hu_values)]
                hu_mean = np.mean(hu_valid) if len(hu_valid) > 0 else np.nan
                
                ai_values = self.ai_judgment[topic_mask]
                ai_valid = ai_values[~np.isnan(ai_values)]
                ai_mean = np.mean(ai_valid) if len(ai_valid) > 0 else np.nan
                
                # Count valid measures
                valid_count = sum([not np.isnan(x) for x in [y_mean, hu_mean, ai_mean]])
                
                # Determine quartile positions (only for non-NaN values)
                n_high = 0
                n_low = 0
                
                if not np.isnan(y_mean) and not np.isnan(y_q25) and not np.isnan(y_q75):
                    if y_mean >= y_q75:
                        n_high += 1
                    elif y_mean <= y_q25:
                        n_low += 1
                
                if not np.isnan(hu_mean) and not np.isnan(hu_q25) and not np.isnan(hu_q75):
                    if hu_mean >= hu_q75:
                        n_high += 1
                    elif hu_mean <= hu_q25:
                        n_low += 1
                
                if not np.isnan(ai_mean) and not np.isnan(ai_q25) and not np.isnan(ai_q75):
                    if ai_mean >= ai_q75:
                        n_high += 1
                    elif ai_mean <= ai_q25:
                        n_low += 1
                
                # Debug output for extreme cases
                if n_high == 3 or n_low == 3:
                    print(f"[DEBUG] Topic {topic_id}: n_high={n_high}, n_low={n_low}, "
                          f"X={y_mean:.2f} (q25={y_q25:.2f}, q75={y_q75:.2f}), "
                          f"HU={hu_mean:.2f} (q25={hu_q25:.2f}, q75={hu_q75:.2f}), "
                          f"AI={ai_mean:.2f} (q25={ai_q25:.2f}, q75={ai_q75:.2f})")
                
                # Assign color based on pattern
                if valid_count == 0:
                    # No valid data
                    color = colors['light_grey']
                elif n_high > 0 and n_low > 0:
                    # Mixed signals (some high, some low)
                    color = colors['dark_grey']
                elif valid_count < 3:
                    # With sparse data, if all available measures agree, use dark color
                    if n_high == valid_count and n_high > 0:
                        color = colors['dark_red']  # All available are high
                    elif n_low == valid_count and n_low > 0:
                        color = colors['dark_blue']  # All available are low
                    elif n_high > 0:
                        color = colors['light_red']  # Some high
                    elif n_low > 0:
                        color = colors['light_blue']  # Some low
                    else:
                        color = colors['light_grey']  # All middle
                else:
                    # All three measures available
                    if n_high == 3:
                        color = colors['dark_red']  # Consensus high
                    elif n_high > 0:
                        color = colors['light_red']  # Some high
                    elif n_low == 3:
                        color = colors['dark_blue']  # Consensus low
                    elif n_low > 0:
                        color = colors['light_blue']  # Some low
                    else:
                        color = colors['light_grey']  # All middle
            
            # Apply color to all words in this topic
            keywords = topic['keywords'].split(' | ')
            for keyword in keywords:
                keyword = keyword.strip()
                if keyword and len(keyword) > 1:
                    word_colors[keyword] = color
        
        # Debug: Print color distribution
        color_counts = {}
        for color in word_colors.values():
            color_counts[color] = color_counts.get(color, 0) + 1
        
        print(f"\n[DEBUG] Color distribution for PC{pc_idx+1}:")
        for color_hex, count in sorted(color_counts.items()):
            color_name = [k for k, v in colors.items() if v == color_hex][0]
            print(f"  {color_name} ({color_hex}): {count} words")
                    
        return word_colors
    
    def _calculate_topic_validity_colors_legacy(self, topics: List[Dict], pc_idx: int) -> Dict[str, str]:
        """
        Legacy method for backward compatibility when X/HU/AI data not available.
        Uses PC-based inference as in the original implementation.
        """
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
        
        if not all_topics or not topics:
            # Return default grey for all words if no topics
            word_colors = {}
            for topic in topics:
                keywords = topic['keywords'].split(' | ')
                for keyword in keywords:
                    keyword = keyword.strip()
                    if keyword and len(keyword) > 1:
                        word_colors[keyword] = colors['light_grey']
            return word_colors
            
        # Calculate quartiles based on avg_percentile
        percentiles = [t['avg_percentile'] for t in all_topics]
        q75 = np.percentile(percentiles, 75)
        q25 = np.percentile(percentiles, 25)
        
        # For validity checking, check PC's relationship with outcomes
        pc_associations = {}
        
        if self.analysis_results and 'debiased_lasso' in self.analysis_results:
            for outcome in ['X', 'AI', 'HU']:
                if outcome in self.analysis_results['debiased_lasso']:
                    coef = self.analysis_results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
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
            
            # Check for opposing signals across outcomes
            has_positive = any(pc_associations.get(outcome) == 'positive' 
                             for outcome in ['X', 'AI', 'HU'])
            has_negative = any(pc_associations.get(outcome) == 'negative' 
                             for outcome in ['X', 'AI', 'HU'])
            
            if has_positive and has_negative:
                # Opposing signals
                color = colors['dark_grey']
            elif is_high_topic:
                # This topic is associated with high PC values
                all_positive = all(pc_associations.get(outcome) == 'positive' 
                                 for outcome in ['X', 'AI', 'HU'])
                if all_positive:
                    color = colors['dark_red']
                else:
                    color = colors['light_red']
            elif is_low_topic:
                # This topic is associated with low PC values
                all_negative = all(pc_associations.get(outcome) == 'negative' 
                                 for outcome in ['X', 'AI', 'HU'])
                if all_negative:
                    color = colors['dark_blue']
                else:
                    color = colors['light_blue']
            else:
                # Topic is in middle quartiles
                color = colors['light_grey']
            
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
            'validity': Color based on X/HU/AI agreement
            
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
                Patch(facecolor='#8B0000', label='Consensus high'),
                Patch(facecolor='#FF6B6B', label='Any high'),
                Patch(facecolor='#00008B', label='Consensus low'),
                Patch(facecolor='#6B9AFF', label='Any low'),
                Patch(facecolor='#4A4A4A', label='Opposing'),
                Patch(facecolor='#B0B0B0', label='All middle')
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