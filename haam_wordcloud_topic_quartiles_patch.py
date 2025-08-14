"""
Patch for HAAM WordCloud to use topic-level quartiles
====================================================
This patch modifies the color assignment to use quartiles based on 
topic means rather than document-level quartiles, making dark colors
achievable.
"""

# The issue: Current code calculates quartiles from all documents,
# but compares against topic means. Topic means are compressed toward
# center due to averaging, rarely reaching document-level extremes.

# Solution: Calculate quartiles from topic means, or use looser thresholds

def _calculate_topic_validity_colors_fixed(self, topics: List[Dict], pc_idx: int) -> Dict[str, str]:
    """
    Fixed version that uses topic-level statistics for more reasonable coloring.
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
    
    # Check if we have Y/HU/AI data for direct measurement
    if self.criterion is None or self.human_judgment is None or self.ai_judgment is None:
        # Fall back to old method if data not available
        return self._calculate_topic_validity_colors_legacy(topics, pc_idx)
    
    # First, calculate means for ALL topics to establish topic-level quartiles
    all_cluster_ids = np.unique(self.topic_analyzer.cluster_labels[self.topic_analyzer.cluster_labels != -1])
    topic_means = {'Y': [], 'HU': [], 'AI': []}
    
    for cluster_id in all_cluster_ids:
        topic_mask = self.topic_analyzer.cluster_labels == cluster_id
        if np.any(topic_mask):
            # Y (criterion)
            y_values = self.criterion[topic_mask]
            y_valid = y_values[~np.isnan(y_values)]
            if len(y_valid) > 0:
                topic_means['Y'].append(np.mean(y_valid))
            
            # HU (human judgment)
            hu_values = self.human_judgment[topic_mask]
            hu_valid = hu_values[~np.isnan(hu_values)]
            if len(hu_valid) > 0:
                topic_means['HU'].append(np.mean(hu_valid))
            
            # AI (ai judgment)
            ai_values = self.ai_judgment[topic_mask]
            ai_valid = ai_values[~np.isnan(ai_values)]
            if len(ai_valid) > 0:
                topic_means['AI'].append(np.mean(ai_valid))
    
    # Calculate quartiles based on TOPIC MEANS, not document values
    # Use 20th/80th percentiles for more achievable thresholds
    topic_quartiles = {}
    for measure in ['Y', 'HU', 'AI']:
        if topic_means[measure]:
            # Use 20/80 instead of 25/75 for more topics in extreme categories
            topic_quartiles[f'{measure}_q20'] = np.percentile(topic_means[measure], 20)
            topic_quartiles[f'{measure}_q80'] = np.percentile(topic_means[measure], 80)
        else:
            topic_quartiles[f'{measure}_q20'] = np.nan
            topic_quartiles[f'{measure}_q80'] = np.nan
    
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
            # Calculate mean Y/HU/AI values for this topic
            y_values = self.criterion[topic_mask]
            y_valid = y_values[~np.isnan(y_values)]
            y_mean = np.mean(y_valid) if len(y_valid) > 0 else np.nan
            
            hu_values = self.human_judgment[topic_mask]
            hu_valid = hu_values[~np.isnan(hu_values)]
            hu_mean = np.mean(hu_valid) if len(hu_valid) > 0 else np.nan
            
            ai_values = self.ai_judgment[topic_mask]
            ai_valid = ai_values[~np.isnan(ai_values)]
            ai_mean = np.mean(ai_valid) if len(ai_valid) > 0 else np.nan
            
            # Determine positions relative to TOPIC-LEVEL quartiles
            # Count how many valid measures we have
            valid_count = sum([not np.isnan(x) for x in [y_mean, hu_mean, ai_mean]])
            
            if valid_count == 0:
                color = colors['light_grey']
            else:
                # Count high/low signals using topic-level thresholds
                n_high = 0
                n_low = 0
                
                if not np.isnan(y_mean):
                    if y_mean >= topic_quartiles['Y_q80']:
                        n_high += 1
                    elif y_mean <= topic_quartiles['Y_q20']:
                        n_low += 1
                
                if not np.isnan(hu_mean):
                    if hu_mean >= topic_quartiles['HU_q80']:
                        n_high += 1
                    elif hu_mean <= topic_quartiles['HU_q20']:
                        n_low += 1
                
                if not np.isnan(ai_mean):
                    if ai_mean >= topic_quartiles['AI_q80']:
                        n_high += 1
                    elif ai_mean <= topic_quartiles['AI_q20']:
                        n_low += 1
                
                # Assign color based on pattern
                # Adjust for partial data - if all available measures agree, use dark color
                if valid_count < 3:
                    # With missing data, check if all AVAILABLE measures agree
                    if n_high == valid_count and n_high > 0:
                        color = colors['dark_red']  # All available are high
                    elif n_low == valid_count and n_low > 0:
                        color = colors['dark_blue']  # All available are low
                    elif n_high > 0 and n_low > 0:
                        color = colors['dark_grey']  # Mixed signals
                    elif n_high > 0:
                        color = colors['light_red']  # Some high
                    elif n_low > 0:
                        color = colors['light_blue']  # Some low
                    else:
                        color = colors['light_grey']  # All middle
                else:
                    # All three measures available - use original logic
                    if n_high > 0 and n_low > 0:
                        color = colors['dark_grey']  # Mixed signals
                    elif n_high == 3:
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
                
    return word_colors