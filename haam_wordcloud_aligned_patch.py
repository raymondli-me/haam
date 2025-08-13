"""
HAAM Word Cloud Alignment Patch
==============================

This patch modifies the _calculate_topic_validity_colors method in haam_wordcloud.py
to use direct Y/HU/AI measurements instead of PC-based inference.

INSTRUCTIONS FOR APPLYING THIS PATCH:
------------------------------------

1. BACKUP the original file:
   cp haam/haam_wordcloud.py haam/haam_wordcloud_backup.py

2. MODIFY PCWordCloudGenerator.__init__ to accept Y/HU/AI data:
   
   Change lines 23-35 from:
   ```python
   def __init__(self, topic_analyzer, analysis_results=None):
       self.topic_analyzer = topic_analyzer
       self.analysis_results = analysis_results
   ```
   
   To:
   ```python
   def __init__(self, topic_analyzer, analysis_results=None, 
                criterion=None, ai_judgment=None, human_judgment=None):
       self.topic_analyzer = topic_analyzer
       self.analysis_results = analysis_results
       self.criterion = criterion
       self.ai_judgment = ai_judgment
       self.human_judgment = human_judgment
   ```

3. REPLACE the entire _calculate_topic_validity_colors method (lines 272-419) with
   the new implementation below.

4. UPDATE calls to PCWordCloudGenerator in your code to pass Y/HU/AI data:
   
   Example:
   ```python
   wordcloud_gen = PCWordCloudGenerator(
       topic_analyzer=haam.topic_analyzer,
       analysis_results=haam.analysis.results,
       criterion=haam.analysis.criterion,
       ai_judgment=haam.analysis.ai_judgment,
       human_judgment=haam.analysis.human_judgment
   )
   ```

NEW METHOD IMPLEMENTATION:
-------------------------
"""

def _calculate_topic_validity_colors(self, topics: List[Dict], pc_idx: int) -> Dict[str, str]:
    """
    Calculate color for each word based on topic's actual Y/HU/AI values.
    
    This method now uses direct measurement of topic means in Y/HU/AI space
    instead of inferring from PC associations. It:
    1. Calculates mean Y/HU/AI values for documents in each topic
    2. Compares those means to global quartiles
    3. Assigns colors based on quartile positions
    
    Parameters
    ----------
    topics : List[Dict]
        List of topic dictionaries from get_pc_high_low_topics
    pc_idx : int
        PC index (kept for compatibility but not used in new logic)
        
    Returns
    -------
    Dict[str, str]
        Mapping of word to color hex code
    """
    if not hasattr(self.topic_analyzer, 'cluster_labels'):
        return {}
        
    # Check if we have the required Y/HU/AI data
    if (self.criterion is None or self.ai_judgment is None or 
        self.human_judgment is None):
        # Fallback to original method if data not available
        print("Warning: Y/HU/AI data not provided, using fallback coloring")
        return self._calculate_topic_validity_colors_fallback(topics, pc_idx)
    
    # Color definitions (same as original)
    colors = {
        'dark_red': '#8B0000',     # All top quartile
        'light_red': '#FF6B6B',    # At least one top quartile
        'dark_blue': '#00008B',    # All bottom quartile
        'light_blue': '#6B9AFF',   # At least one bottom quartile
        'dark_grey': '#4A4A4A',    # Mixed strong (some high, some low)
        'light_grey': '#B0B0B0'    # All middle quartiles
    }
    
    # Calculate global quartiles for each outcome
    y_q25, y_q75 = np.percentile(self.criterion, [25, 75])
    hu_q25, hu_q75 = np.percentile(self.human_judgment, [25, 75])
    ai_q25, ai_q75 = np.percentile(self.ai_judgment, [25, 75])
    
    word_colors = {}
    
    for topic in topics:
        topic_id = topic['topic_id']
        
        # Get documents belonging to this topic
        topic_mask = self.topic_analyzer.cluster_labels == topic_id
        
        if np.sum(topic_mask) == 0:
            # No documents in this topic, use default color
            color = colors['light_grey']
        else:
            # Calculate mean values for this topic
            y_mean = np.mean(self.criterion[topic_mask])
            hu_mean = np.mean(self.human_judgment[topic_mask])
            ai_mean = np.mean(self.ai_judgment[topic_mask])
            
            # Determine quartile positions
            y_high = y_mean > y_q75
            y_low = y_mean < y_q25
            hu_high = hu_mean > hu_q75
            hu_low = hu_mean < hu_q25
            ai_high = ai_mean > ai_q75
            ai_low = ai_mean < ai_q25
            
            # Count high and low signals
            n_high = sum([y_high, hu_high, ai_high])
            n_low = sum([y_low, hu_low, ai_low])
            
            # Determine color based on quartile positions
            if n_high > 0 and n_low > 0:
                # Mixed signals - some outcomes high, some low
                color = colors['dark_grey']
            elif n_high == 3:
                # All three in top quartile - consensus high
                color = colors['dark_red']
            elif n_high > 0:
                # At least one in top quartile
                color = colors['light_red']
            elif n_low == 3:
                # All three in bottom quartile - consensus low
                color = colors['dark_blue']
            elif n_low > 0:
                # At least one in bottom quartile
                color = colors['light_blue']
            else:
                # All in middle quartiles
                color = colors['light_grey']
        
        # Apply color to all words in this topic
        keywords = topic['keywords'].split(' | ')
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword and len(keyword) > 1:
                word_colors[keyword] = color
                
    return word_colors


def _calculate_topic_validity_colors_fallback(self, topics: List[Dict], pc_idx: int) -> Dict[str, str]:
    """
    Fallback method using the original PC-based logic when Y/HU/AI data not available.
    This is the original implementation from lines 272-419.
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
    
    # For validity checking, we need to know if this PC is associated with
    # high or low values in each outcome (Y/SC, HU, AI)
    pc_associations = {}
    
    if self.analysis_results and 'debiased_lasso' in self.analysis_results:
        # Check if this PC is positively or negatively associated with each outcome
        for outcome in ['SC', 'AI', 'HU']:
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
                         for outcome in ['SC', 'AI', 'HU'])
        has_negative = any(pc_associations.get(outcome) == 'negative' 
                         for outcome in ['SC', 'AI', 'HU'])
        
        if has_positive and has_negative:
            # Opposing signals - some say high, some say low
            color = colors['dark_grey']
            
        elif is_high_topic:
            # This topic is associated with high PC values
            # Check if all outcomes view high PC as high class
            all_positive = all(pc_associations.get(outcome) == 'positive' 
                             for outcome in ['SC', 'AI', 'HU'])
            
            if all_positive:
                color = colors['dark_red']  # All agree: valid high marker
            else:
                color = colors['light_red']  # Any signal high
                
        elif is_low_topic:
            # This topic is associated with low PC values
            # Check if all outcomes view low PC as low class
            all_negative = all(pc_associations.get(outcome) == 'negative' 
                             for outcome in ['SC', 'AI', 'HU'])
            
            if all_negative:
                color = colors['dark_blue']  # All agree: valid low marker
            else:
                color = colors['light_blue']  # Any signal low
                
        else:
            # Topic is in middle quartiles
            color = colors['light_grey']  # All middle: weak signal
        
        # Apply color to all words in this topic
        keywords = topic['keywords'].split(' | ')
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword and len(keyword) > 1:
                word_colors[keyword] = color
                
    return word_colors


"""
ADDITIONAL NOTES:
----------------

1. The new method calculates actual mean values of Y/HU/AI for each topic's documents
   rather than inferring from PC associations.

2. Color assignment logic:
   - Dark red (#8B0000): All three outcomes (Y/HU/AI) in top quartile
   - Light red (#FF6B6B): At least one outcome in top quartile
   - Dark blue (#00008B): All three outcomes in bottom quartile  
   - Light blue (#6B9AFF): At least one outcome in bottom quartile
   - Dark grey (#4A4A4A): Mixed signals (some high, some low)
   - Light grey (#B0B0B0): All outcomes in middle quartiles

3. This approach is more direct and transparent than PC-based inference.

4. The fallback method preserves backward compatibility if Y/HU/AI data
   is not provided to the PCWordCloudGenerator.

5. To test the new coloring:
   ```python
   # Generate word clouds with validity coloring
   fig, high_path, low_path = wordcloud_gen.create_pc_wordclouds(
       pc_idx=0,
       color_mode='validity'
   )
   ```

REQUIRED IMPORTS:
----------------
Make sure these imports are at the top of haam_wordcloud.py:
```python
import numpy as np
```

EXAMPLE USAGE AFTER APPLYING PATCH:
----------------------------------

```python
# Example 1: Updating haam_init.py
# In the generate_wordclouds method of HAAMPackage class, change:

self.wordcloud_generator = PCWordCloudGenerator(
    self.topic_analyzer, 
    self.analysis.results
)

# To:

self.wordcloud_generator = PCWordCloudGenerator(
    self.topic_analyzer, 
    self.analysis.results,
    criterion=self.analysis.criterion,
    ai_judgment=self.analysis.ai_judgment,
    human_judgment=self.analysis.human_judgment
)

# Example 2: Standalone usage
from haam.haam_wordcloud import PCWordCloudGenerator

# Initialize with Y/HU/AI data
wordcloud_gen = PCWordCloudGenerator(
    topic_analyzer=topic_analyzer,
    analysis_results=analysis_results,
    criterion=Y_data,
    ai_judgment=AI_data,
    human_judgment=HU_data
)

# Generate word clouds with aligned validity coloring
fig, high_path, low_path = wordcloud_gen.create_pc_wordclouds(
    pc_idx=0,
    k=10,
    color_mode='validity',  # This will now use direct Y/HU/AI measurements
    output_dir='./wordclouds_aligned'
)

# Example 3: Checking if the new method is being used
# Add this debug print in _calculate_topic_validity_colors to verify:
if self.criterion is not None:
    print("Using direct Y/HU/AI measurement for coloring")
else:
    print("Using PC-based inference for coloring (fallback)")
```

BENEFITS OF THIS APPROACH:
-------------------------
1. Direct measurement: Colors now reflect actual Y/HU/AI values for each topic
2. Consistency: Word cloud colors align with topic validity labels
3. Transparency: No indirect inference through PC associations
4. Backward compatibility: Falls back to original method if Y/HU/AI not provided
5. Interpretability: Colors directly indicate whether topics mark high/low class

TESTING THE PATCH:
-----------------
```python
# Test that colors match expectations
# Topics with high Y/HU/AI means should be red
# Topics with low Y/HU/AI means should be blue
# Topics with mixed signals should be grey

# You can verify by checking topic means:
for topic in topics:
    topic_mask = topic_analyzer.cluster_labels == topic['topic_id']
    print(f"Topic {topic['topic_id']}:")
    print(f"  Y mean: {np.mean(Y_data[topic_mask]):.2f}")
    print(f"  HU mean: {np.mean(HU_data[topic_mask]):.2f}")
    print(f"  AI mean: {np.mean(AI_data[topic_mask]):.2f}")
    print(f"  Keywords: {topic['keywords'][:50]}...")
```
"""