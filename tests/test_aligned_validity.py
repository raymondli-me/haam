#!/usr/bin/env python3

"""
Test Script for Aligned Validity Coloring
========================================
This script tests that the validity coloring and labels are now properly aligned.
"""

# For Google Colab
# !pip install wordcloud
# !pip install git+https://github.com/raymondli-me/haam.git

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haam import HAAM

print("="*80)
print("TESTING ALIGNED VALIDITY COLORING")
print("="*80)

# Generate synthetic test data
np.random.seed(42)
n_samples = 1000

# Create embeddings with clear patterns
embeddings = np.random.randn(n_samples, 100)
embeddings[:250, :10] += 2  # Group 1: high on first 10 dims
embeddings[250:500, 10:20] += 2  # Group 2: high on next 10 dims
embeddings[500:750, 20:30] += 2  # Group 3: high on next 10 dims
embeddings[750:, 30:40] += 2  # Group 4: high on next 10 dims

# Create outcomes with clear patterns
# Group 1: High on all measures
# Group 2: High on HU/AI, low on Y
# Group 3: Low on all measures
# Group 4: Mixed signals
criterion = np.zeros(n_samples)
human_judgment = np.zeros(n_samples)
ai_judgment = np.zeros(n_samples)

# Group 1: All high
criterion[:250] = np.random.normal(8, 0.5, 250)
human_judgment[:250] = np.random.normal(8, 0.5, 250)
ai_judgment[:250] = np.random.normal(8, 0.5, 250)

# Group 2: Y low, HU/AI high
criterion[250:500] = np.random.normal(2, 0.5, 250)
human_judgment[250:500] = np.random.normal(8, 0.5, 250)
ai_judgment[250:500] = np.random.normal(8, 0.5, 250)

# Group 3: All low
criterion[500:750] = np.random.normal(2, 0.5, 250)
human_judgment[500:750] = np.random.normal(2, 0.5, 250)
ai_judgment[500:750] = np.random.normal(2, 0.5, 250)

# Group 4: Mixed
criterion[750:] = np.random.normal(8, 0.5, 250)
human_judgment[750:] = np.random.normal(2, 0.5, 250)
ai_judgment[750:] = np.random.normal(5, 0.5, 250)

# Create simple texts
texts = []
for i in range(n_samples):
    if i < 250:
        texts.append("wealthy rich affluent prosperous successful elite")
    elif i < 500:
        texts.append("aspiring ambitious striving hopeful seeking upward")
    elif i < 750:
        texts.append("poor struggling hardship difficult challenging tough")
    else:
        texts.append("mixed varied diverse complex nuanced unclear")

print("\nData created with clear patterns:")
print("- Group 1 (0-250): High Y, High HU, High AI → Should be DARK RED")
print("- Group 2 (250-500): Low Y, High HU, High AI → Should be LIGHT RED/GREY")
print("- Group 3 (500-750): Low Y, Low HU, Low AI → Should be DARK BLUE")
print("- Group 4 (750-1000): High Y, Low HU, Mid AI → Should be DARK GREY (mixed)")

# Initialize HAAM
print("\nInitializing HAAM...")
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    embeddings=embeddings,
    texts=texts,
    n_components=50,
    min_cluster_size=10,
    min_samples=2,
    umap_n_components=3,
    standardize=True,
    sample_split_post_lasso=False,
    auto_run=True
)

print("\nGenerating test word cloud for PC0...")
fig, _, _ = haam.create_pc_wordclouds(
    pc_idx=0,
    k=3,
    max_words=50,
    figsize=(12, 6),
    display=True,
    color_mode='validity'
)

# Also create a function to verify the alignment
def check_topic_alignment(haam_instance, pc_idx=0, k=3):
    """Check if colors and labels align properly."""
    print(f"\n{'='*60}")
    print(f"CHECKING ALIGNMENT FOR PC{pc_idx + 1}")
    print(f"{'='*60}")
    
    # Get topics
    pc_topics = haam_instance.topic_analyzer.get_pc_high_low_topics(
        pc_idx=pc_idx, n_high=k, n_low=k, p_threshold=0.05
    )
    
    # Calculate global quartiles
    y_q25 = np.percentile(haam_instance.criterion, 25)
    y_q75 = np.percentile(haam_instance.criterion, 75)
    hu_q25 = np.percentile(haam_instance.human_judgment, 25)
    hu_q75 = np.percentile(haam_instance.human_judgment, 75)
    ai_q25 = np.percentile(haam_instance.ai_judgment, 25)
    ai_q75 = np.percentile(haam_instance.ai_judgment, 75)
    
    print(f"\nGlobal quartiles:")
    print(f"Y: Q25={y_q25:.2f}, Q75={y_q75:.2f}")
    print(f"HU: Q25={hu_q25:.2f}, Q75={hu_q75:.2f}")
    print(f"AI: Q25={ai_q25:.2f}, Q75={ai_q75:.2f}")
    
    for pole in ['high', 'low']:
        print(f"\n{pole.upper()} POLE TOPICS:")
        topics = pc_topics.get(pole, [])
        
        for i, topic in enumerate(topics):
            topic_id = topic['topic_id']
            topic_mask = haam_instance.topic_analyzer.cluster_labels == topic_id
            
            if np.any(topic_mask):
                y_mean = np.mean(haam_instance.criterion[topic_mask])
                hu_mean = np.mean(haam_instance.human_judgment[topic_mask])
                ai_mean = np.mean(haam_instance.ai_judgment[topic_mask])
                
                # Determine quartiles
                y_pos = 'H' if y_mean >= y_q75 else ('L' if y_mean <= y_q25 else 'M')
                hu_pos = 'H' if hu_mean >= hu_q75 else ('L' if hu_mean <= hu_q25 else 'M')
                ai_pos = 'H' if ai_mean >= ai_q75 else ('L' if ai_mean <= ai_q25 else 'M')
                
                # Expected color
                n_high = sum([p == 'H' for p in [y_pos, hu_pos, ai_pos]])
                n_low = sum([p == 'L' for p in [y_pos, hu_pos, ai_pos]])
                
                if n_high > 0 and n_low > 0:
                    expected_color = "DARK GREY (mixed)"
                elif n_high == 3:
                    expected_color = "DARK RED (all high)"
                elif n_high > 0:
                    expected_color = "LIGHT RED (any high)"
                elif n_low == 3:
                    expected_color = "DARK BLUE (all low)"
                elif n_low > 0:
                    expected_color = "LIGHT BLUE (any low)"
                else:
                    expected_color = "LIGHT GREY (all middle)"
                
                print(f"\nTopic {topic_id}:")
                print(f"  Size: {topic['size']} documents")
                print(f"  Keywords: {topic['keywords'][:40]}...")
                print(f"  Means: Y={y_mean:.2f}, HU={hu_mean:.2f}, AI={ai_mean:.2f}")
                print(f"  Quartiles: Y:{y_pos} HU:{hu_pos} AI:{ai_pos}")
                print(f"  Expected color: {expected_color}")

# Run alignment check
check_topic_alignment(haam, pc_idx=0)

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print("\nThe colors should now match the quartile positions:")
print("- Words from topics with Y:H HU:H AI:H → DARK RED")
print("- Words from topics with Y:L HU:L AI:L → DARK BLUE")
print("- Mixed patterns → appropriate mixed colors")
print("\nCheck the word cloud above to verify alignment!")