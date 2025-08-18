#!/usr/bin/env python3
"""
Word Cloud Validity Coloring Example
====================================

This example demonstrates how to use the validity coloring mode for word clouds.
Validity coloring shows whether topics are:
- Valid markers: Associated with high/low values in all three measures (Y, HU, AI)
- Perceived markers: Associated with high/low values in HU & AI but not Y
- Mixed signals: Inconsistent patterns across measures

Color Legend:
- Dark Red: Valid high marker (top quartile in Y+HU+AI)
- Light Red: Perceived high marker (top quartile in HU+AI only)
- Dark Blue: Valid low marker (bottom quartile in Y+HU+AI)
- Light Blue: Perceived low marker (bottom quartile in HU+AI only)
- Dark Grey: Mixed strong signals
- Light Grey: Mixed weak signals
"""

import numpy as np
import pandas as pd
from haam import HAAM
import os

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 1000

# Create synthetic texts (for demonstration)
texts = []
for i in range(n_samples):
    if i < 250:
        texts.append("Private school education, country club membership, luxury vacation homes")
    elif i < 500:
        texts.append("College degree, professional career, suburban home ownership")
    elif i < 750:
        texts.append("High school diploma, skilled trades, apartment rental")
    else:
        texts.append("Limited education, service jobs, public housing assistance")

# Create synthetic social class (ground truth)
social_class = np.concatenate([
    np.random.normal(4, 0.3, 250),  # High class
    np.random.normal(3, 0.3, 250),  # Upper middle
    np.random.normal(2, 0.3, 250),  # Lower middle
    np.random.normal(1, 0.3, 250),  # Low class
])

# Create AI judgments (mostly accurate)
ai_judgment = social_class + np.random.normal(0, 0.2, n_samples)

# Create human judgments (with some systematic biases)
human_judgment = social_class + np.random.normal(0, 0.3, n_samples)
# Add bias: humans overestimate certain phrases
for i, text in enumerate(texts):
    if "luxury" in text or "country club" in text:
        human_judgment[i] += 0.5  # Humans rate these higher
    if "public housing" in text:
        human_judgment[i] -= 0.3  # Humans rate these lower

print("="*80)
print("WORD CLOUD VALIDITY COLORING EXAMPLE")
print("="*80)

# Initialize HAAM analysis
print("\n1. Running HAAM analysis...")
haam = HAAM(
    criterion=social_class,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=texts,
    n_components=50,  # Use fewer components for example
    auto_run=True,
    standardize=True
)

print("✓ HAAM analysis complete")

# Create output directory
output_dir = './validity_wordcloud_example'
os.makedirs(output_dir, exist_ok=True)

# 2. Generate word clouds with POLE coloring (original)
print("\n2. Generating word clouds with POLE coloring (red/blue gradients)...")
pc_indices = [0, 1, 2, 3, 4]  # First 5 PCs

for pc_idx in pc_indices[:2]:  # Show first 2 for brevity
    print(f"\n   PC{pc_idx + 1} - Pole coloring:")
    fig, high_path, low_path = haam.create_pc_wordclouds(
        pc_idx=pc_idx,
        k=10,
        max_words=100,
        output_dir=os.path.join(output_dir, 'pole_coloring'),
        display=False,
        color_mode='pole'
    )
    print(f"   ✓ Saved: {high_path}")
    print(f"   ✓ Saved: {low_path}")

# 3. Generate word clouds with VALIDITY coloring
print("\n3. Generating word clouds with VALIDITY coloring...")
print("   (Shows which topics are valid vs perceived markers)")

for pc_idx in pc_indices[:2]:  # Show first 2 for brevity
    print(f"\n   PC{pc_idx + 1} - Validity coloring:")
    fig, high_path, low_path = haam.create_pc_wordclouds(
        pc_idx=pc_idx,
        k=10,
        max_words=100,
        output_dir=os.path.join(output_dir, 'validity_coloring'),
        display=False,
        color_mode='validity'
    )
    print(f"   ✓ Saved: {high_path}")
    print(f"   ✓ Saved: {low_path}")

# 4. Create grid visualizations for comparison
print("\n4. Creating grid visualizations...")

# Grid with pole coloring
print("\n   Creating grid with POLE coloring...")
grid_fig_pole = haam.create_top_pcs_wordcloud_grid(
    pc_indices=pc_indices[:9],
    k=8,
    max_words=50,
    output_file=os.path.join(output_dir, 'grid_pole_coloring.png'),
    display=False,
    color_mode='pole'
)
print("   ✓ Saved: grid_pole_coloring.png")

# Grid with validity coloring
print("\n   Creating grid with VALIDITY coloring...")
grid_fig_validity = haam.create_top_pcs_wordcloud_grid(
    pc_indices=pc_indices[:9],
    k=8,
    max_words=50,
    output_file=os.path.join(output_dir, 'grid_validity_coloring.png'),
    display=False,
    color_mode='validity'
)
print("   ✓ Saved: grid_validity_coloring.png")

# 5. Batch generation with validity coloring
print("\n5. Batch generating all word clouds with validity coloring...")
output_paths = haam.create_all_pc_wordclouds(
    pc_indices=pc_indices,
    k=10,
    max_words=100,
    output_dir=os.path.join(output_dir, 'batch_validity'),
    display=False,
    color_mode='validity'
)

print(f"\n✓ Generated {len(output_paths)} word clouds with validity coloring")

# 6. Explain the results
print("\n" + "="*80)
print("UNDERSTANDING VALIDITY COLORING")
print("="*80)

print("\nValidity coloring helps identify:")
print("1. VALID MARKERS (dark colors):")
print("   - Topics that genuinely indicate high/low social class")
print("   - Consistent across ground truth (Y), human ratings (HU), and AI ratings")
print("   - These are reliable indicators")

print("\n2. PERCEIVED MARKERS (light colors):")
print("   - Topics that humans and AI associate with high/low class")
print("   - But NOT actually correlated with true social class")
print("   - These represent biases or stereotypes")

print("\n3. MIXED SIGNALS (grey colors):")
print("   - Topics with inconsistent associations")
print("   - May indicate complex or ambiguous markers")

print("\nColor meanings:")
print("- Dark Red: Valid high-class markers")
print("- Light Red: Perceived (but not valid) high-class markers") 
print("- Dark Blue: Valid low-class markers")
print("- Light Blue: Perceived (but not valid) low-class markers")
print("- Grey: Mixed or weak signals")

print(f"\n✓ All results saved to: {output_dir}/")
print("\nDone!")