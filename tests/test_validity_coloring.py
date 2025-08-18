#!/usr/bin/env python3
"""
Test script for validity coloring feature
"""

import numpy as np
from haam import HAAM
import matplotlib.pyplot as plt

# Generate simple test data
np.random.seed(42)
n_samples = 500

# Create distinct text patterns
texts = []
criterion = []
ai_judgment = []
human_judgment = []

# Group 1: Valid high markers (high in all three)
for i in range(100):
    texts.append("Elite university degree investment banking private equity")
    criterion.append(4 + np.random.normal(0, 0.2))
    ai_judgment.append(4 + np.random.normal(0, 0.2))
    human_judgment.append(4 + np.random.normal(0, 0.2))

# Group 2: Perceived high markers (high in HU/AI, not Y)
for i in range(100):
    texts.append("Designer clothes luxury brands expensive jewelry")
    criterion.append(2.5 + np.random.normal(0, 0.2))  # Actually middle class
    ai_judgment.append(4 + np.random.normal(0, 0.2))  # AI thinks high
    human_judgment.append(4 + np.random.normal(0, 0.2))  # Humans think high

# Group 3: Valid low markers (low in all three)
for i in range(100):
    texts.append("Public assistance food stamps welfare housing")
    criterion.append(1 + np.random.normal(0, 0.2))
    ai_judgment.append(1 + np.random.normal(0, 0.2))
    human_judgment.append(1 + np.random.normal(0, 0.2))

# Group 4: Perceived low markers (low in HU/AI, not Y)
for i in range(100):
    texts.append("Fast food retail jobs community college")
    criterion.append(2.5 + np.random.normal(0, 0.2))  # Actually middle class
    ai_judgment.append(1 + np.random.normal(0, 0.2))  # AI thinks low
    human_judgment.append(1 + np.random.normal(0, 0.2))  # Humans think low

# Group 5: Mixed signals
for i in range(100):
    texts.append("Small business entrepreneur startup founder")
    criterion.append(3 + np.random.normal(0, 0.5))
    ai_judgment.append(2 + np.random.normal(0, 0.5))
    human_judgment.append(4 + np.random.normal(0, 0.5))

# Convert to arrays
criterion = np.array(criterion)
ai_judgment = np.array(ai_judgment)
human_judgment = np.array(human_judgment)

print("Running HAAM analysis on test data...")
haam = HAAM(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    texts=texts,
    n_components=20,
    auto_run=True,
    standardize=True
)

print("\nGenerating word clouds with validity coloring...")

# Test on first PC
fig, _, _ = haam.create_pc_wordclouds(
    pc_idx=0,
    k=10,
    max_words=50,
    color_mode='validity',
    display=True
)

print("\n✓ Validity coloring test complete!")
print("\nExpected colors:")
print("- 'elite university investment': Dark red (valid high)")
print("- 'designer luxury jewelry': Light red (perceived high)")
print("- 'assistance stamps welfare': Dark blue (valid low)")
print("- 'fast retail community': Light blue (perceived low)")
print("- 'business entrepreneur startup': Grey (mixed signals)")