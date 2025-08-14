#!/usr/bin/env python3
"""
Test script for UMAP PC arrow plot coloring modes
=================================================

This script tests both legacy and validity coloring modes
to ensure they work correctly.
"""

import numpy as np
import pandas as pd
from haam import HAAM

# Create synthetic test data
np.random.seed(42)
n_samples = 1000

# Create data with clear patterns
# Group 1: High Y, HU, AI (indices 0-333)
# Group 2: Low Y, HU, AI (indices 334-666)
# Group 3: Mixed patterns (indices 667-999)

y_values = np.concatenate([
    np.random.normal(4.0, 0.3, 334),  # High group
    np.random.normal(2.0, 0.3, 333),  # Low group
    np.random.normal(3.0, 0.5, 333)   # Mixed group
])

hu_values = np.concatenate([
    np.random.normal(4.0, 0.3, 334),  # High group
    np.random.normal(2.0, 0.3, 333),  # Low group
    np.random.normal(3.0, 0.5, 333)   # Mixed group
])

ai_values = np.concatenate([
    np.random.normal(4.0, 0.3, 334),  # High group
    np.random.normal(2.0, 0.3, 333),  # Low group
    np.random.normal(3.0, 0.5, 333)   # Mixed group
])

# Create embeddings with some structure
embeddings = np.random.randn(n_samples, 100)
# Add structure based on groups
embeddings[:334] += np.random.randn(100) * 0.5  # High group bias
embeddings[334:667] -= np.random.randn(100) * 0.5  # Low group bias

# Create texts
texts = [f"Sample text {i} group {i//334}" for i in range(n_samples)]

print("Initializing HAAM with synthetic data...")
haam = HAAM(
    criterion=y_values,
    ai_judgment=ai_values,
    human_judgment=hu_values,
    embeddings=embeddings,
    texts=texts,
    n_components=50,
    min_cluster_size=10,
    min_samples=2,
    standardize=True,
    auto_run=True
)

print("\nCreating arrow plots with different color modes...")

# Test 1: Legacy mode (default)
print("\n1. Creating arrow plot with LEGACY coloring (PC-based inference)...")
output_legacy = haam.create_3d_umap_with_pc_arrows(
    pc_indices=[0, 1, 2],
    arrow_mode='list',
    color_mode='legacy',  # Default
    output_dir='./test_arrow_plots',
    display=False
)
print(f"   Saved: {output_legacy}")

# Test 2: Validity mode (new)
print("\n2. Creating arrow plot with VALIDITY coloring (direct measurement)...")
output_validity = haam.create_3d_umap_with_pc_arrows(
    pc_indices=[0, 1, 2],
    arrow_mode='list',
    color_mode='validity',  # New mode
    output_dir='./test_arrow_plots',
    display=False
)
print(f"   Saved: {output_validity}")

# Test 3: Single PC with validity mode
print("\n3. Creating single PC arrow with validity coloring...")
output_single = haam.create_3d_umap_with_pc_arrows(
    pc_indices=0,
    arrow_mode='single',
    color_mode='validity',
    output_dir='./test_arrow_plots',
    display=False
)
print(f"   Saved: {output_single}")

# Test 4: Compare with word clouds
print("\n4. Creating word cloud for comparison...")
fig, high_path, low_path = haam.create_pc_wordclouds(
    pc_idx=0,
    k=3,
    color_mode='validity',  # Should use same coloring
    output_dir='./test_arrow_plots',
    display=False
)
print(f"   Saved word clouds: {high_path}, {low_path}")

print("\n✅ All tests completed!")
print("\nColor meanings:")
print("- Dark Red (#8B0000): Consensus high (all in top quartile)")
print("- Light Red (#FF6B6B): Some high (at least one in top quartile)")
print("- Dark Blue (#00008B): Consensus low (all in bottom quartile)")
print("- Light Blue (#6B9AFF): Some low (at least one in bottom quartile)")
print("- Dark Grey (#4A4A4A): Opposing signals (mix of high and low)")
print("- Light Grey (#B0B0B0): All middle quartiles")
print("\nCheck the output files to verify colors are consistent between:")
print("1. Legacy vs Validity modes")
print("2. Arrow plots vs Word clouds (when using validity mode)")