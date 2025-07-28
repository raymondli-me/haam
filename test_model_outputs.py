import numpy as np
import os
from haam.haam_package import HAAMAnalysis

# Set random seed
np.random.seed(456)

# Generate data with more structure for better testing
n = 1500
p = 50  # Fewer PCs for easier visualization

# Create structured embeddings
embeddings = np.random.randn(n, p)
# Add some correlation structure
for i in range(5):
    embeddings[:, i*5:(i+1)*5] = embeddings[:, i*5:i*5+1] + 0.3 * np.random.randn(n, 5)

# Create outcomes with known relationships to specific PCs
# Y depends on PC1, PC2, PC10
criterion = (0.5 * embeddings[:, 0] + 0.3 * embeddings[:, 1] + 
             0.2 * embeddings[:, 9] + 0.5 * np.random.randn(n))

# AI depends on Y and PC3, PC4
ai_judgment = (0.4 * criterion + 0.3 * embeddings[:, 2] + 
               0.2 * embeddings[:, 3] + 0.6 * np.random.randn(n))

# HU depends on Y and PC5, PC6 
human_judgment = (0.5 * criterion + 0.25 * embeddings[:, 4] + 
                  0.15 * embeddings[:, 5] + 0.5 * np.random.randn(n))

print("Testing comprehensive model outputs display and export...")
print("="*80)

# Run analysis
analysis = HAAMAnalysis(
    criterion=criterion,
    ai_judgment=ai_judgment,
    human_judgment=human_judgment,
    embeddings=embeddings,
    n_components=p,  # Use all components
    standardize=False
)

# Fit models - this will display all outputs automatically
results = analysis.fit_debiased_lasso()

print("\n\nNow testing export functionality...")
print("="*80)

# Export results
export_dir = os.getcwd()
export_paths = analysis.export_results(output_dir=export_dir)

print("\n\nChecking exported files...")
# Check if files exist and show first few lines
import pandas as pd

print("\n1. LASSO Model Outputs (first 10 rows):")
lasso_df = pd.read_csv('lasso_model_outputs.csv')
print(lasso_df.head(10))
print(f"Shape: {lasso_df.shape}")

print("\n\n2. Post-LASSO Model Outputs (first 10 rows):")
post_lasso_df = pd.read_csv('post_lasso_model_outputs.csv')
print(post_lasso_df.head(10))
print(f"Shape: {post_lasso_df.shape}")

print("\n\n3. Model Summary:")
summary_df = pd.read_csv('model_summary.csv')
print(summary_df)

print("\n\nTest complete! Check the display above and the exported CSV files.")