import numpy as np
from haam.haam_package import HAAMAnalysis

# Set random seed
np.random.seed(42)

# Generate synthetic data
n = 1000
p = 200

# Embeddings (X)
embeddings = np.random.randn(n, p)

# Outcomes
criterion = np.random.randn(n)  # Y
ai_judgment = 0.3 * criterion + 0.1 * embeddings[:, 0] + 0.1 * embeddings[:, 1] + np.random.randn(n) * 0.5
human_judgment = 0.3 * criterion + 0.1 * embeddings[:, 2] + 0.1 * embeddings[:, 3] + np.random.randn(n) * 0.5

# Run analysis with HAAMAnalysis directly
print("Running HAAM Analysis with new display format...")
analysis = HAAMAnalysis(
    criterion=criterion,
    ai_judgment=ai_judgment, 
    human_judgment=human_judgment,
    embeddings=embeddings,
    standardize=False
)

# Fit debiased LASSO
results = analysis.fit_debiased_lasso()

# Display global statistics - this will show the new format
analysis.display_global_statistics()

print("\n\nThe display now shows:")
print("✓ Coefficient table with β and β̌ (no stars)")
print("✓ Separate sections for C, G, PoMA values")
print("✓ Properly labeled value-prediction correlations")
print("✓ CV R² clearly labeled in feature selection summary")