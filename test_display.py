import numpy as np
from haam import HAAM

# Set random seed
np.random.seed(42)

# Generate synthetic data
n = 1000
p = 200

X = np.random.randn(n, p)
Y = np.random.randn(n)
AI = 0.3 * Y + 0.1 * X[:, 0] + 0.1 * X[:, 1] + np.random.randn(n) * 0.5
HU = 0.3 * Y + 0.1 * X[:, 2] + 0.1 * X[:, 3] + np.random.randn(n) * 0.5

# Run analysis
haam = HAAM(Y, AI, HU, standardize=False)
results = haam.fit_debiased_lasso(X)

print("\nTest complete! The display should now show:")
print("1. Clean coefficient table without stars")
print("2. Separate sections for C, G, PoMA values")
print("3. Properly labeled value-prediction correlations")
print("4. CV R² clearly labeled in feature selection summary")