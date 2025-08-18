# Degrees of Freedom Analysis in HAAM Package

## Summary of Findings

After analyzing the HAAM package code, here's how degrees of freedom (df) are calculated for different components:

## 1. Total Effects (Simple OLS Regression)

### Location
- File: `/haam/haam_package.py`
- Function: `_calculate_total_effects_dml()` (lines ~423-553)

### Sample Size (n) Calculation
- For Y → AI path: `mask = ~(np.isnan(self.criterion) | np.isnan(self.ai_judgment))`
- For Y → HU path: `mask = ~(np.isnan(self.criterion) | np.isnan(self.human_judgment))`  
- For HU → AI path: `mask = ~(np.isnan(self.human_judgment) | np.isnan(self.ai_judgment))`
- n = `mask.sum()` (number of non-missing observations)

### Degrees of Freedom
- The OLS model is fitted using statsmodels: `model = sm.OLS(y, X).fit()`
- Where X includes a constant: `X = sm.add_constant(X_data)`
- **Implicit df = n - 2** (n observations minus 2 parameters: intercept and slope)
- The p-values are calculated using normal distribution: `pval = 2 * stats.norm.cdf(-abs(te['coefficient']/te['se']))`

### Note on P-value Calculation
- The code uses `stats.norm.cdf()` (normal distribution) rather than `stats.t.cdf()` (t-distribution)
- This is appropriate for large samples where t-distribution converges to normal
- The threshold check `if mask.sum() > 50` ensures sufficient sample size

## 2. DML (Double Machine Learning) Check Betas

### Location
- File: `/haam/haam_package.py`
- Function: `_calculate_dml_check_beta()` (lines ~554-691)

### Sample Size (n) Calculation
- Uses 5-fold cross-validation
- n = `len(X_var)` (total number of observations)
- Each fold uses approximately n/5 observations for testing

### Standard Error Calculation
- Uses sandwich estimator following Chernozhukov et al. (2018)
- Formula: `variance = (1/n) * Σ(ψᵢ²) / (𝔼[e²ₓᵢ])²`
- Where ψᵢ = e_Xi * e_Yi - θ̂ * e_Xi²
- Standard error: `se = √(variance / n)`

### Degrees of Freedom
- **Uses asymptotic normal distribution** (not t-distribution)
- P-values: `pval = 2 * stats.norm.cdf(-np.abs(t_stat))`
- 95% CI: `theta ± 1.96 * se`
- This is standard for DML as it relies on asymptotic theory

## 3. Post-LASSO OLS Regression

### Location
- File: `/haam/haam_package.py`
- Function: `_fit_single_debiased_lasso()` (lines ~234-345)

### Two Scenarios

#### With Sample Splitting (n ≥ 100)
- Stage 1: LASSO on first half (n/2 observations)
- Stage 2: OLS on second half (n/2 observations)
- OLS model: `ols_model = sm.OLS(y_estimate, sm.add_constant(X_selected))`
- Uses robust standard errors: `ols_result = ols_model.fit(cov_type='HC3')`
- **Implicit df = n/2 - (k+1)** where k = number of selected features

#### Without Sample Splitting (n < 100)
- Uses all n observations
- Same OLS fitting with HC3 robust standard errors
- **Implicit df = n - (k+1)** where k = number of selected features

## Key Observations

1. **The package never explicitly calculates df = n - 2**
   - For simple OLS (total effects), statsmodels handles df internally
   - The code retrieves standard errors via `model.bse[1]`

2. **Asymptotic assumptions**
   - All p-values use normal distribution, not t-distribution
   - This is justified by the sample size checks (mask.sum() > 50)
   - DML theory is inherently asymptotic

3. **Robust standard errors**
   - Post-LASSO OLS uses HC3 (heteroskedasticity-consistent) standard errors
   - This affects the standard error calculation but not the degrees of freedom

4. **Sample size used for each calculation**
   - Total effects: Only non-missing pairs of observations
   - DML: All available observations with cross-validation
   - Post-LASSO: Half the data (with splitting) or all data (without splitting)

## Conclusion

The HAAM package doesn't explicitly calculate degrees of freedom. Instead:
- For OLS models, it relies on statsmodels' internal df calculation
- For inference, it uses asymptotic normal distribution throughout
- The effective degrees of freedom would be:
  - Total effects OLS: df = n - 2
  - Post-LASSO OLS: df = n_used - (k_selected + 1)
  - DML: Uses asymptotic distribution (effectively df = ∞)