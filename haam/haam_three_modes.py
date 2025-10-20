"""
HAAM Three Modes Extension
===========================

This module extends the HAAM package to support three estimation modes:
1. post-lasso: LASSO for variable selection + OLS on selected (original)
2. lasso: LASSO coefficients only (maximum regularization)
3. multiple-regression: OLS on ALL PCs (no selection, vanilla regression)

Developer: Updated 2025-10-20
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')


def fit_all_three_modes(haam_analysis_instance,
                        use_sample_splitting: bool = True) -> Dict[str, Any]:
    """
    Fit all three estimation modes for HAAM analysis.

    This replaces the standard fit_debiased_lasso() method to generate
    results for all three estimation approaches.

    Parameters
    ----------
    haam_analysis_instance : HAAMAnalysis
        Instance of HAAMAnalysis class
    use_sample_splitting : bool, default=True
        Whether to use sample splitting for post-lasso mode

    Returns
    -------
    Dict[str, Any]
        Dictionary with results for all three modes
    """
    analysis = haam_analysis_instance

    outcomes = {
        'X': analysis.criterion,
        'AI': analysis.ai_judgment,
        'HU': analysis.human_judgment
    }

    # Initialize storage for all three modes
    analysis.results['post_lasso'] = {}
    analysis.results['lasso'] = {}
    analysis.results['multiple_regression'] = {}

    print("\n" + "="*80)
    print("FITTING ALL THREE ESTIMATION MODES")
    print("="*80)

    for outcome_name, outcome_values in outcomes.items():
        print(f"\n{outcome_name} Model:")
        print("-" * 60)

        # Remove NaN values
        mask = ~np.isnan(outcome_values)
        if mask.sum() < 50:
            print(f"  Insufficient data (n={mask.sum()})")
            continue

        X = analysis.results['pca_features'][mask]
        y = outcome_values[mask]
        n_features = X.shape[1]

        # ======================================================================
        # MODE 1: POST-LASSO (Original)
        # ======================================================================
        print(f"  [1/3] Post-LASSO...")
        post_lasso_results = _fit_post_lasso(
            X, y, use_sample_splitting, analysis.random_state, n_features
        )
        analysis.results['post_lasso'][outcome_name] = post_lasso_results
        print(f"       Selected: {post_lasso_results['n_selected']} PCs, "
              f"R²(CV): {post_lasso_results['r2_cv']:.4f}")

        # ======================================================================
        # MODE 2: LASSO ONLY
        # ======================================================================
        print(f"  [2/3] LASSO only...")
        lasso_results = _fit_lasso_only(X, y, analysis.random_state, n_features)
        analysis.results['lasso'][outcome_name] = lasso_results
        n_nonzero_lasso = np.sum(np.abs(lasso_results['coefs_std']) > 1e-10)
        print(f"       Non-zero: {n_nonzero_lasso} PCs, "
              f"R²(CV): {lasso_results['r2_cv']:.4f}")

        # ======================================================================
        # MODE 3: MULTIPLE REGRESSION (All PCs)
        # ======================================================================
        print(f"  [3/3] Multiple Regression (all {n_features} PCs)...")
        mr_results = _fit_multiple_regression(X, y, analysis.random_state, n_features)
        analysis.results['multiple_regression'][outcome_name] = mr_results
        print(f"       Selected: {n_features} PCs (all), "
              f"R²(CV): {mr_results['r2_cv']:.4f}")

    # Store current mode (for backward compatibility with existing code)
    # Default to post-lasso
    analysis.results['debiased_lasso'] = analysis.results['post_lasso']
    analysis.results['current_mode'] = 'post_lasso'

    print("\n" + "="*80)
    print("THREE MODES FIT COMPLETE")
    print("="*80)
    print("\nModes available:")
    print("  • post_lasso: LASSO selection + OLS on selected")
    print("  • lasso: LASSO coefficients only")
    print("  • multiple_regression: OLS on all PCs")

    # Calculate treatment effects for each mode
    for mode_name in ['post_lasso', 'lasso', 'multiple_regression']:
        print(f"\nCalculating treatment effects for {mode_name}...")
        _calculate_treatment_effects_for_mode(analysis, mode_name)

    # Display statistics for post-lasso mode (default)
    analysis.results['debiased_lasso'] = analysis.results['post_lasso']
    analysis.display_global_statistics()
    analysis.display_coefficient_tables()

    return {
        'post_lasso': analysis.results['post_lasso'],
        'lasso': analysis.results['lasso'],
        'multiple_regression': analysis.results['multiple_regression']
    }


def _fit_post_lasso(X: np.ndarray, y: np.ndarray,
                    use_sample_splitting: bool,
                    random_state: int,
                    n_features: int) -> Dict[str, Any]:
    """Fit post-LASSO (LASSO + OLS on selected)."""
    from sklearn.preprocessing import StandardScaler

    n_samples, _ = X.shape
    scaler_y = StandardScaler()
    y_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    if use_sample_splitting and n_samples >= 100:
        # Sample splitting version
        np.random.seed(random_state)
        split_idx = np.random.permutation(n_samples)
        n_half = n_samples // 2

        # Stage 1: LASSO on first half
        lasso = LassoCV(cv=5, random_state=random_state, max_iter=2000)
        lasso.fit(X[split_idx[:n_half]], y_std[split_idx[:n_half]])
        selected = np.where(np.abs(lasso.coef_) > 1e-10)[0]

        # Stage 2: OLS on second half
        if len(selected) > 0:
            X_selected = X[split_idx[n_half:]][:, selected]
            y_estimate = y_std[split_idx[n_half:]]

            ols_model = sm.OLS(y_estimate, sm.add_constant(X_selected))
            ols_result = ols_model.fit(cov_type='HC3')

            coefs = np.zeros(n_features)
            ses = np.zeros(n_features)
            coefs[selected] = ols_result.params[1:]
            ses[selected] = ols_result.bse[1:]

            y_pred = ols_result.predict()
            r2_insample = 1 - np.sum((y_estimate - y_pred)**2) / np.sum((y_estimate - y_estimate.mean())**2)
        else:
            coefs = np.zeros(n_features)
            ses = np.zeros(n_features)
            r2_insample = 0.0
            ols_result = None
            selected = np.array([])
    else:
        # No sample splitting
        lasso = LassoCV(cv=5, random_state=random_state, max_iter=2000)
        lasso.fit(X, y_std)
        selected = np.where(np.abs(lasso.coef_) > 1e-10)[0]

        if len(selected) > 0:
            X_selected = X[:, selected]
            ols_model = sm.OLS(y_std, sm.add_constant(X_selected))
            ols_result = ols_model.fit(cov_type='HC3')

            coefs = np.zeros(n_features)
            ses = np.zeros(n_features)
            coefs[selected] = ols_result.params[1:]
            ses[selected] = ols_result.bse[1:]

            y_pred = ols_result.predict()
            r2_insample = 1 - np.sum((y_std - y_pred)**2) / np.sum((y_std - y_std.mean())**2)
        else:
            coefs = np.zeros(n_features)
            ses = np.zeros(n_features)
            r2_insample = 0.0
            ols_result = None

    # Calculate CV R²
    r2_cv, r2_folds = _calculate_cv_r2_postlasso(X, y_std, selected, random_state)

    # Unstandardize
    coefs_original = coefs * scaler_y.scale_[0]
    ses_original = ses * scaler_y.scale_[0]

    return {
        'coefs': coefs_original,
        'coefs_std': coefs,
        'ses': ses_original,
        'ses_std': ses,
        'selected': selected,
        'n_selected': len(selected),
        'r2_insample': r2_insample,
        'r2_cv': r2_cv,
        'r2_folds': r2_folds,
        'r2_lasso': r2_insample,  # For compatibility with display methods
        'r2_cv_lasso': r2_cv,
        'r2_folds_lasso': r2_folds,
        'lasso_coefs': coefs_original,
        'lasso_coefs_std': coefs,
        'lasso_alpha': lasso.alpha_ if 'lasso' in locals() else None,
        'scaler_y': scaler_y,
        'ols_result': ols_result,
        'mode': 'post_lasso'
    }


def _fit_lasso_only(X: np.ndarray, y: np.ndarray,
                    random_state: int,
                    n_features: int) -> Dict[str, Any]:
    """Fit LASSO only (no post-LASSO OLS step)."""
    from sklearn.preprocessing import StandardScaler

    scaler_y = StandardScaler()
    y_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Fit LASSO
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=2000)
    lasso.fit(X, y_std)

    coefs = lasso.coef_.copy()
    selected = np.where(np.abs(coefs) > 1e-10)[0]

    # For LASSO, we don't have standard errors from OLS
    # Use bootstrap or set to zero
    ses = np.zeros(n_features)

    # Calculate R²
    y_pred = lasso.predict(X)
    r2_insample = 1 - np.sum((y_std - y_pred)**2) / np.sum((y_std - y_std.mean())**2)

    # Calculate CV R²
    r2_cv, r2_folds = _calculate_cv_r2_lasso(X, y_std, random_state)

    # Unstandardize
    coefs_original = coefs * scaler_y.scale_[0]
    ses_original = ses * scaler_y.scale_[0]

    return {
        'coefs': coefs_original,
        'coefs_std': coefs,
        'ses': ses_original,
        'ses_std': ses,
        'selected': selected,
        'n_selected': len(selected),
        'r2_insample': r2_insample,
        'r2_cv': r2_cv,
        'r2_folds': r2_folds,
        'r2_lasso': r2_insample,  # For compatibility with display methods
        'r2_cv_lasso': r2_cv,
        'r2_folds_lasso': r2_folds,
        'lasso_coefs': coefs_original,
        'lasso_coefs_std': coefs,
        'lasso_alpha': lasso.alpha_,
        'scaler_y': scaler_y,
        'ols_result': None,
        'mode': 'lasso'
    }


def _fit_multiple_regression(X: np.ndarray, y: np.ndarray,
                             random_state: int,
                             n_features: int) -> Dict[str, Any]:
    """Fit multiple regression on ALL PCs (no variable selection)."""
    from sklearn.preprocessing import StandardScaler

    scaler_y = StandardScaler()
    y_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Fit OLS on ALL features
    ols_model = sm.OLS(y_std, sm.add_constant(X))
    ols_result = ols_model.fit(cov_type='HC3')

    coefs = ols_result.params[1:]  # Exclude intercept
    ses = ols_result.bse[1:]
    selected = np.arange(n_features)  # All selected

    # Calculate R²
    y_pred = ols_result.predict()[:]
    r2_insample = 1 - np.sum((y_std - y_pred)**2) / np.sum((y_std - y_std.mean())**2)

    # Calculate CV R²
    r2_cv, r2_folds = _calculate_cv_r2_ols_all(X, y_std, random_state)

    # Unstandardize
    coefs_original = coefs * scaler_y.scale_[0]
    ses_original = ses * scaler_y.scale_[0]

    return {
        'coefs': coefs_original,
        'coefs_std': coefs,
        'ses': ses_original,
        'ses_std': ses,
        'selected': selected,
        'n_selected': n_features,
        'r2_insample': r2_insample,
        'r2_cv': r2_cv,
        'r2_folds': r2_folds,
        'r2_lasso': r2_insample,  # For compatibility with display methods
        'r2_cv_lasso': r2_cv,
        'r2_folds_lasso': r2_folds,
        'lasso_coefs': coefs_original,
        'lasso_coefs_std': coefs,
        'lasso_alpha': 0.0,  # No regularization
        'scaler_y': scaler_y,
        'ols_result': ols_result,
        'mode': 'multiple_regression'
    }


def _calculate_cv_r2_postlasso(X, y_std, selected, random_state):
    """Calculate CV R² for post-LASSO."""
    if len(selected) == 0:
        return 0.0, [0.0] * 5

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_std[train_idx], y_std[test_idx]

        X_train_selected = X_train[:, selected]
        X_test_selected = X_test[:, selected]

        ols = sm.OLS(y_train, sm.add_constant(X_train_selected))
        ols_fit = ols.fit()

        y_pred = ols_fit.predict(sm.add_constant(X_test_selected))

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        cv_scores.append(r2)

    return np.mean(cv_scores), cv_scores


def _calculate_cv_r2_lasso(X, y_std, random_state):
    """Calculate CV R² for LASSO."""
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_std[train_idx], y_std[test_idx]

        lasso = LassoCV(cv=5, random_state=random_state, max_iter=2000)
        lasso.fit(X_train, y_train)

        y_pred = lasso.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        cv_scores.append(r2)

    return np.mean(cv_scores), cv_scores


def _calculate_cv_r2_ols_all(X, y_std, random_state):
    """Calculate CV R² for OLS on all features."""
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_std[train_idx], y_std[test_idx]

        ols = sm.OLS(y_train, sm.add_constant(X_train))
        ols_fit = ols.fit()

        y_pred = ols_fit.predict(sm.add_constant(X_test))

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        cv_scores.append(r2)

    return np.mean(cv_scores), cv_scores


def _calculate_treatment_effects_for_mode(analysis, mode_name):
    """Calculate treatment effects for a specific mode."""
    # Temporarily set the mode
    original_debiased_lasso = analysis.results.get('debiased_lasso')

    # Set to current mode
    analysis.results['debiased_lasso'] = analysis.results[mode_name]

    # Calculate treatment effects
    analysis._calculate_treatment_effects()

    # Store mode-specific results
    analysis.results[f'{mode_name}_treatment_effects'] = analysis.results.get('total_effects', {})
    analysis.results[f'{mode_name}_residual_correlations'] = analysis.results.get('residual_correlations', {})
    analysis.results[f'{mode_name}_policy_similarities'] = analysis.results.get('policy_similarities', {})
    analysis.results[f'{mode_name}_mediation_analysis'] = analysis.results.get('mediation_analysis', {})

    # Restore original
    if original_debiased_lasso is not None:
        analysis.results['debiased_lasso'] = original_debiased_lasso


def create_visualization_for_mode(haam_instance, mode_name: str, output_dir: str):
    """
    Create HTML visualization for a specific estimation mode.

    Parameters
    ----------
    haam_instance : HAAM
        HAAM instance with fitted models
    mode_name : str
        One of 'post_lasso', 'lasso', or 'multiple_regression'
    output_dir : str
        Directory to save visualization

    Returns
    -------
    str
        Path to saved HTML file
    """
    import os

    # Store original state
    original_debiased_lasso = haam_instance.analysis.results.get('debiased_lasso')
    original_effects = haam_instance.analysis.results.get('total_effects')
    original_residuals = haam_instance.analysis.results.get('residual_correlations')
    original_policy = haam_instance.analysis.results.get('policy_similarities')
    original_mediation = haam_instance.analysis.results.get('mediation_analysis')

    # Set to requested mode
    haam_instance.analysis.results['debiased_lasso'] = haam_instance.analysis.results[mode_name]
    haam_instance.analysis.results['total_effects'] = haam_instance.analysis.results.get(f'{mode_name}_treatment_effects', {})
    haam_instance.analysis.results['residual_correlations'] = haam_instance.analysis.results.get(f'{mode_name}_residual_correlations', {})
    haam_instance.analysis.results['policy_similarities'] = haam_instance.analysis.results.get(f'{mode_name}_policy_similarities', {})
    haam_instance.analysis.results['mediation_analysis'] = haam_instance.analysis.results.get(f'{mode_name}_mediation_analysis', {})

    # Update visualizer results
    haam_instance.visualizer.results = haam_instance.analysis.results

    # Create output filename
    mode_labels = {
        'post_lasso': 'post_lasso',
        'lasso': 'lasso',
        'multiple_regression': 'multiple_regression'
    }
    output_file = os.path.join(output_dir, f'haam_main_visualization_{mode_labels[mode_name]}.html')

    # Create visualization
    top_pcs = haam_instance.analysis.get_top_pcs(n_top=9, ranking_method='HU')
    viz_path = haam_instance.visualizer.create_main_visualization(top_pcs, output_file, pc_names=None)

    # Restore original state
    if original_debiased_lasso is not None:
        haam_instance.analysis.results['debiased_lasso'] = original_debiased_lasso
    if original_effects is not None:
        haam_instance.analysis.results['total_effects'] = original_effects
    if original_residuals is not None:
        haam_instance.analysis.results['residual_correlations'] = original_residuals
    if original_policy is not None:
        haam_instance.analysis.results['policy_similarities'] = original_policy
    if original_mediation is not None:
        haam_instance.analysis.results['mediation_analysis'] = original_mediation
    haam_instance.visualizer.results = haam_instance.analysis.results

    print(f"✓ {mode_labels[mode_name]} visualization saved: {output_file}")

    return viz_path
