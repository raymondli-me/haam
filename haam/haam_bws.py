#!/usr/bin/env python3
"""
HAAM with BWS (Best-Worst Scaling) Support
==========================================
A specialized HAAM class for analyzing interpretable BWS features
while maintaining compatibility with standard HAAM visualizations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
from scipy import stats


class HAAMwithBWS:
    """
    HAAM analysis for Best-Worst Scaling (BWS) or other interpretable features.
    
    This class provides HAAM analysis capabilities for pre-computed interpretable
    features, bypassing PCA while maintaining all statistical analysis and
    visualization capabilities of standard HAAM.
    
    Parameters
    ----------
    criterion : array-like
        Ground truth variable (e.g., social class)
    ai_judgment : array-like
        AI predictions/ratings
    human_judgment : array-like
        Human ratings
    bws_features : array-like
        Pre-computed interpretable features (e.g., BWS scores)
        Shape: (n_samples, n_features)
    feature_names : list, optional
        Names for each BWS feature for interpretability
    texts : list of str, optional
        Original texts (for supplementary analysis)
    standardize : bool, default=True
        Whether to standardize features before analysis
    sample_split_post_lasso : bool, default=True
        Whether to use sample splitting for post-LASSO inference
        True: Conservative inference with valid p-values
        False: Maximum power but potential selection bias
    auto_run : bool, default=True
        Whether to automatically run the full pipeline
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self,
                 criterion: Union[np.ndarray, pd.Series, list],
                 ai_judgment: Union[np.ndarray, pd.Series, list], 
                 human_judgment: Union[np.ndarray, pd.Series, list],
                 bws_features: Union[np.ndarray, pd.DataFrame],
                 feature_names: Optional[List[str]] = None,
                 texts: Optional[List[str]] = None,
                 standardize: bool = True,
                 sample_split_post_lasso: bool = True,
                 auto_run: bool = True,
                 random_state: int = 42):
        
        # Store parameters
        self.criterion = self._ensure_array(criterion)
        self.ai_judgment = self._ensure_array(ai_judgment)
        self.human_judgment = self._ensure_array(human_judgment)
        self.bws_features = self._ensure_array(bws_features)
        self.texts = texts
        self.standardize = standardize
        self.sample_split_post_lasso = sample_split_post_lasso
        self.random_state = random_state
        
        # Handle feature names
        if feature_names is None:
            self.feature_names = [f'BWS_{i+1}' for i in range(self.bws_features.shape[1])]
        else:
            self.feature_names = feature_names
            
        # Validate dimensions
        self._validate_inputs()
        
        # Initialize results
        self.results = {}
        
        # Run analysis if requested
        if auto_run:
            self.run_analysis()
    
    def _ensure_array(self, data):
        """Convert input to numpy array."""
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, pd.DataFrame):
            return data.values
        return data
    
    def _validate_inputs(self):
        """Validate input dimensions and data quality."""
        n_samples = len(self.criterion)
        
        # Check dimensions
        assert len(self.ai_judgment) == n_samples, "AI judgment length mismatch"
        assert len(self.human_judgment) == n_samples, "Human judgment length mismatch"
        assert len(self.bws_features) == n_samples, "BWS features length mismatch"
        
        # Check for sufficient non-NaN data
        human_valid = (~np.isnan(self.human_judgment)).sum()
        if human_valid < n_samples * 0.1:
            warnings.warn(f"Only {human_valid} human ratings available ({human_valid/n_samples*100:.1f}%)")
        
        print(f"Data validated: {n_samples} samples, {self.bws_features.shape[1]} BWS features")
    
    def run_analysis(self):
        """Run the complete HAAM analysis pipeline with BWS features."""
        print("\n" + "="*60)
        print("HAAM ANALYSIS WITH BWS FEATURES")
        print("="*60)
        
        # Standardize features if requested
        if self.standardize:
            print("\n1. Standardizing BWS features...")
            scaler = StandardScaler()
            features_std = scaler.fit_transform(self.bws_features)
        else:
            features_std = self.bws_features
        
        # Run debiased LASSO
        print(f"\n2. Running debiased LASSO (sample_split={self.sample_split_post_lasso})...")
        self.results['debiased_lasso'] = self._fit_debiased_lasso_all(features_std)
        
        # Calculate comprehensive metrics
        print("\n3. Calculating comprehensive metrics...")
        self._calculate_metrics(features_std)
        
        # Create model summary
        self._create_model_summary()
        
        print("\n✓ Analysis complete!")
        self._print_summary()
    
    def _fit_debiased_lasso_all(self, features):
        """Fit debiased LASSO for all outcomes."""
        results = {}
        
        for outcome_name, outcome_values in [
            ('X', self.criterion),
            ('AI', self.ai_judgment),
            ('HU', self.human_judgment)
        ]:
            print(f"  Fitting {outcome_name} model...")
            results[outcome_name] = self._fit_single_debiased_lasso(
                features, outcome_values, outcome_name
            )
        
        return results
    
    def _fit_single_debiased_lasso(self, X, y, outcome_name):
        """Fit single debiased LASSO model."""
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 50:
            print(f"    Insufficient data for {outcome_name} (n={mask.sum()})")
            return None
        
        X_clean = X[mask]
        y_clean = y[mask]
        n_samples = len(y_clean)
        
        # Standardize y
        y_std = StandardScaler().fit_transform(y_clean.reshape(-1, 1)).ravel()
        
        if self.sample_split_post_lasso and n_samples >= 100:
            # Sample splitting approach
            n_half = n_samples // 2
            np.random.seed(self.random_state)
            indices = np.random.permutation(n_samples)
            
            # First half: variable selection
            X_split1 = X_clean[indices[:n_half]]
            y_split1 = y_std[indices[:n_half]]
            
            lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=self.random_state)
            lasso_cv.fit(X_split1, y_split1)
            selected = np.where(np.abs(lasso_cv.coef_) > 1e-10)[0]
            
            if len(selected) == 0:
                print(f"    No features selected for {outcome_name}")
                return {'selected': [], 'coefficients': np.zeros(X.shape[1]), 'r2_cv': 0}
            
            # Second half: inference
            X_split2 = X_clean[indices[n_half:]][:, selected]
            y_split2 = y_std[indices[n_half:]]
            
            X_with_const = sm.add_constant(X_split2)
            ols_model = sm.OLS(y_split2, X_with_const).fit(cov_type='HC3')
            
            # Store results
            coefficients = np.zeros(X.shape[1])
            coefficients[selected] = ols_model.params[1:]
            
            return {
                'selected': selected,
                'coefficients': coefficients,
                'std_errors': ols_model.bse[1:] if len(selected) > 0 else [],
                'p_values': ols_model.pvalues[1:] if len(selected) > 0 else [],
                'r2': ols_model.rsquared,
                'r2_adj': ols_model.rsquared_adj,
                'n_samples': n_samples,
                'alpha': lasso_cv.alpha_
            }
        else:
            # Full sample approach
            lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=self.random_state)
            lasso_cv.fit(X_clean, y_std)
            selected = np.where(np.abs(lasso_cv.coef_) > 1e-10)[0]
            
            if len(selected) == 0:
                print(f"    No features selected for {outcome_name}")
                return {'selected': [], 'coefficients': np.zeros(X.shape[1]), 'r2_cv': 0}
            
            # Post-LASSO OLS
            X_selected = X_clean[:, selected]
            X_with_const = sm.add_constant(X_selected)
            ols_model = sm.OLS(y_std, X_with_const).fit(cov_type='HC3')
            
            # Store results
            coefficients = np.zeros(X.shape[1])
            coefficients[selected] = ols_model.params[1:]
            
            # Calculate CV R²
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(lasso_cv, X_clean, y_std, cv=5, 
                                       scoring='r2')
            
            return {
                'selected': selected,
                'coefficients': coefficients,
                'std_errors': ols_model.bse[1:] if len(selected) > 0 else [],
                'p_values': ols_model.pvalues[1:] if len(selected) > 0 else [],
                'r2': ols_model.rsquared,
                'r2_adj': ols_model.rsquared_adj,
                'r2_cv': np.mean(cv_scores),
                'n_samples': n_samples,
                'alpha': lasso_cv.alpha_
            }
    
    def _calculate_metrics(self, features):
        """Calculate comprehensive HAAM metrics."""
        # Store coefficients
        coef_data = {'Feature': self.feature_names}
        
        for outcome in ['X', 'AI', 'HU']:
            if self.results['debiased_lasso'].get(outcome):
                res = self.results['debiased_lasso'][outcome]
                coef_data[f'{outcome}_coef'] = res['coefficients']
            else:
                coef_data[f'{outcome}_coef'] = np.zeros(len(self.feature_names))
        
        self.results['coefficients'] = pd.DataFrame(coef_data)
        
        # Calculate total effects (simplified)
        self.results['total_effects'] = self._calculate_total_effects()
    
    def _calculate_total_effects(self):
        """Calculate total effects between outcomes."""
        effects = {}
        
        # X → AI
        mask = ~np.isnan(self.criterion) & ~np.isnan(self.ai_judgment)
        if mask.sum() > 50:
            corr = np.corrcoef(self.criterion[mask], self.ai_judgment[mask])[0, 1]
            effects['X_AI'] = {'coefficient': corr}
        
        # X → HU
        mask = ~np.isnan(self.criterion) & ~np.isnan(self.human_judgment)
        if mask.sum() > 50:
            corr = np.corrcoef(self.criterion[mask], self.human_judgment[mask])[0, 1]
            effects['X_HU'] = {'coefficient': corr}
        
        return effects
    
    def _create_model_summary(self):
        """Create model summary DataFrame."""
        summary_data = []
        
        for outcome in ['X', 'AI', 'HU']:
            if self.results['debiased_lasso'].get(outcome):
                res = self.results['debiased_lasso'][outcome]
                summary_data.append({
                    'Outcome': outcome,
                    'N': res.get('n_samples', 0),
                    'N_selected': len(res.get('selected', [])),
                    'R2': res.get('r2', 0),
                    'R2_adj': res.get('r2_adj', 0),
                    'R2_cv': res.get('r2_cv', 0)
                })
        
        self.results['model_summary'] = pd.DataFrame(summary_data)
    
    def _print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        if 'model_summary' in self.results:
            print("\nModel Performance:")
            for _, row in self.results['model_summary'].iterrows():
                print(f"  {row['Outcome']}: R²(CV)={row['R2_cv']:.3f}, Selected={row['N_selected']}")
        
        if 'total_effects' in self.results:
            print("\nTotal Effects:")
            for path, effect in self.results['total_effects'].items():
                print(f"  {path}: r={effect['coefficient']:.3f}")
    
    def export_results(self, output_dir: str = '.'):
        """Export results to files."""
        import os
        import json
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save coefficients
        if 'coefficients' in self.results:
            path = os.path.join(output_dir, 'bws_coefficients.csv')
            self.results['coefficients'].to_csv(path, index=False)
            print(f"✓ Saved: {path}")
        
        # Save model summary
        if 'model_summary' in self.results:
            path = os.path.join(output_dir, 'bws_model_summary.csv')
            self.results['model_summary'].to_csv(path, index=False)
            print(f"✓ Saved: {path}")
        
        # Save metrics
        metrics = {
            'analysis_type': 'HAAM with BWS',
            'n_features': self.bws_features.shape[1],
            'sample_split_used': self.sample_split_post_lasso,
            'timestamp': datetime.now().isoformat()
        }
        
        path = os.path.join(output_dir, 'bws_metrics.json')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"✓ Saved: {path}")