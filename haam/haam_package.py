"""
HAAM: Human-AI Accuracy Model Analysis Package
==============================================

A lightweight package for analyzing human-AI accuracy models with 
sample-split post-lasso regression and interactive visualizations.

Author: HAAM Development Team
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer
import statsmodels.api as sm
from scipy import stats
import umap
import warnings
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class HAAMAnalysis:
    """
    Main class for Human-AI Accuracy Model analysis.
    
    This class performs sample-split post-lasso regression analysis
    and generates various visualizations for understanding the relationships
    between human judgments, AI judgments, and a criterion variable.
    """
    
    def __init__(self, 
                 criterion: np.ndarray,
                 ai_judgment: np.ndarray, 
                 human_judgment: np.ndarray,
                 embeddings: Optional[np.ndarray] = None,
                 texts: Optional[List[str]] = None,
                 n_components: int = 200,
                 random_state: int = 42,
                 standardize: bool = False):
        """
        Initialize HAAM Analysis.
        
        Parameters
        ----------
        criterion : np.ndarray
            Criterion variable (e.g., social class)
        ai_judgment : np.ndarray
            AI predictions/ratings
        human_judgment : np.ndarray
            Human ratings
        embeddings : np.ndarray, optional
            Pre-computed embeddings. If None, will be generated from texts
        texts : List[str], optional
            Text data for generating embeddings if not provided
        n_components : int, default=200
            Number of PCA components to extract
        random_state : int, default=42
            Random state for reproducibility
        standardize : bool, default=False
            Whether to standardize X and Y variables for both total effects and DML calculations.
            When True, all coefficients will be in standardized units.
        """
        self.criterion = self._validate_input(criterion, "criterion")
        self.ai_judgment = self._validate_input(ai_judgment, "ai_judgment")
        self.human_judgment = self._validate_input(human_judgment, "human_judgment")
        self.n_components = n_components
        self.random_state = random_state
        self.standardize = standardize
        
        # Handle embeddings
        if embeddings is None and texts is None:
            raise ValueError("Either embeddings or texts must be provided")
        
        if embeddings is None:
            print("Generating embeddings from texts...")
            self.embeddings = self.generate_embeddings(texts)
        else:
            self.embeddings = embeddings
            
        # Validate dimensions
        n_samples = len(self.criterion)
        if not all(len(x) == n_samples for x in [self.ai_judgment, self.human_judgment, self.embeddings]):
            raise ValueError("All inputs must have the same number of samples")
            
        # Initialize results storage
        self.results = {
            'pca': None,
            'pca_features': None,
            'debiased_lasso': {},
            'topics': {},
            'visualizations': {},
            'exports': {}
        }
        
        # Run PCA
        self._perform_pca()
        
    @staticmethod
    def _validate_input(data: Union[list, np.ndarray], name: str) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{name} must be a numpy array or list")
        return data.flatten()
    
    @staticmethod
    def generate_embeddings(texts: List[str], 
                          model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                          batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings using MiniLM model.
        
        Parameters
        ----------
        texts : List[str]
            List of text documents
        model_name : str
            Name of the sentence transformer model
        batch_size : int
            Batch size for encoding
            
        Returns
        -------
        np.ndarray
            Embedding matrix (n_samples, embedding_dim)
        """
        print(f"Loading {model_name}...")
        model = SentenceTransformer(model_name)
        model.max_seq_length = 512
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def _perform_pca(self):
        """Perform PCA on embeddings."""
        print(f"\nPerforming PCA with {self.n_components} components...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # PCA
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        pca_features_raw = pca.fit_transform(embeddings_scaled)
        
        # Standardize PCA features
        pca_features = StandardScaler().fit_transform(pca_features_raw)
        
        # Store results
        self.results['pca'] = pca
        self.results['pca_features'] = pca_features
        self.results['variance_explained'] = pca.explained_variance_ratio_
        
        print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
        
    def fit_debiased_lasso(self, 
                          use_sample_splitting: bool = True,
                          alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Fit debiased lasso models for all outcomes.
        
        Parameters
        ----------
        use_sample_splitting : bool, default=True
            Whether to use sample splitting for valid inference
        alpha : float, optional
            Regularization parameter. If None, uses CV
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all results
        """
        outcomes = {
            'SC': self.criterion,
            'AI': self.ai_judgment,
            'HU': self.human_judgment
        }
        
        print("\nFitting debiased lasso models...")
        print("=" * 60)
        
        for outcome_name, outcome_values in outcomes.items():
            print(f"\n{outcome_name} Model:")
            print("-" * 40)
            
            # Remove NaN values
            mask = ~np.isnan(outcome_values)
            if mask.sum() < 50:
                print(f"  Insufficient data (n={mask.sum()})")
                continue
                
            X = self.results['pca_features'][mask]
            y = outcome_values[mask]
            
            # Fit model
            results = self._fit_single_debiased_lasso(X, y, use_sample_splitting)
            
            # Store results
            self.results['debiased_lasso'][outcome_name] = results
            
            # Print summary
            print(f"  Variables selected: {results['n_selected']}")
            print(f"  R² (in-sample): {results['r2_insample']:.4f}")
            print(f"  R² (CV): {results['r2_cv']:.4f}")
            
        # Calculate treatment effects
        self._calculate_treatment_effects()
        
        return self.results['debiased_lasso']
    
    def _fit_single_debiased_lasso(self, X: np.ndarray, y: np.ndarray, 
                                   use_sample_splitting: bool = True) -> Dict[str, Any]:
        """Fit debiased lasso for a single outcome."""
        n_samples, n_features = X.shape
        
        # Standardize y
        scaler_y = StandardScaler()
        y_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        if use_sample_splitting and n_samples >= 100:
            # Sample splitting
            np.random.seed(self.random_state)
            split_idx = np.random.permutation(n_samples)
            n_half = n_samples // 2
            
            # Stage 1: Variable selection on first half
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=2000)
            lasso.fit(X[split_idx[:n_half]], y_std[split_idx[:n_half]])
            selected = np.where(np.abs(lasso.coef_) > 1e-10)[0]
            
            # Store LASSO coefficients for global metrics
            lasso_coefs = lasso.coef_.copy()
            
            # Stage 2: OLS on second half
            if len(selected) > 0:
                X_selected = X[split_idx[n_half:]][:, selected]
                y_estimate = y_std[split_idx[n_half:]]
                
                ols_model = sm.OLS(y_estimate, sm.add_constant(X_selected))
                ols_result = ols_model.fit(cov_type='HC3')
                
                # Map back coefficients
                coefs = np.zeros(n_features)
                ses = np.zeros(n_features)
                coefs[selected] = ols_result.params[1:]
                ses[selected] = ols_result.bse[1:]
                
                # Calculate R²
                y_pred = ols_result.predict()
                r2_insample = 1 - np.sum((y_estimate - y_pred)**2) / np.sum((y_estimate - y_estimate.mean())**2)
                
            else:
                coefs = np.zeros(n_features)
                ses = np.zeros(n_features)
                r2_insample = 0.0
                ols_result = None
                
        else:
            # No sample splitting
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=2000)
            lasso.fit(X, y_std)
            selected = np.where(np.abs(lasso.coef_) > 1e-10)[0]
            
            # Store LASSO coefficients for global metrics
            lasso_coefs = lasso.coef_.copy()
            
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
        
        # Calculate CV R² for LASSO (for global metrics)
        if 'lasso_coefs' in locals():
            # Calculate R² using LASSO predictions
            y_pred_lasso = X @ lasso_coefs
            r2_lasso = 1 - np.sum((y_std - y_pred_lasso)**2) / np.sum((y_std - y_std.mean())**2)
            r2_cv_lasso = self._calculate_cv_r2(X, y_std, selected, lasso_coefs)
        else:
            lasso_coefs = coefs  # Fallback if not using sample splitting
            r2_lasso = r2_insample
            r2_cv_lasso = r2_cv
            
        # Calculate CV R² for post-LASSO OLS (for display)
        r2_cv = self._calculate_cv_r2(X, y_std, selected, coefs)
        
        # Unstandardize coefficients
        coefs_original = coefs * scaler_y.scale_[0]
        ses_original = ses * scaler_y.scale_[0]
        lasso_coefs_original = lasso_coefs * scaler_y.scale_[0]
        
        return {
            'coefs': coefs_original,  # Post-LASSO OLS coefficients (for display)
            'coefs_std': coefs,  # Post-LASSO OLS standardized (for display)
            'lasso_coefs': lasso_coefs_original,  # LASSO coefficients (for global metrics)
            'lasso_coefs_std': lasso_coefs,  # LASSO standardized (for global metrics)
            'ses': ses_original,
            'ses_std': ses,
            'selected': selected,
            'n_selected': len(selected),
            'r2_insample': r2_insample,  # Post-LASSO OLS R²
            'r2_cv': r2_cv,  # Post-LASSO OLS CV R²
            'r2_lasso': r2_lasso,  # LASSO R²
            'r2_cv_lasso': r2_cv_lasso,  # LASSO CV R²
            'lasso_alpha': lasso.alpha_ if 'lasso' in locals() else None,
            'scaler_y': scaler_y,
            'ols_result': ols_result
        }
    
    def _calculate_cv_r2(self, X: np.ndarray, y: np.ndarray, 
                        selected: np.ndarray, coefs: np.ndarray) -> float:
        """Calculate cross-validated R²."""
        if len(selected) == 0:
            return 0.0
            
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Refit on training data
            if len(selected) > 0:
                X_train_selected = X_train[:, selected]
                X_test_selected = X_test[:, selected]
                
                ols = sm.OLS(y_train, sm.add_constant(X_train_selected))
                ols_fit = ols.fit()
                
                y_pred = ols_fit.predict(sm.add_constant(X_test_selected))
                
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                cv_scores.append(r2)
        
        return np.mean(cv_scores)
    
    def _calculate_treatment_effects(self):
        """Calculate comprehensive metrics including DML treatment effects, residual correlations, and policy similarities."""
        print("\nCalculating comprehensive metrics...")
        
        # Initialize results storage
        self.results['total_effects'] = {}
        self.results['residual_correlations'] = {}
        self.results['policy_similarities'] = {}
        self.results['mediation_analysis'] = {}
        
        # Calculate total effects (DML coefficients)
        self._calculate_total_effects_dml()
        
        # Calculate residual correlations
        self._calculate_residual_correlations()
        
        # Calculate policy similarities
        self._calculate_policy_similarities()
        
        # Calculate mediation analysis (PoMA)
        self._calculate_mediation_analysis()
    
    def _calculate_total_effects_dml(self):
        """Calculate Double Machine Learning (DML) total effects with proper residual-on-residual regression."""
        # Get predictions from each model
        predictions = {}
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                X = self.results['pca_features']
                coefs = self.results['debiased_lasso'][outcome]['coefs_std']
                predictions[outcome] = X @ coefs
        
        # Calculate total effects for key paths
        # Y -> AI (direct effect of criterion on AI)
        if 'SC' in self.results['debiased_lasso'] and 'AI' in self.results['debiased_lasso']:
            mask = ~(np.isnan(self.criterion) | np.isnan(self.ai_judgment))
            if mask.sum() > 50:
                # Prepare data for total effect
                X_data = self.criterion[mask]
                y_data = self.ai_judgment[mask]
                
                # Standardize if requested
                if self.standardize:
                    scaler_X = StandardScaler()
                    scaler_y = StandardScaler()
                    X_data = scaler_X.fit_transform(X_data.reshape(-1, 1)).ravel()
                    y_data = scaler_y.fit_transform(y_data.reshape(-1, 1)).ravel()
                
                # Simple OLS for total effect
                X = sm.add_constant(X_data)
                y = y_data
                model = sm.OLS(y, X).fit()
                
                # Calculate beta_check using DML residual-on-residual regression
                check_beta = self._calculate_dml_check_beta(
                    self.criterion[mask], 
                    self.ai_judgment[mask], 
                    self.results['pca_features'][mask],
                    self.results['debiased_lasso']['SC']['selected'],
                    self.results['debiased_lasso']['AI']['selected']
                )
                
                self.results['total_effects']['Y_AI'] = {
                    'coefficient': model.params[1],
                    'se': model.bse[1],
                    'check_beta': check_beta
                }
        
        # Y -> HU (direct effect of criterion on human)
        if 'SC' in self.results['debiased_lasso'] and 'HU' in self.results['debiased_lasso']:
            mask = ~(np.isnan(self.criterion) | np.isnan(self.human_judgment))
            if mask.sum() > 50:
                # Prepare data for total effect
                X_data = self.criterion[mask]
                y_data = self.human_judgment[mask]
                
                # Standardize if requested
                if self.standardize:
                    scaler_X = StandardScaler()
                    scaler_y = StandardScaler()
                    X_data = scaler_X.fit_transform(X_data.reshape(-1, 1)).ravel()
                    y_data = scaler_y.fit_transform(y_data.reshape(-1, 1)).ravel()
                
                # Simple OLS for total effect
                X = sm.add_constant(X_data)
                y = y_data
                model = sm.OLS(y, X).fit()
                
                # Calculate beta_check using DML
                check_beta = self._calculate_dml_check_beta(
                    self.criterion[mask], 
                    self.human_judgment[mask], 
                    self.results['pca_features'][mask],
                    self.results['debiased_lasso']['SC']['selected'],
                    self.results['debiased_lasso']['HU']['selected']
                )
                
                self.results['total_effects']['Y_HU'] = {
                    'coefficient': model.params[1],
                    'se': model.bse[1],
                    'check_beta': check_beta
                }
        
        # HU -> AI (effect of human on AI)
        if 'HU' in self.results['debiased_lasso'] and 'AI' in self.results['debiased_lasso']:
            mask = ~(np.isnan(self.human_judgment) | np.isnan(self.ai_judgment))
            if mask.sum() > 50:
                # Prepare data for total effect
                X_data = self.human_judgment[mask]
                y_data = self.ai_judgment[mask]
                
                # Standardize if requested
                if self.standardize:
                    scaler_X = StandardScaler()
                    scaler_y = StandardScaler()
                    X_data = scaler_X.fit_transform(X_data.reshape(-1, 1)).ravel()
                    y_data = scaler_y.fit_transform(y_data.reshape(-1, 1)).ravel()
                
                # Simple OLS for total effect
                X = sm.add_constant(X_data)
                y = y_data
                model = sm.OLS(y, X).fit()
                
                # Calculate beta_check using DML
                check_beta = self._calculate_dml_check_beta(
                    self.human_judgment[mask], 
                    self.ai_judgment[mask], 
                    self.results['pca_features'][mask],
                    self.results['debiased_lasso']['HU']['selected'],
                    self.results['debiased_lasso']['AI']['selected']
                )
                
                self.results['total_effects']['HU_AI'] = {
                    'coefficient': model.params[1],
                    'se': model.bse[1],
                    'check_beta': check_beta
                }
    
    def _calculate_dml_check_beta(self, X_var, Y_var, pca_features, X_selected, Y_selected):
        """
        Calculate DML check beta using residual-on-residual regression with cross-validation.
        
        This implements the proper DML procedure:
        1. Use 5-fold cross-validation
        2. For each fold:
           - Train models on training folds
           - Get residuals on test fold
           - Regress residuals on residuals
        3. Average the coefficients across folds
        
        Parameters
        ----------
        X_var : np.ndarray
            Treatment variable (e.g., criterion for Y->AI)
        Y_var : np.ndarray
            Outcome variable (e.g., AI judgment for Y->AI)
        pca_features : np.ndarray
            PCA features for predictions
        X_selected : np.ndarray
            Selected features for X model
        Y_selected : np.ndarray
            Selected features for Y model
            
        Returns
        -------
        float
            DML check beta coefficient
        """
        from sklearn.model_selection import KFold
        
        n_samples = len(X_var)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        check_betas = []
        
        for train_idx, test_idx in kf.split(X_var):
            # Split data
            X_train, X_test = X_var[train_idx], X_var[test_idx]
            Y_train, Y_test = Y_var[train_idx], Y_var[test_idx]
            pca_train, pca_test = pca_features[train_idx], pca_features[test_idx]
            
            # Standardize if requested
            if self.standardize:
                scaler_X = StandardScaler()
                scaler_Y = StandardScaler()
                
                X_train_std = scaler_X.fit_transform(X_train.reshape(-1, 1)).ravel()
                X_test_std = scaler_X.transform(X_test.reshape(-1, 1)).ravel()
                Y_train_std = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()
                Y_test_std = scaler_Y.transform(Y_test.reshape(-1, 1)).ravel()
            else:
                X_train_std = X_train
                X_test_std = X_test
                Y_train_std = Y_train
                Y_test_std = Y_test
            
            # Get residuals for X (treatment)
            if len(X_selected) > 0:
                X_features_train = pca_train[:, X_selected]
                X_features_test = pca_test[:, X_selected]
                
                # Fit model on train
                model_X = sm.OLS(X_train_std, sm.add_constant(X_features_train)).fit()
                # Predict on test
                X_pred_test = model_X.predict(sm.add_constant(X_features_test))
                X_resid_test = X_test_std - X_pred_test
            else:
                X_resid_test = X_test_std
            
            # Get residuals for Y (outcome)
            if len(Y_selected) > 0:
                Y_features_train = pca_train[:, Y_selected]
                Y_features_test = pca_test[:, Y_selected]
                
                # Fit model on train
                model_Y = sm.OLS(Y_train_std, sm.add_constant(Y_features_train)).fit()
                # Predict on test
                Y_pred_test = model_Y.predict(sm.add_constant(Y_features_test))
                Y_resid_test = Y_test_std - Y_pred_test
            else:
                Y_resid_test = Y_test_std
            
            # Residual-on-residual regression
            if len(X_resid_test) > 10:  # Need enough test samples
                model_resid = sm.OLS(Y_resid_test, sm.add_constant(X_resid_test)).fit()
                check_betas.append(model_resid.params[1])
        
        # Return average check beta across folds
        return np.mean(check_betas) if check_betas else 0.0
    
    def _calculate_residual_correlations(self):
        """
        Calculate residual correlations (C's) between residuals after controlling for PCs.
        
        These are the correlations between the residuals of each variable after
        regressing on the selected PCs (mediators). This is similar to DML but
        just computes the correlation between residuals.
        """
        # Get the actual values for each outcome
        outcome_values = {
            'SC': self.criterion,
            'AI': self.ai_judgment,
            'HU': self.human_judgment
        }
        
        # Standardize each outcome variable (ignoring NaNs)
        standardized_values = {}
        for outcome, values in outcome_values.items():
            mask = ~np.isnan(values)
            if mask.sum() > 0:
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                standardized = (values - mean_val) / std_val if std_val > 0 else values - mean_val
                standardized_values[outcome] = standardized
            else:
                standardized_values[outcome] = values
        
        # Get predictions from PC models (standardized scale)
        predictions = {}
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                selected = self.results['debiased_lasso'][outcome]['selected']
                if len(selected) > 0:
                    X = self.results['pca_features'][:, selected]
                    # Get the fitted values on standardized scale
                    coefs = self.results['debiased_lasso'][outcome]['coefs_std'][selected]
                    predictions[outcome] = X @ coefs
                else:
                    predictions[outcome] = np.zeros(len(self.criterion))
        
        # C(AI,HU) - correlation between AI and HU residuals after controlling for their PC predictions
        mask = ~(np.isnan(self.ai_judgment) | np.isnan(self.human_judgment))
        if mask.sum() > 50 and 'AI' in predictions and 'HU' in predictions:
            # Get residuals after controlling for PCs (on standardized scale)
            resid_ai = standardized_values['AI'][mask] - predictions['AI'][mask]
            resid_hu = standardized_values['HU'][mask] - predictions['HU'][mask]
            
            # C(AI,HU) = corr(e_AI, e_HU)
            if np.std(resid_ai) > 0 and np.std(resid_hu) > 0:
                self.results['residual_correlations']['AI_HU'] = np.corrcoef(resid_ai, resid_hu)[0, 1]
            else:
                self.results['residual_correlations']['AI_HU'] = 0.0
        else:
            self.results['residual_correlations']['AI_HU'] = 0.0
        
        # C(Y,AI) - correlation between Y and AI residuals after controlling for their PC predictions
        mask = ~(np.isnan(self.criterion) | np.isnan(self.ai_judgment))
        if mask.sum() > 50 and 'SC' in predictions and 'AI' in predictions:
            # Get residuals after controlling for PCs (on standardized scale)
            resid_y = standardized_values['SC'][mask] - predictions['SC'][mask]
            resid_ai = standardized_values['AI'][mask] - predictions['AI'][mask]
            
            # C(Y,AI) = corr(e_Y, e_AI)
            if np.std(resid_y) > 0 and np.std(resid_ai) > 0:
                self.results['residual_correlations']['Y_AI'] = np.corrcoef(resid_y, resid_ai)[0, 1]
            else:
                self.results['residual_correlations']['Y_AI'] = 0.0
        else:
            self.results['residual_correlations']['Y_AI'] = 0.0
        
        # C(Y,HU) - correlation between Y and HU residuals after controlling for their PC predictions
        mask = ~(np.isnan(self.criterion) | np.isnan(self.human_judgment))
        if mask.sum() > 50 and 'SC' in predictions and 'HU' in predictions:
            # Get residuals after controlling for PCs (on standardized scale)
            resid_y = standardized_values['SC'][mask] - predictions['SC'][mask]
            resid_hu = standardized_values['HU'][mask] - predictions['HU'][mask]
            
            # C(Y,HU) = corr(e_Y, e_HU)
            if np.std(resid_y) > 0 and np.std(resid_hu) > 0:
                self.results['residual_correlations']['Y_HU'] = np.corrcoef(resid_y, resid_hu)[0, 1]
            else:
                self.results['residual_correlations']['Y_HU'] = 0.0
        else:
            self.results['residual_correlations']['Y_HU'] = 0.0
    
    def _calculate_policy_similarities(self):
        """Calculate correlations between model predictions (policy similarities)."""
        # Get predictions from each model using LASSO coefficients
        predictions = {}
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                X = self.results['pca_features']
                # Use LASSO coefficients for global metrics
                coefs = self.results['debiased_lasso'][outcome]['lasso_coefs_std']
                predictions[outcome] = X @ coefs
        
        # Calculate pairwise correlations between predictions
        if 'SC' in predictions and 'AI' in predictions:
            mask = ~(np.isnan(predictions['SC']) | np.isnan(predictions['AI']))
            if mask.sum() > 50:
                self.results['policy_similarities']['Y_AI'] = np.corrcoef(
                    predictions['SC'][mask], predictions['AI'][mask]
                )[0, 1]
            else:
                self.results['policy_similarities']['Y_AI'] = 0.0
        
        if 'SC' in predictions and 'HU' in predictions:
            mask = ~(np.isnan(predictions['SC']) | np.isnan(predictions['HU']))
            if mask.sum() > 50:
                self.results['policy_similarities']['Y_HU'] = np.corrcoef(
                    predictions['SC'][mask], predictions['HU'][mask]
                )[0, 1]
            else:
                self.results['policy_similarities']['Y_HU'] = 0.0
        
        if 'AI' in predictions and 'HU' in predictions:
            mask = ~(np.isnan(predictions['AI']) | np.isnan(predictions['HU']))
            if mask.sum() > 50:
                self.results['policy_similarities']['AI_HU'] = np.corrcoef(
                    predictions['AI'][mask], predictions['HU'][mask]
                )[0, 1]
            else:
                self.results['policy_similarities']['AI_HU'] = 0.0
    
    def _calculate_mediation_analysis(self):
        """
        Calculate proportion of maximum achievable (PoMA) mediation using DML.
        
        PoMA = 1 - (β_check / β_total)
        
        Where:
        - β_total = total effect (from simple regression)
        - β_check = DML check beta (direct effect after controlling for mediators)
        """
        # For Y -> AI path
        if 'Y_AI' in self.results['total_effects']:
            total_effect = self.results['total_effects']['Y_AI']['coefficient']
            check_beta = self.results['total_effects']['Y_AI']['check_beta']
            
            # PoMA = 1 - (direct effect / total effect)
            # The indirect effect is what's mediated through PCs
            direct_effect = check_beta  # DML check beta is the direct effect
            indirect_effect = total_effect - direct_effect
            
            self.results['mediation_analysis']['AI'] = {
                'total_effect': total_effect,
                'indirect_effect': indirect_effect,
                'direct_effect': direct_effect
            }
        
        # For Y -> HU path
        if 'Y_HU' in self.results['total_effects']:
            total_effect = self.results['total_effects']['Y_HU']['coefficient']
            check_beta = self.results['total_effects']['Y_HU']['check_beta']
            
            direct_effect = check_beta
            indirect_effect = total_effect - direct_effect
            
            self.results['mediation_analysis']['HU'] = {
                'total_effect': total_effect,
                'indirect_effect': indirect_effect,
                'direct_effect': direct_effect
            }
        
        # For HU -> AI path
        if 'HU_AI' in self.results['total_effects']:
            total_effect = self.results['total_effects']['HU_AI']['coefficient']
            check_beta = self.results['total_effects']['HU_AI']['check_beta']
            
            direct_effect = check_beta
            indirect_effect = total_effect - direct_effect
            
            self.results['mediation_analysis']['HU_AI'] = {
                'total_effect': total_effect,
                'indirect_effect': indirect_effect,
                'direct_effect': direct_effect
            }
        
    def get_top_pcs(self, 
                   n_top: int = 9,
                   ranking_method: str = 'triple') -> List[int]:
        """
        Get top PCs based on ranking method.
        
        Parameters
        ----------
        n_top : int, default=9
            Number of top PCs to return
        ranking_method : str, default='triple'
            Method for ranking: 'SC', 'AI', 'HU', or 'triple'
            
        Returns
        -------
        List[int]
            Indices of top PCs (0-based)
        """
        if ranking_method == 'triple':
            # Get top 3 from each model
            top_pcs = []
            for outcome in ['SC', 'AI', 'HU']:
                if outcome in self.results['debiased_lasso']:
                    coefs = np.abs(self.results['debiased_lasso'][outcome]['coefs_std'])
                    top_indices = np.argsort(coefs)[::-1][:3]
                    top_pcs.extend(top_indices)
            
            # Remove duplicates while preserving order
            seen = set()
            top_pcs_unique = []
            for pc in top_pcs:
                if pc not in seen:
                    seen.add(pc)
                    top_pcs_unique.append(pc)
                    
            return top_pcs_unique[:n_top]
            
        else:
            # Rank by single outcome
            # Handle Y -> SC mapping for criterion
            outcome_key = 'SC' if ranking_method == 'Y' else ranking_method
            
            if outcome_key not in self.results['debiased_lasso']:
                raise ValueError(f"No results for outcome: {ranking_method}")
                
            coefs = np.abs(self.results['debiased_lasso'][outcome_key]['coefs_std'])
            return np.argsort(coefs)[::-1][:n_top].tolist()
    
    def export_results(self, 
                      output_dir: Optional[str] = None,
                      prefix: str = 'haam_results') -> Dict[str, str]:
        """
        Export results to CSV files.
        
        Parameters
        ----------
        output_dir : str, optional
            Output directory. If None, uses current directory
        prefix : str, default='haam_results'
            Prefix for output files
            
        Returns
        -------
        Dict[str, str]
            Dictionary of output file paths
        """
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export coefficient matrix
        coef_data = []
        for pc_idx in range(self.n_components):
            row = {'PC': pc_idx + 1}  # 1-based indexing for export
            
            for outcome in ['SC', 'AI', 'HU']:
                if outcome in self.results['debiased_lasso']:
                    row[f'{outcome}_coef'] = self.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                    row[f'{outcome}_se'] = self.results['debiased_lasso'][outcome]['ses_std'][pc_idx]
                else:
                    row[f'{outcome}_coef'] = 0
                    row[f'{outcome}_se'] = 0
                    
            coef_data.append(row)
        
        coef_df = pd.DataFrame(coef_data)
        coef_path = os.path.join(output_dir, f'{prefix}_coefficients_{timestamp}.csv')
        coef_df.to_csv(coef_path, index=False)
        
        # Export model summary
        summary_data = []
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                res = self.results['debiased_lasso'][outcome]
                summary_data.append({
                    'Outcome': outcome,
                    'N_selected': res['n_selected'],
                    'R2_insample': res['r2_lasso'],  # Use LASSO R² for global metrics
                    'R2_cv': res['r2_cv_lasso'],  # Use LASSO CV R² for global metrics
                    'Alpha': res['lasso_alpha']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, f'{prefix}_summary_{timestamp}.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Store paths
        self.results['exports'] = {
            'coefficients': coef_path,
            'summary': summary_path
        }
        
        print(f"\nResults exported to:")
        print(f"  - {coef_path}")
        print(f"  - {summary_path}")
        
        # Display in Colab if available
        self._display_in_colab(coef_df, summary_df)
        
        return self.results['exports']
    
    def _display_in_colab(self, *dataframes):
        """Display dataframes and visualizations in Google Colab if available."""
        try:
            from IPython.display import display, HTML
            import google.colab
            in_colab = True
        except ImportError:
            in_colab = False
            return
            
        if in_colab:
            for i, df in enumerate(dataframes):
                if isinstance(df, pd.DataFrame):
                    # Determine table title based on content
                    if 'PC' in df.columns and any(col.endswith('_coef') for col in df.columns):
                        title = "Post-LASSO Coefficients (Standardized)"
                        # Show only first 20 rows for coefficients
                        display(HTML(f"<h3>{title}</h3>"))
                        display(df.head(20).style.format({
                            col: '{:.4f}' for col in df.columns if col.endswith(('_coef', '_se'))
                        }).set_caption("Showing first 20 PCs"))
                    elif 'Outcome' in df.columns and 'R2_cv' in df.columns:
                        title = "Model Summary Statistics"
                        display(HTML(f"<h3>{title}</h3>"))
                        display(df.style.format({
                            'R2_insample': '{:.4f}',
                            'R2_cv': '{:.4f}',
                            'Alpha': '{:.2e}'
                        }))
                    else:
                        display(df)
                    
    def display_mediation_results(self):
        """Display mediation analysis results with visualization in Colab."""
        if 'mediation_analysis' not in self.results:
            print("No mediation analysis results available.")
            return
            
        # Create summary dataframe
        med_data = []
        for outcome, results in self.results['mediation_analysis'].items():
            total_effect = results.get('total_effect', 0)
            direct_effect = results.get('direct_effect', 0)
            indirect_effect = results.get('indirect_effect', 0)
            
            if total_effect != 0:
                poma = 1 - (direct_effect / total_effect)
            else:
                poma = np.nan
                
            med_data.append({
                'Path': f'Y -> {outcome}' if outcome != 'HU_AI' else 'HU -> AI',
                'Total Effect': total_effect,
                'Direct Effect': direct_effect,
                'Indirect Effect': indirect_effect,
                'PoMA': poma,
                'PoMA (%)': poma * 100 if not np.isnan(poma) else np.nan
            })
            
        med_df = pd.DataFrame(med_data)
        
        # Display in Colab if available
        try:
            from IPython.display import display, HTML
            import google.colab
            
            # Create HTML table with styling
            html = """
            <div style="margin: 20px 0;">
                <h3>Mediation Analysis Results (Difference of Coefficients Method)</h3>
                <style>
                    .mediation-table {
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }
                    .mediation-table th, .mediation-table td {
                        border: 1px solid #ddd;
                        padding: 12px;
                        text-align: left;
                    }
                    .mediation-table th {
                        background-color: #4CAF50;
                        color: white;
                    }
                    .mediation-table tr:nth-child(even) {
                        background-color: #f2f2f2;
                    }
                    .formula {
                        background-color: #e8f4f8;
                        padding: 10px;
                        margin: 10px 0;
                        border-radius: 5px;
                        font-family: monospace;
                    }
                </style>
            """
            
            # Convert dataframe to HTML
            html += med_df.to_html(index=False, classes='mediation-table', float_format=lambda x: f'{x:.4f}')
            
            # Add formula explanation
            html += """
            <div class="formula">
                <strong>PoMA (Proportion of Mediated Accuracy)</strong> = 1 - (Direct Effect / Total Effect)<br>
                <strong>Indirect Effect</strong> = Total Effect - Direct Effect<br>
                <br>
                Where:<br>
                - Total Effect: Simple regression coefficient (no controls)<br>
                - Direct Effect: DML check beta (controlling for PC mediators)<br>
                - Indirect Effect: Effect mediated through the PCs
            </div>
            </div>
            """
            
            display(HTML(html))
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Effects comparison
            paths = med_df['Path'].values
            x = np.arange(len(paths))
            width = 0.25
            
            ax1.bar(x - width, med_df['Total Effect'], width, label='Total Effect', alpha=0.8)
            ax1.bar(x, med_df['Direct Effect'], width, label='Direct Effect', alpha=0.8)
            ax1.bar(x + width, med_df['Indirect Effect'], width, label='Indirect Effect', alpha=0.8)
            ax1.set_xlabel('Causal Path')
            ax1.set_ylabel('Effect Size')
            ax1.set_title('Effect Decomposition by Path')
            ax1.set_xticks(x)
            ax1.set_xticklabels(paths, rotation=45)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # PoMA values
            ax2.bar(paths, med_df['PoMA (%)'], alpha=0.8, color='green')
            ax2.set_xlabel('Causal Path')
            ax2.set_ylabel('PoMA (%)')
            ax2.set_title('Proportion of Mediated Accuracy')
            ax2.set_ylim(0, 100)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(med_df['PoMA (%)'].values):
                if not np.isnan(v):
                    ax2.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            # Not in Colab, just print results
            print("\nMediation Analysis Results:")
            print(med_df.to_string(index=False))
    
    def display_all_results(self):
        """Display all HAAM results including coefficients and statistics in Colab."""
        try:
            from IPython.display import display, HTML
            import google.colab
            
            # Display header
            display(HTML("""
            <div style="margin: 20px 0;">
                <h2>HAAM Analysis Results</h2>
            </div>
            """))
            
            # Create and display coefficient matrix
            if 'debiased_lasso' in self.results:
                # Create coefficient dataframe
                coef_data = []
                for pc_idx in range(self.n_components):
                    row = {'PC': pc_idx + 1}
                    for outcome in ['SC', 'AI', 'HU']:
                        if outcome in self.results['debiased_lasso']:
                            row[f'{outcome}_coef'] = self.results['debiased_lasso'][outcome]['coefs_std'][pc_idx]
                            # Check if PC was selected by LASSO
                            row[f'{outcome}_selected'] = '✓' if self.results['debiased_lasso'][outcome]['selected'][pc_idx] else ''
                        else:
                            row[f'{outcome}_coef'] = 0
                            row[f'{outcome}_selected'] = ''
                    coef_data.append(row)
                
                coef_df = pd.DataFrame(coef_data)
                
                # Display coefficient table
                display(HTML("<h3>Post-LASSO Standardized Coefficients</h3>"))
                
                # Show top 20 PCs with highest absolute coefficients
                # Calculate max absolute coefficient for each PC
                coef_df['max_abs_coef'] = coef_df[['SC_coef', 'AI_coef', 'HU_coef']].abs().max(axis=1)
                top_pcs = coef_df.nlargest(20, 'max_abs_coef')
                
                # Format for display
                styled_df = top_pcs[['PC', 'SC_coef', 'SC_selected', 'AI_coef', 'AI_selected', 'HU_coef', 'HU_selected']].style\
                    .format({
                        'SC_coef': '{:.4f}',
                        'AI_coef': '{:.4f}',
                        'HU_coef': '{:.4f}'
                    })\
                    .set_caption("Top 20 PCs by maximum absolute coefficient (✓ = selected by LASSO)")
                
                display(styled_df)
            
            # Display model summary
            if 'debiased_lasso' in self.results:
                summary_data = []
                for outcome in ['SC', 'AI', 'HU']:
                    if outcome in self.results['debiased_lasso']:
                        res = self.results['debiased_lasso'][outcome]
                        summary_data.append({
                            'Outcome': 'Y' if outcome == 'SC' else outcome,
                            'N_selected': res['n_selected'],
                            'R²(CV)': res['r2_cv'],
                            'R²(In-sample)': res['r2_insample'],
                            'LASSO α': res['lasso_alpha']
                        })
                
                summary_df = pd.DataFrame(summary_data)
                display(HTML("<h3>Model Performance Summary</h3>"))
                display(summary_df.style.format({
                    'R²(CV)': '{:.4f}',
                    'R²(In-sample)': '{:.4f}',
                    'LASSO α': '{:.2e}'
                }))
                
        except ImportError:
            print("Display functions require Google Colab environment")