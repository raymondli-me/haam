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
                 random_state: int = 42):
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
        """
        self.criterion = self._validate_input(criterion, "criterion")
        self.ai_judgment = self._validate_input(ai_judgment, "ai_judgment")
        self.human_judgment = self._validate_input(human_judgment, "human_judgment")
        self.n_components = n_components
        self.random_state = random_state
        
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
        r2_cv = self._calculate_cv_r2(X, y_std, selected, coefs)
        
        # Unstandardize coefficients
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
        """Calculate Double Machine Learning (DML) total effects."""
        # Get predictions from each model
        predictions = {}
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                X = self.results['pca_features']
                coefs = self.results['debiased_lasso'][outcome]['coefs_std']
                predictions[outcome] = X @ coefs
        
        # Calculate total effects for key paths
        # Y -> AI (direct effect of criterion on AI)
        if 'SC' in predictions and 'AI' in self.results['debiased_lasso']:
            mask = ~(np.isnan(self.criterion) | np.isnan(self.ai_judgment))
            if mask.sum() > 50:
                # Simple OLS for total effect
                X = sm.add_constant(self.criterion[mask])
                y = self.ai_judgment[mask]
                model = sm.OLS(y, X).fit()
                self.results['total_effects']['Y_AI'] = {
                    'coefficient': model.params[1],
                    'se': model.bse[1],
                    'check_beta': np.corrcoef(predictions['SC'][mask], self.ai_judgment[mask])[0, 1]
                }
        
        # Y -> HU (direct effect of criterion on human)
        if 'SC' in predictions and 'HU' in self.results['debiased_lasso']:
            mask = ~(np.isnan(self.criterion) | np.isnan(self.human_judgment))
            if mask.sum() > 50:
                X = sm.add_constant(self.criterion[mask])
                y = self.human_judgment[mask]
                model = sm.OLS(y, X).fit()
                self.results['total_effects']['Y_HU'] = {
                    'coefficient': model.params[1],
                    'se': model.bse[1],
                    'check_beta': np.corrcoef(predictions['SC'][mask], self.human_judgment[mask])[0, 1]
                }
        
        # HU -> AI (effect of human on AI)
        if 'HU' in predictions and 'AI' in self.results['debiased_lasso']:
            mask = ~(np.isnan(self.human_judgment) | np.isnan(self.ai_judgment))
            if mask.sum() > 50:
                X = sm.add_constant(self.human_judgment[mask])
                y = self.ai_judgment[mask]
                model = sm.OLS(y, X).fit()
                self.results['total_effects']['HU_AI'] = {
                    'coefficient': model.params[1],
                    'se': model.bse[1],
                    'check_beta': np.corrcoef(predictions['HU'][mask], self.ai_judgment[mask])[0, 1]
                }
    
    def _calculate_residual_correlations(self):
        """Calculate residual correlations (C's) between residuals after controlling for other variables."""
        # C(AI,HU) - residual correlation between e_AI and e_HU after controlling for Y
        mask = ~(np.isnan(self.criterion) | np.isnan(self.ai_judgment) | np.isnan(self.human_judgment))
        if mask.sum() > 50:
            # Residualize AI and HU with respect to Y
            X = sm.add_constant(self.criterion[mask])
            
            # Get AI residuals (e_AI)
            model_ai = sm.OLS(self.ai_judgment[mask], X).fit()
            resid_ai = model_ai.resid
            
            # Get HU residuals (e_HU)
            model_hu = sm.OLS(self.human_judgment[mask], X).fit()
            resid_hu = model_hu.resid
            
            # C(AI,HU) = corr(e_AI, e_HU)
            self.results['residual_correlations']['AI_HU'] = np.corrcoef(resid_ai, resid_hu)[0, 1]
        else:
            self.results['residual_correlations']['AI_HU'] = 0.0
        
        # C(Y,AI) - residual correlation between e_Y and e_AI after controlling for HU
        mask_y_ai = ~(np.isnan(self.criterion) | np.isnan(self.ai_judgment) | np.isnan(self.human_judgment))
        if mask_y_ai.sum() > 50:
            # Control for HU
            X_hu = sm.add_constant(self.human_judgment[mask_y_ai])
            
            # Get Y residuals after controlling for HU
            model_y = sm.OLS(self.criterion[mask_y_ai], X_hu).fit()
            resid_y = model_y.resid
            
            # Get AI residuals after controlling for HU
            model_ai = sm.OLS(self.ai_judgment[mask_y_ai], X_hu).fit()
            resid_ai = model_ai.resid
            
            # C(Y,AI) = corr(e_Y, e_AI)
            self.results['residual_correlations']['Y_AI'] = np.corrcoef(resid_y, resid_ai)[0, 1]
        else:
            self.results['residual_correlations']['Y_AI'] = 0.0
        
        # C(Y,HU) - residual correlation between e_Y and e_HU after controlling for AI
        mask_y_hu = ~(np.isnan(self.criterion) | np.isnan(self.human_judgment) | np.isnan(self.ai_judgment))
        if mask_y_hu.sum() > 50:
            # Control for AI
            X_ai = sm.add_constant(self.ai_judgment[mask_y_hu])
            
            # Get Y residuals after controlling for AI
            model_y = sm.OLS(self.criterion[mask_y_hu], X_ai).fit()
            resid_y = model_y.resid
            
            # Get HU residuals after controlling for AI
            model_hu = sm.OLS(self.human_judgment[mask_y_hu], X_ai).fit()
            resid_hu = model_hu.resid
            
            # C(Y,HU) = corr(e_Y, e_HU)
            self.results['residual_correlations']['Y_HU'] = np.corrcoef(resid_y, resid_hu)[0, 1]
        else:
            self.results['residual_correlations']['Y_HU'] = 0.0
    
    def _calculate_policy_similarities(self):
        """Calculate correlations between model predictions (policy similarities)."""
        # Get predictions from each model
        predictions = {}
        for outcome in ['SC', 'AI', 'HU']:
            if outcome in self.results['debiased_lasso']:
                X = self.results['pca_features']
                coefs = self.results['debiased_lasso'][outcome]['coefs_std']
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
        """Calculate proportion of maximum achievable (PoMA) mediation."""
        # For Y -> AI path
        if 'Y_AI' in self.results['total_effects'] and 'SC' in self.results['debiased_lasso']:
            total_effect = self.results['total_effects']['Y_AI']['coefficient']
            
            # Calculate indirect effect through PCs
            # This is simplified - a full implementation would use proper mediation analysis
            X = self.results['pca_features']
            sc_coefs = self.results['debiased_lasso']['SC']['coefs_std']
            ai_coefs = self.results['debiased_lasso']['AI']['coefs_std'] if 'AI' in self.results['debiased_lasso'] else np.zeros_like(sc_coefs)
            
            # Indirect effect approximation
            indirect_effect = np.sum(sc_coefs * ai_coefs) * np.std(self.criterion) * np.std(self.ai_judgment)
            
            self.results['mediation_analysis']['AI'] = {
                'total_effect': total_effect,
                'indirect_effect': indirect_effect,
                'direct_effect': total_effect - indirect_effect
            }
        
        # For Y -> HU path
        if 'Y_HU' in self.results['total_effects'] and 'SC' in self.results['debiased_lasso']:
            total_effect = self.results['total_effects']['Y_HU']['coefficient']
            
            X = self.results['pca_features']
            sc_coefs = self.results['debiased_lasso']['SC']['coefs_std']
            hu_coefs = self.results['debiased_lasso']['HU']['coefs_std'] if 'HU' in self.results['debiased_lasso'] else np.zeros_like(sc_coefs)
            
            indirect_effect = np.sum(sc_coefs * hu_coefs) * np.std(self.criterion) * np.std(self.human_judgment)
            
            self.results['mediation_analysis']['HU'] = {
                'total_effect': total_effect,
                'indirect_effect': indirect_effect,
                'direct_effect': total_effect - indirect_effect
            }
        
        # For HU -> AI path
        if 'HU_AI' in self.results['total_effects'] and 'HU' in self.results['debiased_lasso']:
            total_effect = self.results['total_effects']['HU_AI']['coefficient']
            
            X = self.results['pca_features']
            hu_coefs = self.results['debiased_lasso']['HU']['coefs_std'] if 'HU' in self.results['debiased_lasso'] else np.zeros(self.n_components)
            ai_coefs = self.results['debiased_lasso']['AI']['coefs_std'] if 'AI' in self.results['debiased_lasso'] else np.zeros(self.n_components)
            
            indirect_effect = np.sum(hu_coefs * ai_coefs) * np.std(self.human_judgment) * np.std(self.ai_judgment)
            
            self.results['mediation_analysis']['HU_AI'] = {
                'total_effect': total_effect,
                'indirect_effect': indirect_effect,
                'direct_effect': total_effect - indirect_effect
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
                    'R2_insample': res['r2_insample'],
                    'R2_cv': res['r2_cv'],
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
        
        return self.results['exports']