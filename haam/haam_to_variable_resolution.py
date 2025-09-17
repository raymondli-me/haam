"""
HAAM to Variable Resolution Data Standard Converter
==================================================

This module provides functionality to convert HAAM analysis outputs
into the Variable Resolution Data Standard format for visualization.

The converter takes HAAM results and produces a JSON file that conforms
to the Variable Resolution Data Standard v1.0, enabling seamless
integration with the Variable Resolution visualization engine.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import warnings
from pathlib import Path

class HAAMToVariableResolution:
    """
    Converts HAAM analysis results to Variable Resolution Data Standard format.
    
    This class takes the outputs from HAAM analysis (including PCs, scores,
    clusters, and positions) and formats them according to the Variable
    Resolution Data Standard for use in the visualization engine.
    """
    
    def __init__(self, haam_instance=None):
        """
        Initialize the converter.
        
        Parameters
        ----------
        haam_instance : HAAM, optional
            An instance of HAAM class with completed analysis.
            If provided, will extract data directly from the instance.
        """
        self.haam = haam_instance
        self.data_standard = {
            "version": "1.0",
            "metadata": {},
            "schema": {
                "variables": {},
                "clusters": None,
                "positioning": None
            },
            "data": {
                "items": []
            }
        }
        
    def convert_from_haam(self,
                         title: str = "HAAM Analysis Results",
                         description: str = "Human-AI alignment analysis with principal components",
                         author: Optional[str] = None,
                         include_pcs: int = 10,
                         pc_prefix: str = "PC",
                         include_all_pcs: bool = False) -> Dict:
        """
        Convert HAAM instance data to Variable Resolution format.
        
        Parameters
        ----------
        title : str
            Title for the dataset
        description : str
            Description of the dataset
        author : str, optional
            Author of the dataset
        include_pcs : int, default=10
            Number of top principal components to include as variables
        pc_prefix : str, default="PC"
            Prefix for principal component variable names
        include_all_pcs : bool, default=False
            If True, includes all PCs regardless of include_pcs value
            
        Returns
        -------
        Dict
            Variable Resolution Data Standard formatted dictionary
        """
        if not self.haam:
            raise ValueError("HAAM instance not provided. Initialize with a HAAM instance.")
            
        if not hasattr(self.haam, 'results') or not self.haam.results:
            raise ValueError("HAAM analysis not completed. Run haam.run_full_analysis() first.")
            
        # Set metadata
        self._set_metadata(title, description, author)
        
        # Define variables schema
        self._define_variables_schema(include_pcs, pc_prefix, include_all_pcs)
        
        # Set clustering and positioning info
        self._set_clustering_schema()
        self._set_positioning_schema()
        
        # Convert data items
        self._convert_data_items()
        
        return self.data_standard
    
    def convert_from_data(self,
                         criterion: np.ndarray,
                         human_judgment: np.ndarray,
                         ai_judgment: np.ndarray,
                         texts: List[str],
                         ids: Optional[List[Union[str, int]]] = None,
                         pca_features: Optional[np.ndarray] = None,
                         clusters: Optional[Dict] = None,
                         positions: Optional[np.ndarray] = None,
                         additional_variables: Optional[Dict[str, np.ndarray]] = None,
                         title: str = "HAAM Analysis Results",
                         description: str = "Human-AI alignment analysis",
                         author: Optional[str] = None,
                         include_pcs: int = 10) -> Dict:
        """
        Convert raw data arrays to Variable Resolution format.
        
        Parameters
        ----------
        criterion : np.ndarray
            Criterion/ground truth values
        human_judgment : np.ndarray
            Human ratings/judgments
        ai_judgment : np.ndarray
            AI predictions/ratings
        texts : List[str]
            Text content for each item
        ids : List[Union[str, int]], optional
            IDs for each item. If None, will use indices
        pca_features : np.ndarray, optional
            Principal component values (n_samples, n_components)
        clusters : Dict, optional
            Clustering information with 'ids' and 'labels' keys
        positions : np.ndarray, optional
            3D positions (n_samples, 3)
        additional_variables : Dict[str, np.ndarray], optional
            Additional variables to include
        title : str
            Title for the dataset
        description : str
            Description of the dataset
        author : str, optional
            Author of the dataset
        include_pcs : int
            Number of principal components to include
            
        Returns
        -------
        Dict
            Variable Resolution Data Standard formatted dictionary
        """
        # Validate inputs
        n_samples = len(criterion)
        if not all(len(x) == n_samples for x in [human_judgment, ai_judgment, texts]):
            raise ValueError("All inputs must have the same number of samples")
            
        # Set metadata
        self._set_metadata(title, description, author, n_samples)
        
        # Define variables schema
        variables_schema = {
            "criterion": {
                "type": "continuous",
                "displayName": "Criterion Score",
                "range": [float(np.min(criterion)), float(np.max(criterion))],
                "description": "Ground truth or criterion variable"
            },
            "human_judgment": {
                "type": "continuous",
                "displayName": "Human Rating",
                "range": [float(np.min(human_judgment)), float(np.max(human_judgment))],
                "description": "Human ratings or judgments"
            },
            "ai_judgment": {
                "type": "continuous", 
                "displayName": "AI Rating",
                "range": [float(np.min(ai_judgment)), float(np.max(ai_judgment))],
                "description": "AI predictions or ratings"
            }
        }
        
        # Add PCA features if provided
        if pca_features is not None:
            n_pcs = min(include_pcs, pca_features.shape[1])
            for i in range(n_pcs):
                pc_values = pca_features[:, i]
                variables_schema[f"PC{i+1}"] = {
                    "type": "continuous",
                    "displayName": f"Principal Component {i+1}",
                    "range": [float(np.min(pc_values)), float(np.max(pc_values))],
                    "description": f"Principal component {i+1} from embedding analysis"
                }
        
        # Add additional variables
        if additional_variables:
            for var_name, var_values in additional_variables.items():
                if len(var_values) != n_samples:
                    warnings.warn(f"Variable '{var_name}' has incorrect length, skipping")
                    continue
                variables_schema[var_name] = {
                    "type": "continuous",
                    "displayName": var_name.replace("_", " ").title(),
                    "range": [float(np.min(var_values)), float(np.max(var_values))],
                    "description": f"Additional variable: {var_name}"
                }
        
        self.data_standard["schema"]["variables"] = variables_schema
        
        # Set clustering schema if provided
        if clusters is not None:
            self.data_standard["schema"]["clusters"] = {
                "method": "HDBSCAN/BERTopic",
                "field": "cluster",
                "labelField": "cluster.label"
            }
            
        # Set positioning schema if provided
        if positions is not None and positions.shape[1] >= 3:
            self.data_standard["schema"]["positioning"] = {
                "method": "UMAP",
                "dimensions": 3,
                "field": "position",
                "parameters": {
                    "n_neighbors": 5,
                    "min_dist": 0.0,
                    "metric": "cosine"
                }
            }
        
        # Convert data items
        items = []
        for i in range(n_samples):
            item = {
                "id": ids[i] if ids else i+1,
                "content": texts[i],
                "title": f"Item {i+1}",
                "values": {
                    "criterion": float(criterion[i]),
                    "human_judgment": float(human_judgment[i]),
                    "ai_judgment": float(ai_judgment[i])
                }
            }
            
            # Add PC values
            if pca_features is not None:
                for j in range(n_pcs):
                    item["values"][f"PC{j+1}"] = float(pca_features[i, j])
                    
            # Add additional variable values
            if additional_variables:
                for var_name, var_values in additional_variables.items():
                    if var_name in variables_schema:
                        item["values"][var_name] = float(var_values[i])
            
            # Add cluster info
            if clusters is not None and i < len(clusters.get("ids", [])):
                cluster_idx = clusters["ids"][i]
                item["cluster"] = {
                    "id": int(cluster_idx),
                    "label": clusters.get("labels", {}).get(cluster_idx, f"Cluster {cluster_idx}")
                }
                
            # Add position
            if positions is not None and i < len(positions):
                item["position"] = {
                    "x": float(positions[i, 0]),
                    "y": float(positions[i, 1]),
                    "z": float(positions[i, 2]) if positions.shape[1] >= 3 else 0.0
                }
                
            items.append(item)
            
        self.data_standard["data"]["items"] = items
        
        return self.data_standard
    
    def _set_metadata(self, title: str, description: str, author: Optional[str], 
                     n_samples: Optional[int] = None):
        """Set metadata section."""
        self.data_standard["metadata"] = {
            "title": title,
            "description": description,
            "created": datetime.now().isoformat() + "Z",
            "source": "HAAM Analysis Package",
            "tags": ["haam", "human-ai-alignment", "principal-components", "dml-lm"]
        }
        
        if author:
            self.data_standard["metadata"]["author"] = author
            
        if n_samples or (self.haam and hasattr(self.haam, 'criterion')):
            size = n_samples or len(self.haam.criterion)
            self.data_standard["metadata"]["datasetSize"] = size
            
        # Add processing info if HAAM instance available
        if self.haam:
            self.data_standard["metadata"]["processingInfo"] = {
                "method": "HAAM (Human-AI Alignment Model)",
                "parameters": {
                    "n_components": getattr(self.haam, 'n_components', 200),
                    "min_cluster_size": getattr(self.haam, 'min_cluster_size', 10),
                    "umap_n_components": getattr(self.haam, 'umap_n_components', 3),
                    "standardize": getattr(self.haam, 'standardize', False)
                },
                "timestamp": datetime.now().isoformat() + "Z"
            }
    
    def _define_variables_schema(self, include_pcs: int, pc_prefix: str, include_all_pcs: bool):
        """Define variables in the schema."""
        if not self.haam:
            return
            
        variables = {}
        
        # Core HAAM variables
        core_vars = {
            "criterion": ("Criterion Score", self.haam.criterion, 
                         "Ground truth or criterion variable (X)"),
            "human_judgment": ("Human Rating", self.haam.human_judgment,
                             "Human ratings or judgments (HU)"),
            "ai_judgment": ("AI Rating", self.haam.ai_judgment,
                           "AI predictions or ratings (AI)")
        }
        
        for var_name, (display_name, data, description) in core_vars.items():
            variables[var_name] = {
                "type": "continuous",
                "displayName": display_name,
                "range": [float(np.nanmin(data)), float(np.nanmax(data))],
                "description": description,
                "unit": "score"
            }
        
        # Add principal components
        if hasattr(self.haam.analysis, 'results') and 'pca_features' in self.haam.analysis.results:
            pca_features = self.haam.analysis.results['pca_features']
            n_pcs = pca_features.shape[1] if include_all_pcs else min(include_pcs, pca_features.shape[1])
            
            for i in range(n_pcs):
                pc_data = pca_features[:, i]
                var_name = f"{pc_prefix}{i+1}"
                
                # Get variance explained if available
                var_exp = ""
                if 'variance_explained' in self.haam.analysis.results:
                    var_exp = f" ({self.haam.analysis.results['variance_explained'][i]*100:.1f}% variance)"
                
                variables[var_name] = {
                    "type": "continuous",
                    "displayName": f"Principal Component {i+1}",
                    "range": [float(np.min(pc_data)), float(np.max(pc_data))],
                    "description": f"PC{i+1} from text embeddings{var_exp}",
                    "metadata": {
                        "component_index": i,
                        "variance_explained": float(self.haam.analysis.results.get('variance_explained', [0]*n_pcs)[i])
                    }
                }
        
        # Add derived metrics if available
        if hasattr(self.haam, 'results') and 'model_summary' in self.haam.results:
            summary = self.haam.results['model_summary']
            
            # Add correlations as metadata
            for key, value in summary.items():
                if 'correlation' in key.lower() and isinstance(value, (int, float)):
                    var_parts = key.split('_')
                    if len(var_parts) >= 2:
                        var1, var2 = var_parts[0], var_parts[1]
                        if var1 in variables:
                            if 'metadata' not in variables[var1]:
                                variables[var1]['metadata'] = {}
                            variables[var1]['metadata'][f'correlation_with_{var2}'] = float(value)
        
        self.data_standard["schema"]["variables"] = variables
    
    def _set_clustering_schema(self):
        """Set clustering information in schema."""
        if self.haam and hasattr(self.haam, 'topic_analyzer') and self.haam.topic_analyzer:
            self.data_standard["schema"]["clusters"] = {
                "method": "HDBSCAN with BERTopic",
                "field": "cluster",
                "labelField": "cluster.label",
                "parameters": {
                    "min_cluster_size": getattr(self.haam, 'min_cluster_size', 10),
                    "min_samples": getattr(self.haam, 'min_samples', 2)
                }
            }
    
    def _set_positioning_schema(self):
        """Set positioning information in schema."""
        if self.haam and hasattr(self.haam, 'topic_analyzer') and self.haam.topic_analyzer:
            if hasattr(self.haam.topic_analyzer, 'umap_embeddings'):
                self.data_standard["schema"]["positioning"] = {
                    "method": "UMAP",
                    "dimensions": 3,
                    "field": "position",
                    "parameters": {
                        "n_neighbors": 5,
                        "min_dist": 0.0,
                        "n_components": 3,
                        "metric": "cosine"
                    }
                }
    
    def _convert_data_items(self):
        """Convert HAAM data to Variable Resolution items."""
        if not self.haam:
            return
            
        n_samples = len(self.haam.criterion)
        items = []
        
        # Get data arrays
        texts = getattr(self.haam, 'texts', [f"Text {i+1}" for i in range(n_samples)])
        pca_features = None
        if hasattr(self.haam.analysis, 'results') and 'pca_features' in self.haam.analysis.results:
            pca_features = self.haam.analysis.results['pca_features']
            
        # Get clustering info
        clusters = None
        if hasattr(self.haam, 'topic_analyzer') and self.haam.topic_analyzer:
            if hasattr(self.haam.topic_analyzer, 'clusters'):
                clusters = self.haam.topic_analyzer.clusters
                
        # Get positions
        positions = None
        if hasattr(self.haam, 'topic_analyzer') and self.haam.topic_analyzer:
            if hasattr(self.haam.topic_analyzer, 'umap_embeddings'):
                positions = self.haam.topic_analyzer.umap_embeddings
        
        # Create items
        for i in range(n_samples):
            item = {
                "id": i + 1,
                "title": f"Item {i+1}",
                "content": texts[i] if i < len(texts) else f"Item {i+1}",
                "values": {
                    "criterion": float(self.haam.criterion[i]),
                    "human_judgment": float(self.haam.human_judgment[i]),
                    "ai_judgment": float(self.haam.ai_judgment[i])
                }
            }
            
            # Add PC values
            if pca_features is not None:
                n_pcs = min(pca_features.shape[1], 
                          len([k for k in self.data_standard["schema"]["variables"] if k.startswith("PC")]))
                for j in range(n_pcs):
                    item["values"][f"PC{j+1}"] = float(pca_features[i, j])
            
            # Add cluster info
            if clusters is not None and i in clusters:
                cluster_id = clusters[i]
                item["cluster"] = {
                    "id": int(cluster_id) if cluster_id != -1 else -1,
                    "label": f"Topic {cluster_id}" if cluster_id != -1 else "Outlier"
                }
                
                # Add topic label if available
                if hasattr(self.haam, 'topic_summaries') and cluster_id in self.haam.topic_summaries:
                    summary = self.haam.topic_summaries[cluster_id]
                    if 'representative_docs' in summary and i in summary['representative_docs']:
                        item["cluster"]["label"] = summary.get('label', f"Topic {cluster_id}")
                        item["cluster"]["confidence"] = 0.9  # High confidence for representative docs
            
            # Add position
            if positions is not None and i < len(positions):
                item["position"] = {
                    "x": float(positions[i, 0]),
                    "y": float(positions[i, 1]),
                    "z": float(positions[i, 2]) if positions.shape[1] >= 3 else 0.0
                }
                
            # Add metadata
            item["metadata"] = {
                "index": i,
                "timestamp": datetime.now().isoformat() + "Z"
            }
            
            items.append(item)
        
        self.data_standard["data"]["items"] = items
    
    def save_to_file(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """
        Save the Variable Resolution data to a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the JSON file
        indent : int
            Indentation level for JSON formatting
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.data_standard, f, indent=indent)
            
        print(f"✓ Saved Variable Resolution data to: {filepath}")
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the data standard structure.
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ["version", "metadata", "schema", "data"]
        for key in required_keys:
            if key not in self.data_standard:
                errors.append(f"Missing required top-level key: {key}")
        
        # Check metadata
        if "metadata" in self.data_standard:
            required_metadata = ["title", "description", "created"]
            for key in required_metadata:
                if key not in self.data_standard["metadata"]:
                    errors.append(f"Missing required metadata field: {key}")
        
        # Check schema
        if "schema" in self.data_standard:
            if "variables" not in self.data_standard["schema"]:
                errors.append("Missing 'variables' in schema")
            else:
                # Check each variable
                for var_name, var_def in self.data_standard["schema"]["variables"].items():
                    if "type" not in var_def:
                        errors.append(f"Variable '{var_name}' missing 'type'")
                    if "displayName" not in var_def:
                        errors.append(f"Variable '{var_name}' missing 'displayName'")
        
        # Check data items
        if "data" in self.data_standard and "items" in self.data_standard["data"]:
            items = self.data_standard["data"]["items"]
            if len(items) > 0:
                # Check first item as sample
                item = items[0]
                if "id" not in item:
                    errors.append("Data items missing 'id' field")
                if "values" not in item:
                    errors.append("Data items missing 'values' field")
                    
                # Check that all defined variables have values
                if "values" in item and "variables" in self.data_standard.get("schema", {}):
                    for var_name in self.data_standard["schema"]["variables"]:
                        if var_name not in item["values"]:
                            errors.append(f"Variable '{var_name}' not found in data item values")
        
        return len(errors) == 0, errors


# Convenience function for direct conversion
def haam_to_variable_resolution(haam_instance,
                               output_file: Optional[Union[str, Path]] = None,
                               **kwargs) -> Dict:
    """
    Convert HAAM analysis results to Variable Resolution Data Standard.
    
    This is a convenience function that creates a converter instance,
    performs the conversion, and optionally saves to file.
    
    Parameters
    ----------
    haam_instance : HAAM
        A HAAM instance with completed analysis
    output_file : str or Path, optional
        If provided, saves the result to this file
    **kwargs
        Additional arguments passed to convert_from_haam()
        
    Returns
    -------
    Dict
        Variable Resolution Data Standard formatted dictionary
        
    Examples
    --------
    >>> from haam import HAAM
    >>> from haam.haam_to_variable_resolution import haam_to_variable_resolution
    >>> 
    >>> # Run HAAM analysis
    >>> haam = HAAM(criterion=X, ai_judgment=AI, human_judgment=HU, texts=texts)
    >>> 
    >>> # Convert to Variable Resolution format
    >>> vr_data = haam_to_variable_resolution(
    ...     haam, 
    ...     output_file="haam_results.json",
    ...     title="My HAAM Analysis",
    ...     include_pcs=15
    ... )
    """
    converter = HAAMToVariableResolution(haam_instance)
    result = converter.convert_from_haam(**kwargs)
    
    if output_file:
        converter.save_to_file(output_file)
        
    return result