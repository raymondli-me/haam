"""
HAAM Topic Analysis Module
==========================

Functions for topic modeling and PC interpretation.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import HDBSCAN
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TopicAnalyzer:
    """
    Analyze topics and their relationships with principal components.
    """
    
    def __init__(self, texts: List[str], 
                 embeddings: np.ndarray,
                 pca_features: np.ndarray,
                 min_cluster_size: int = 10,
                 min_samples: int = 2,
                 umap_n_components: int = 3):
        """
        Initialize topic analyzer with enhanced parameters.
        
        Now uses optimized hyperparameters:
        - UMAP: n_neighbors=5, min_dist=0.0, metric='cosine'
        - HDBSCAN: min_cluster_size=10, min_samples=2
        - c-TF-IDF: BERTopic formula for better topic extraction
        
        Parameters
        ----------
        texts : List[str]
            Original text documents
        embeddings : np.ndarray
            Document embeddings
        pca_features : np.ndarray
            PCA-transformed features
        min_cluster_size : int
            Minimum cluster size for HDBSCAN (default: 10, matches BERTopic-style)
        min_samples : int
            Minimum samples for core points (default: 2)
        umap_n_components : int
            Number of UMAP components for clustering (default: 3)
        """
        self.texts = texts
        self.embeddings = embeddings
        self.pca_features = pca_features
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_components = umap_n_components
        
        # Perform clustering
        self._cluster_documents()
        
        # Extract keywords
        self._extract_keywords()
        
    def _cluster_documents(self):
        """Cluster documents using HDBSCAN."""
        print("Clustering documents...")
        
        # Use UMAP to reduce dimensionality for better clustering
        print("  Reducing dimensions with UMAP for clustering...")
        import umap
        
        # Use UMAP to get better clustering space (matching my_colab.py exactly)
        umap_reducer = umap.UMAP(
            n_components=self.umap_n_components,  # 3D by default
            n_neighbors=5,  # Changed from 15 to 5 to match my_colab.py
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        clustering_embeddings = umap_reducer.fit_transform(self.embeddings)
        
        # Store UMAP embeddings for later visualization
        self.umap_embeddings = clustering_embeddings
        
        # Check HDBSCAN version compatibility
        import inspect
        hdbscan_params = inspect.signature(HDBSCAN.__init__).parameters
        
        # Base parameters (matching my_colab.py exactly)
        params = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'cluster_selection_method': 'eom',
            'metric': 'euclidean'
        }
        
        # Add version-specific parameters only if supported
        if 'prediction_data' in hdbscan_params:
            params['prediction_data'] = True
            
        if 'core_dist_n_jobs' in hdbscan_params:
            params['core_dist_n_jobs'] = -1
            
        clusterer = HDBSCAN(**params)
        
        self.cluster_labels = clusterer.fit_predict(clustering_embeddings)
        self.n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        
        print(f"Found {self.n_clusters} clusters")
        
    def _extract_keywords(self, n_keywords: int = 10):
        """
        Extract keywords for each cluster using c-TF-IDF (BERTopic style).
        
        Enhanced implementation with:
        - max_features=1000 for richer vocabulary
        - ngram_range=(1,2) for unigrams and bigrams
        - min_df=2 for noise reduction
        
        c-TF-IDF formula: tf_td * log(1 + (A / tf_t))
        where:
            tf_td = term frequency in topic d
            A = average number of words per topic
            tf_t = total term frequency across all documents
        """
        print("Extracting cluster keywords using c-TF-IDF...")
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Initialize CountVectorizer with exact parameters from my_colab.py
        vectorizer = CountVectorizer(
            max_features=1000,  # Changed from 100 to match my_colab.py
            stop_words='english',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2
        )
        
        # Get unique topics (excluding noise)
        unique_topics = np.unique(self.cluster_labels[self.cluster_labels != -1])
        self.topic_keywords = {}
        
        # First, fit vectorizer on all documents
        vectorizer.fit(self.texts)
        vocab = vectorizer.get_feature_names_out()
        
        # Create document-term matrix for all docs
        doc_term_matrix = vectorizer.transform(self.texts)
        
        print(f"  Processing {len(unique_topics)} topics with c-TF-IDF...")
        
        # Calculate c-TF-IDF for each topic
        for topic_id in unique_topics:
            # Get documents in this topic
            topic_mask = self.cluster_labels == topic_id
            
            if np.sum(topic_mask) < 3:  # Skip very small topics
                self.topic_keywords[topic_id] = f"Topic {topic_id} (n={np.sum(topic_mask)})"
                continue
                
            try:
                # Get document-term matrix for this topic
                topic_doc_term = doc_term_matrix[topic_mask]
                
                # Calculate term frequency in this topic
                tf_topic = np.array(topic_doc_term.sum(axis=0)).flatten()
                
                # Calculate total term frequency across all documents
                tf_all = np.array(doc_term_matrix.sum(axis=0)).flatten()
                
                # Calculate c-TF-IDF using BERTopic's formula
                avg_words_per_topic = np.sum(tf_all) / len(unique_topics)
                
                # Avoid division by zero
                tf_all_safe = np.where(tf_all == 0, 1, tf_all)
                
                # Calculate c-TF-IDF
                ctfidf = tf_topic * np.log(1 + (avg_words_per_topic / tf_all_safe))
                
                # Normalize
                ctfidf_norm = ctfidf / np.max(ctfidf) if np.max(ctfidf) > 0 else ctfidf
                
                # Get top keywords
                top_indices = ctfidf_norm.argsort()[-n_keywords:][::-1]
                top_keywords = [vocab[idx] for idx in top_indices]
                
                # Create keyword string (top 5 for display)
                self.topic_keywords[topic_id] = " | ".join(top_keywords[:5])
                
            except Exception as e:
                print(f"  Warning: Could not extract keywords for topic {topic_id}: {e}")
                self.topic_keywords[topic_id] = f"Topic {topic_id}"
            
    def get_pc_topic_associations(self, 
                                 pc_indices: Optional[List[int]] = None,
                                 n_topics: int = 15) -> Dict[int, List[Dict]]:
        """
        Get topic associations for specified PCs.
        
        Parameters
        ----------
        pc_indices : List[int], optional
            PC indices to analyze. If None, analyzes all
        n_topics : int
            Number of top/bottom topics to return per PC
            
        Returns
        -------
        Dict[int, List[Dict]]
            Dictionary mapping PC index to list of topic associations
        """
        if pc_indices is None:
            pc_indices = list(range(self.pca_features.shape[1]))
            
        associations = {}
        
        for pc_idx in pc_indices:
            pc_values = self.pca_features[:, pc_idx]
            topic_stats = []
            
            # Calculate stats for each topic
            for topic_id in range(self.n_clusters):
                topic_mask = self.cluster_labels == topic_id
                if topic_mask.sum() < 5:  # Skip small topics
                    continue
                    
                topic_pc_values = pc_values[topic_mask]
                
                # Calculate percentile of mean PC value
                topic_mean = np.mean(topic_pc_values)
                percentile = stats.percentileofscore(pc_values, topic_mean)
                
                # Calculate effect size (Cohen's d)
                other_mask = ~topic_mask & (self.cluster_labels != -1)
                if other_mask.sum() > 0:
                    other_values = pc_values[other_mask]
                    pooled_std = np.sqrt(
                        (np.var(topic_pc_values) * len(topic_pc_values) + 
                         np.var(other_values) * len(other_values)) / 
                        (len(topic_pc_values) + len(other_values))
                    )
                    effect_size = (topic_mean - np.mean(other_values)) / pooled_std if pooled_std > 0 else 0
                else:
                    effect_size = 0
                
                # Statistical test
                _, p_value = stats.ttest_1samp(topic_pc_values, np.mean(pc_values))
                
                topic_stats.append({
                    'topic_id': topic_id,
                    'keywords': self.topic_keywords.get(topic_id, f'Topic {topic_id}'),
                    'size': topic_mask.sum(),
                    'mean_pc': topic_mean,
                    'avg_percentile': percentile,
                    'effect_size': effect_size,
                    'p_value': p_value
                })
            
            # Sort by percentile
            topic_stats.sort(key=lambda x: x['avg_percentile'], reverse=True)
            associations[pc_idx] = topic_stats
            
        return associations
    
    def get_pc_high_low_topics(self, 
                              pc_idx: int,
                              n_high: int = 5,
                              n_low: int = 5,
                              p_threshold: float = 0.05) -> Dict[str, List[Dict]]:
        """
        Get high and low topics for a specific PC.
        
        Parameters
        ----------
        pc_idx : int
            PC index (0-based)
        n_high : int
            Number of high topics to return
        n_low : int
            Number of low topics to return
        p_threshold : float
            P-value threshold for significance
            
        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary with 'high' and 'low' topic lists
        """
        associations = self.get_pc_topic_associations([pc_idx])[pc_idx]
        
        # Filter by significance
        significant = [a for a in associations if a['p_value'] < p_threshold]
        
        # Get high and low
        high_topics = [a for a in significant if a['avg_percentile'] > 75][:n_high]
        low_topics = [a for a in significant if a['avg_percentile'] < 25][-n_low:]
        
        return {
            'high': high_topics,
            'low': low_topics,
            'all': associations
        }
    
    def create_topic_summary_for_pcs(self, 
                                    pc_indices: List[int],
                                    n_keywords: int = 5,
                                    n_topics_per_side: int = 3) -> Dict[int, Dict[str, List[str]]]:
        """
        Create concise topic summaries for specified PCs.
        
        Parameters
        ----------
        pc_indices : List[int]
            PC indices to summarize
        n_keywords : int
            Number of keywords to show per topic
        n_topics_per_side : int
            Number of high/low topics to include
            
        Returns
        -------
        Dict[int, Dict[str, List[str]]]
            PC index -> {'high_topics': [...], 'low_topics': [...]}
        """
        summaries = {}
        
        for pc_idx in pc_indices:
            high_low = self.get_pc_high_low_topics(
                pc_idx, 
                n_high=n_topics_per_side, 
                n_low=n_topics_per_side
            )
            
            # Extract high topics
            high_topics = []
            for topic in high_low['high']:
                kw_list = topic['keywords'].split(' | ')[:n_keywords]
                high_topics.append(' | '.join(kw_list))
                
            # Extract low topics  
            low_topics = []
            for topic in high_low['low']:
                kw_list = topic['keywords'].split(' | ')[:n_keywords]
                low_topics.append(' | '.join(kw_list))
                
            summaries[pc_idx] = {
                'high_topics': high_topics if high_topics else ['No significant high topics'],
                'low_topics': low_topics if low_topics else ['No significant low topics'],
                'name': f'PC{pc_idx + 1}'  # 1-based for display
            }
            
        return summaries