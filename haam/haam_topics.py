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
                 min_cluster_size: int = 5,
                 min_samples: int = 3,
                 cluster_selection_epsilon: float = 0.0):
        """
        Initialize topic analyzer.
        
        Parameters
        ----------
        texts : List[str]
            Original text documents
        embeddings : np.ndarray
            Document embeddings
        pca_features : np.ndarray
            PCA-transformed features
        min_cluster_size : int
            Minimum cluster size for HDBSCAN (default: 5 for fine-grained clusters)
        min_samples : int
            Minimum samples for core points (default: 3)
        cluster_selection_epsilon : float
            Epsilon for cluster selection (default: 0.0)
        """
        self.texts = texts
        self.embeddings = embeddings
        self.pca_features = pca_features
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        
        # Perform clustering
        self._cluster_documents()
        
        # Extract keywords
        self._extract_keywords()
        
    def _cluster_documents(self):
        """Cluster documents using HDBSCAN."""
        print("Clustering documents...")
        
        # Check HDBSCAN version compatibility
        import inspect
        hdbscan_params = inspect.signature(HDBSCAN.__init__).parameters
        
        # Base parameters
        params = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'metric': 'euclidean',
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'cluster_selection_method': 'eom'  # Use EOM for better small clusters
        }
        
        # Add prediction_data only if supported
        if 'prediction_data' in hdbscan_params:
            params['prediction_data'] = True
            
        clusterer = HDBSCAN(**params)
        
        self.cluster_labels = clusterer.fit_predict(self.embeddings)
        self.n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        
        print(f"Found {self.n_clusters} clusters")
        
    def _extract_keywords(self, n_keywords: int = 10):
        """Extract keywords for each cluster using c-TF-IDF."""
        print("Extracting cluster keywords...")
        
        # Prepare documents by cluster
        cluster_docs = {}
        for idx, label in enumerate(self.cluster_labels):
            if label != -1:  # Skip noise
                if label not in cluster_docs:
                    cluster_docs[label] = []
                cluster_docs[label].append(self.texts[idx])
        
        # Create merged documents per cluster
        merged_docs = []
        cluster_ids = []
        for cluster_id, docs in sorted(cluster_docs.items()):
            merged_docs.append(' '.join(docs))
            cluster_ids.append(cluster_id)
        
        # TF-IDF on merged documents
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(merged_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top keywords per cluster
        self.topic_keywords = {}
        
        for idx, cluster_id in enumerate(cluster_ids):
            # Get top keywords
            tfidf_scores = tfidf_matrix[idx].toarray()[0]
            top_indices = np.argsort(tfidf_scores)[::-1][:n_keywords]
            
            keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            self.topic_keywords[cluster_id] = ' | '.join(keywords[:5])  # Top 5 for display
            
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