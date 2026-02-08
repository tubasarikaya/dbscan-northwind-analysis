import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, Any, List
from .preprocessing import DataPreprocessor
from .optimization import ParameterOptimizer
from .visualization import ClusterVisualizer


class ClusterAnalyzer:
    def __init__(self, feature_columns: List[str], name: str):
        self.feature_columns = feature_columns
        self.name = name
        self.optimizer = ParameterOptimizer()
        self.visualizer = ClusterVisualizer()
    
    def analyze(self, df: pd.DataFrame, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
        features = df[self.feature_columns]
        
        if features.isnull().any().any():
            features = features.fillna(0)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        if eps is None:
            eps = self.optimizer.find_optimal_eps(scaled_features, min_samples or 2)
        
        if min_samples is None:
            min_samples = self.optimizer.find_optimal_min_samples(len(df))
        
        self.visualizer.plot_k_distance(scaled_features, self.name, min_samples)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_features)
        
        df['cluster'] = clusters
        
        cluster_stats = {}
        for col in self.feature_columns:
            cluster_stats[col] = df.groupby('cluster')[col].agg(['mean', 'std']).to_dict()
        
        return {
            f'{self.name}s': df.to_dict(orient='records'),
            'outliers': df[df['cluster'] == -1].to_dict(orient='records'),
            'cluster_stats': cluster_stats,
            'params': {
                'eps': eps,
                'min_samples': min_samples
            }
        }
