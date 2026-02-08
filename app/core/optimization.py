import numpy as np
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import pandas as pd


class ParameterOptimizer:
    @staticmethod
    def find_optimal_eps(scaled_features: np.ndarray, min_samples: int = 2) -> float:
        try:
            nbrs = NearestNeighbors(n_neighbors=min_samples)
            nbrs.fit(scaled_features)
            distances, _ = nbrs.kneighbors(scaled_features)
            
            distances = np.sort(distances[:, -1])
            
            kneedle = KneeLocator(
                range(1, len(distances) + 1),
                distances,
                curve='convex',
                direction='increasing'
            )
            
            return distances[kneedle.knee] if kneedle.knee else 0.5
        except Exception:
            return 0.5
    
    @staticmethod
    def find_optimal_min_samples(n_samples: int) -> int:
        if n_samples < 100:
            return 2
        elif n_samples < 1000:
            return 3
        else:
            return 4
