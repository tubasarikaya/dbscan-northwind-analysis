import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


class ClusterVisualizer:
    @staticmethod
    def plot_k_distance(scaled_features: np.ndarray, name: str, min_samples: int = 2) -> None:
        try:
            nbrs = NearestNeighbors(n_neighbors=min_samples)
            nbrs.fit(scaled_features)
            distances, _ = nbrs.kneighbors(scaled_features)
            
            distances = np.sort(distances[:, -1])
            
            plt.figure(figsize=(10, 6))
            plt.plot(distances)
            plt.title(f'K-Distance Graph for {name.capitalize()} Data')
            plt.xlabel('Point Index')
            plt.ylabel(f'{min_samples}-neighbor distance')
            
            kneedle = KneeLocator(
                range(1, len(distances) + 1),
                distances,
                curve='convex',
                direction='increasing'
            )
            
            if kneedle.knee:
                plt.axvline(x=kneedle.knee, color='r', linestyle='--', 
                           label=f'Optimal eps={distances[kneedle.knee]:.2f}')
                plt.legend()
            
            plt.savefig(f'{name}_k_distance.png')
            plt.close()
        except Exception:
            pass
