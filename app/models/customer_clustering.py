import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from kneed import KneeLocator

def get_customer_features(db: Session) -> pd.DataFrame:
    """
    Retrieves and processes customer features from the database.
    """
    try:
        query = """
        SELECT 
            c.customer_id,
            COUNT(DISTINCT o.order_id) as order_count,
            COALESCE(SUM(od.quantity), 0) as total_quantity,
            COALESCE(AVG(od.unit_price), 0) as avg_unit_price,
            COUNT(DISTINCT p.category_id) as unique_categories
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        LEFT JOIN order_details od ON o.order_id = od.order_id
        LEFT JOIN products p ON od.product_id = p.product_id
        GROUP BY c.customer_id
        """
        
        df = pd.read_sql(query, db.connection())
        if df.empty:
            raise ValueError("No customer data found in the database")
        
        missing_values = df.isnull().sum()
        if missing_values.any():
            df = df.fillna(0)
        
        return df
    except SQLAlchemyError as e:
        raise
    except Exception as e:
        raise

def find_optimal_eps_customer(df: pd.DataFrame, min_samples: int = 2) -> float:
    """
    Finds the optimal eps value for customer data.
    """
    try:
        features = df[['order_count', 'total_quantity', 'avg_unit_price', 'unique_categories']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
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
        
        return distances[kneedle.knee]
    except Exception as e:
        return 0.5

def find_optimal_min_samples_customer(df: pd.DataFrame) -> int:
    """
    Finds the optimal min_samples value for customer data.
    """
    try:
        n_samples = len(df)
        if n_samples < 100:
            return 2
        elif n_samples < 1000:
            return 3
        else:
            return 4
    except Exception as e:
        return 2

def plot_k_distance_customer(df: pd.DataFrame, min_samples: int = 2) -> None:
    """
    Creates and saves the k-distance graph for customer data.
    """
    try:
        features = df[['order_count', 'total_quantity', 'avg_unit_price', 'unique_categories']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        nbrs = NearestNeighbors(n_neighbors=min_samples)
        nbrs.fit(scaled_features)
        distances, _ = nbrs.kneighbors(scaled_features)
        
        distances = np.sort(distances[:, -1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.title('K-Distance Graph for Customer Data')
        plt.xlabel('Point Index')
        plt.ylabel(f'{min_samples}-neighbor distance')
        
        kneedle = KneeLocator(
            range(1, len(distances) + 1),
            distances,
            curve='convex',
            direction='increasing'
        )
        plt.axvline(x=kneedle.knee, color='r', linestyle='--', label=f'Knee Point (eps={distances[kneedle.knee]:.2f})')
        plt.legend()
        
        plt.savefig('customer_k_distance.png')
        plt.close()
    except Exception as e:
        pass

def perform_customer_clustering(df: pd.DataFrame, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
    """
    Performs customer clustering using DBSCAN.
    """
    try:
        features = df[['order_count', 'total_quantity', 'avg_unit_price', 'unique_categories']]
        
        if features.isnull().any().any():
            features = features.fillna(0)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        if eps is None:
            eps = find_optimal_eps_customer(df)
        if min_samples is None:
            min_samples = find_optimal_min_samples_customer(df)
        
        plot_k_distance_customer(df, min_samples)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_features)
        
        df['cluster'] = clusters
        results = {
            'customers': df.to_dict(orient='records'),
            'outliers': df[df['cluster'] == -1].to_dict(orient='records'),
            'cluster_stats': df.groupby('cluster').agg({
                'order_count': ['mean', 'std'],
                'total_quantity': ['mean', 'std'],
                'avg_unit_price': ['mean', 'std'],
                'unique_categories': ['mean', 'std']
            }).to_dict(),
            'params': {
                'eps': eps,
                'min_samples': min_samples
            }
        }
        
        return results
    except Exception as e:
        raise

def analyze_customer_clusters(db: Session, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
    """
    Performs customer clustering analysis and returns results.
    """
    try:
        df = get_customer_features(db)
        results = perform_customer_clustering(df, eps, min_samples)
        return results
    except Exception as e:
        raise 