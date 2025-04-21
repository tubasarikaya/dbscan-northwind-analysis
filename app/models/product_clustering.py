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

def get_product_features(db: Session) -> pd.DataFrame:
    """
    Retrieves and processes product features from the database.
    """
    try:
        query = """
        SELECT 
            p.product_id,
            p.unit_price,
            p.units_in_stock,
            p.units_on_order,
            p.reorder_level,
            COUNT(DISTINCT o.order_id) as order_count,
            COALESCE(SUM(od.quantity), 0) as total_quantity,
            COUNT(DISTINCT c.customer_id) as unique_customers
        FROM products p
        LEFT JOIN order_details od ON p.product_id = od.product_id
        LEFT JOIN orders o ON od.order_id = o.order_id
        LEFT JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY p.product_id, p.unit_price, p.units_in_stock, p.units_on_order, p.reorder_level
        """
        
        df = pd.read_sql(query, db.connection())
        if df.empty:
            raise ValueError("No product data found in the database")
        
        missing_values = df.isnull().sum()
        if missing_values.any():
            df = df.fillna(0)
        
        return df
    except SQLAlchemyError as e:
        raise
    except Exception as e:
        raise

def find_optimal_eps_product(df: pd.DataFrame, min_samples: int = 2) -> float:
    """
    Finds the optimal eps value for product data.
    """
    try:
        features = df[['unit_price', 'units_in_stock', 'units_on_order', 'reorder_level', 
                      'order_count', 'total_quantity', 'unique_customers']]
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

def find_optimal_min_samples_product(df: pd.DataFrame) -> int:
    """
    Finds the optimal min_samples value for product data.
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

def plot_k_distance_product(df: pd.DataFrame, min_samples: int = 2) -> None:
    """
    Creates and saves the k-distance graph for product data.
    """
    try:
        features = df[['unit_price', 'units_in_stock', 'units_on_order', 'reorder_level', 
                      'order_count', 'total_quantity', 'unique_customers']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        nbrs = NearestNeighbors(n_neighbors=min_samples)
        nbrs.fit(scaled_features)
        distances, _ = nbrs.kneighbors(scaled_features)
        
        distances = np.sort(distances[:, -1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.title('K-Distance Graph for Product Data')
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
        
        plt.savefig('product_k_distance.png')
        plt.close()
    except Exception as e:
        pass

def perform_product_clustering(df: pd.DataFrame, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
    """
    Performs product clustering using DBSCAN.
    """
    try:
        features = df[['unit_price', 'units_in_stock', 'units_on_order', 'reorder_level', 
                      'order_count', 'total_quantity', 'unique_customers']]
        
        if features.isnull().any().any():
            features = features.fillna(0)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        if eps is None:
            eps = find_optimal_eps_product(df)
        if min_samples is None:
            min_samples = find_optimal_min_samples_product(df)
        
        plot_k_distance_product(df, min_samples)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_features)
        
        df['cluster'] = clusters
        results = {
            'products': df.to_dict(orient='records'),
            'outliers': df[df['cluster'] == -1].to_dict(orient='records'),
            'cluster_stats': df.groupby('cluster').agg({
                'unit_price': ['mean', 'std'],
                'units_in_stock': ['mean', 'std'],
                'units_on_order': ['mean', 'std'],
                'reorder_level': ['mean', 'std'],
                'order_count': ['mean', 'std'],
                'total_quantity': ['mean', 'std'],
                'unique_customers': ['mean', 'std']
            }).to_dict(),
            'params': {
                'eps': eps,
                'min_samples': min_samples
            }
        }
        
        return results
    except Exception as e:
        raise

def analyze_product_clusters(db: Session, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
    """
    Performs product clustering analysis and returns results.
    """
    try:
        df = get_product_features(db)
        results = perform_product_clustering(df, eps, min_samples)
        return results
    except Exception as e:
        raise 