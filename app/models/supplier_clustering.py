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

def get_supplier_features(db: Session) -> pd.DataFrame:
    """
    Retrieves and processes supplier features from the database.
    """
    try:
        query = """
        SELECT 
            s.supplier_id,
            COUNT(DISTINCT p.product_id) as product_count,
            COALESCE(SUM(p.units_in_stock), 0) as total_stock,
            COALESCE(SUM(p.units_on_order), 0) as total_on_order,
            COALESCE(AVG(p.unit_price), 0) as avg_product_price,
            COUNT(DISTINCT c.customer_id) as unique_customers
        FROM suppliers s
        LEFT JOIN products p ON s.supplier_id = p.supplier_id
        LEFT JOIN order_details od ON p.product_id = od.product_id
        LEFT JOIN orders o ON od.order_id = o.order_id
        LEFT JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY s.supplier_id
        """
        
        df = pd.read_sql(query, db.connection())
        if df.empty:
            raise ValueError("No supplier data found in the database")
        
        missing_values = df.isnull().sum()
        if missing_values.any():
            df = df.fillna(0)
        
        return df
    except SQLAlchemyError as e:
        raise
    except Exception as e:
        raise

def find_optimal_eps_supplier(df: pd.DataFrame, min_samples: int = 2) -> float:
    """
    Finds the optimal eps value for supplier data.
    """
    try:
        features = df[['product_count', 'total_stock', 'total_on_order', 
                      'avg_product_price', 'unique_customers']]
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

def find_optimal_min_samples_supplier(df: pd.DataFrame) -> int:
    """
    Finds the optimal min_samples value for supplier data.
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

def plot_k_distance_supplier(df: pd.DataFrame, min_samples: int = 2) -> None:
    """
    Creates and saves the k-distance graph for supplier data.
    """
    try:
        features = df[['product_count', 'total_stock', 'total_on_order', 
                      'avg_product_price', 'unique_customers']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        nbrs = NearestNeighbors(n_neighbors=min_samples)
        nbrs.fit(scaled_features)
        distances, _ = nbrs.kneighbors(scaled_features)
        
        distances = np.sort(distances[:, -1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.title('K-Distance Graph for Supplier Data')
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
        
        plt.savefig('supplier_k_distance.png')
        plt.close()
    except Exception as e:
        pass

def perform_supplier_clustering(df: pd.DataFrame, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
    """
    Performs supplier clustering using DBSCAN.
    """
    try:
        features = df[['product_count', 'total_stock', 'total_on_order', 
                      'avg_product_price', 'unique_customers']]
        
        if features.isnull().any().any():
            features = features.fillna(0)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        if eps is None:
            eps = find_optimal_eps_supplier(df)
        if min_samples is None:
            min_samples = find_optimal_min_samples_supplier(df)
        
        plot_k_distance_supplier(df, min_samples)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_features)
        
        df['cluster'] = clusters
        results = {
            'suppliers': df.to_dict(orient='records'),
            'outliers': df[df['cluster'] == -1].to_dict(orient='records'),
            'cluster_stats': df.groupby('cluster').agg({
                'product_count': ['mean', 'std'],
                'total_stock': ['mean', 'std'],
                'total_on_order': ['mean', 'std'],
                'avg_product_price': ['mean', 'std'],
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

def analyze_supplier_clusters(db: Session, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
    """
    Performs supplier clustering analysis and returns results.
    """
    try:
        df = get_supplier_features(db)
        results = perform_supplier_clustering(df, eps, min_samples)
        return results
    except Exception as e:
        raise 