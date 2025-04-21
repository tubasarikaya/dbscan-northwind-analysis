import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator
from app.models.customer_clustering import get_customer_features
from app.models.product_clustering import get_product_features
from app.models.supplier_clustering import get_supplier_features
from app.models.country_clustering import get_country_features
from app.database import get_db

def plot_k_distance(features, title, filename, min_samples=2):
    """
    Creates and saves the k-distance graph for the given features.
    """
    try:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        nbrs = NearestNeighbors(n_neighbors=min_samples)
        nbrs.fit(scaled_features)
        distances, _ = nbrs.kneighbors(scaled_features)
        
        distances = np.sort(distances[:, -1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.title(title)
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
        
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        pass

def main():
    db = next(get_db())
    try:
        # Customer data
        customer_df = get_customer_features(db)
        customer_features = customer_df[['order_count', 'total_quantity', 'avg_unit_price', 'unique_categories']]
        plot_k_distance(customer_features, 'K-Distance Graph for Customer Data', 'customer_k_distance.png')

        # Product data
        product_df = get_product_features(db)
        product_features = product_df[['unit_price', 'units_in_stock', 'units_on_order', 
                                     'reorder_level', 'order_count', 'total_quantity', 'avg_discount']]
        plot_k_distance(product_features, 'K-Distance Graph for Product Data', 'product_k_distance.png')

        # Supplier data
        supplier_df = get_supplier_features(db)
        supplier_features = supplier_df[['product_count', 'total_stock', 'total_on_order', 
                                       'avg_product_price', 'unique_customers']]
        plot_k_distance(supplier_features, 'K-Distance Graph for Supplier Data', 'supplier_k_distance.png')

        # Country data
        country_df = get_country_features(db)
        country_features = country_df[['order_count', 'total_quantity', 'avg_unit_price', 'unique_categories']]
        plot_k_distance(country_features, 'K-Distance Graph for Country Data', 'country_k_distance.png')
    finally:
        db.close()

if __name__ == "__main__":
    main() 