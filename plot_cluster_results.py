import matplotlib.pyplot as plt
import seaborn as sns
from app.models.customer_clustering import get_customer_features
from app.models.product_clustering import get_product_features
from app.models.supplier_clustering import get_supplier_features
from app.models.country_clustering import get_country_features
from app.database import get_db

def plot_customer_clusters():
    db = next(get_db())
    try:
        df = get_customer_features(db)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='order_count', y='total_quantity', 
                       hue='cluster_label', palette='viridis', s=100)
        plt.title('Customer Clusters by Order Count and Total Quantity')
        plt.xlabel('Order Count')
        plt.ylabel('Total Quantity')
        plt.savefig('customer_clusters.png')
        plt.close()
    finally:
        db.close()

def plot_product_clusters():
    db = next(get_db())
    try:
        df = get_product_features(db)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='order_count', y='total_quantity', 
                       hue='cluster_label', palette='viridis', s=100)
        plt.title('Product Clusters by Order Count and Total Quantity')
        plt.xlabel('Order Count')
        plt.ylabel('Total Quantity')
        plt.savefig('product_clusters.png')
        plt.close()
    finally:
        db.close()

def plot_supplier_clusters():
    db = next(get_db())
    try:
        df = get_supplier_features(db)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='product_count', y='total_stock', 
                       hue='cluster_label', palette='viridis', s=100)
        plt.title('Supplier Clusters by Product Count and Total Stock')
        plt.xlabel('Product Count')
        plt.ylabel('Total Stock')
        plt.savefig('supplier_clusters.png')
        plt.close()
    finally:
        db.close()

def plot_country_clusters():
    db = next(get_db())
    try:
        df = get_country_features(db)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='order_count', y='total_quantity', 
                       hue='cluster_label', palette='viridis', s=100)
        plt.title('Country Clusters by Order Count and Total Quantity')
        plt.xlabel('Order Count')
        plt.ylabel('Total Quantity')
        plt.savefig('country_clusters.png')
        plt.close()
    finally:
        db.close()

if __name__ == "__main__":
    plot_customer_clusters()
    plot_product_clusters()
    plot_supplier_clusters()
    plot_country_clusters() 