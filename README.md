# Customer Segmentation Analysis with DBSCAN

This project implements a customer segmentation analysis system using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. The system analyzes customer, product, supplier, and country data to identify meaningful clusters and outliers.

## Features

- Customer clustering based on order patterns
- Product clustering based on sales and inventory metrics
- Supplier clustering based on performance indicators
- Country clustering based on market characteristics
- Automatic parameter optimization for DBSCAN
- Visualization tools for cluster analysis

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following content:
```bash
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/your_database_name
```

   - Replace `your_password` with your PostgreSQL password
   - Replace `your_database_name` with your database name
   - If your database is hosted elsewhere, replace `localhost` with the host address
   - If using a different port, replace `5432` with your port number

4. Start the API:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `/api/customers/clusters` - Get customer clusters
- `/api/products/clusters` - Get product clusters
- `/api/suppliers/clusters` - Get supplier clusters
- `/api/countries/clusters` - Get country clusters

## R&D: DBSCAN Parameter Optimization

### Overview
The project includes an R&D component focused on optimizing DBSCAN parameters (eps and min_samples) for each data type. The optimization process uses the k-distance graph method to find the optimal eps value and a data-driven approach for min_samples.

### Methodology

1. **K-Distance Graph Analysis**
   - For each data type (customers, products, suppliers, countries), we:
     - Scale the features using StandardScaler
     - Calculate k-nearest neighbor distances
     - Plot the sorted distances
     - Use the KneeLocator algorithm to find the optimal eps value at the "elbow" point

2. **Min_samples Optimization**
   - Determined based on dataset size:
     - Small datasets (<100 samples): min_samples = 2
     - Medium datasets (100-1000 samples): min_samples = 3
     - Large datasets (>1000 samples): min_samples = 4

### Optimized Parameters

| Data Type | Optimal eps | Optimal min_samples |
|-----------|-------------|---------------------|
| Customers | 0.70        | 2                   |
| Products  | 2.36        | 2                   |
| Suppliers | 1.98        | 2                   |
| Countries | 1.25        | 3                   |

### Visualization Tools

The project includes two visualization scripts:
1. `plot_k_distance_graphs.py` - Generates k-distance graphs for parameter optimization
2. `plot_cluster_results.py` - Creates scatter plots of the clustering results

## Data Features

### Customer Features
- Order count
- Total quantity
- Average unit price
- Unique categories

### Product Features
- Unit price
- Units in stock
- Units on order
- Reorder level
- Order count
- Total quantity
- Average discount

### Supplier Features
- Product count
- Total stock
- Total on order
- Average product price
- Unique customers

### Country Features
- Order count
- Total quantity
- Average unit price
- Unique categories

## Dependencies

- FastAPI
- SQLAlchemy
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Kneed

