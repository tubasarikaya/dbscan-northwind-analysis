# DBSCAN Customer Segmentation Analysis

A FastAPI-based clustering system that applies DBSCAN to analyze customer behavior, product performance, supplier metrics, and country-level sales patterns using the Northwind database.

## Contributors

Tuba SARIKAYA & Gül ERTEN -- GYK1

## Overview

This project uses DBSCAN clustering to identify patterns and outliers in sales data across four dimensions:
- Customer purchasing behavior
- Product sales and inventory patterns
- Supplier performance metrics
- Country-level market characteristics

The system automatically optimizes clustering parameters using k-distance graphs and provides visualization tools for analysis.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with your database connection:

```bash
DATABASE_URL=postgresql://postgres:password@localhost:5432/northwind
```

Run the API:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/customers/clusters` | Customer clustering results |
| `GET /api/products/clusters` | Product clustering results |
| `GET /api/suppliers/clusters` | Supplier clustering results |
| `GET /api/countries/clusters` | Country clustering results |

All endpoints accept optional query parameters:
- `eps` - DBSCAN epsilon (neighborhood radius)
- `min_samples` - Minimum points to form a cluster

## Parameter Optimization

The system uses k-distance graphs with the elbow method to find optimal DBSCAN parameters for each dataset.

### Default Parameters

| Data Type | eps | min_samples |
|-----------|-----|-------------|
| Customers | 0.70 | 2 |
| Products | 2.36 | 2 |
| Suppliers | 1.98 | 2 |
| Countries | 1.25 | 3 |

Parameters are automatically calculated based on:
- K-nearest neighbor distance analysis
- Dataset size (affects min_samples selection)
- Elbow point detection using KneeLocator

### Visualization Scripts

Two utility scripts are included for analysis:

1. `plot_k_distance_graphs.py` - Generate k-distance graphs for parameter tuning
2. `plot_cluster_results.py` - Visualize clustering results in 2D/3D space

## Features Used

### Customer Analysis
- Order count
- Total quantity purchased
- Average unit price
- Product category diversity

### Product Analysis
- Unit price
- Stock levels (in stock, on order)
- Reorder level
- Order frequency
- Total quantity sold
- Unique customer count

### Supplier Analysis
- Product count
- Total inventory (stock + orders)
- Average product price
- Unique customer reach

### Country Analysis
- Order volume
- Total quantity
- Average unit price
- Category diversity

## Tech Stack

- **FastAPI** - REST API framework
- **SQLAlchemy** - Database ORM
- **Pandas** - Data manipulation
- **Scikit-learn** - DBSCAN clustering
- **Matplotlib/Seaborn** - Visualization
- **Kneed** - Elbow detection
- **PostgreSQL** - Database

## Project Structure

```
app/
├── core/              # Core clustering logic
│   ├── clustering.py     # DBSCAN implementation
│   ├── optimization.py   # Parameter tuning
│   ├── preprocessing.py  # Data cleaning
│   └── visualization.py  # K-distance graphs
├── models/               # Domain-specific clustering
│   ├── customer_clustering.py
│   ├── product_clustering.py
│   ├── supplier_clustering.py
│   └── country_clustering.py
├── database.py          # Database connection
└── main.py              # FastAPI application
```
