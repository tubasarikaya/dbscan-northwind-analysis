from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from .database import get_db
from .models.customer_clustering import analyze_customer_clusters
from .models.product_clustering import analyze_product_clusters
from .models.supplier_clustering import analyze_supplier_clusters
from .models.country_clustering import analyze_country_clusters

app = FastAPI(
    title="DBSCAN Sales Analysis API",
    description="Sales data analysis and segmentation using DBSCAN"
)

@app.get("/")
async def root():
    return {"message": "Welcome to DBSCAN Sales Analysis API"}

@app.get("/api/customers/clusters", response_model=Dict[str, Any])
async def get_customer_clusters(
    eps: float = None,
    min_samples: int = None,
    db: Session = Depends(get_db)
):
    """
    Returns customer clustering results.
    
    Parameters:
    - eps: DBSCAN epsilon parameter
    - min_samples: DBSCAN minimum samples parameter
    
    Returns:
    - Customer clusters and statistics
    """
    try:
        results = analyze_customer_clusters(db, eps, min_samples)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products/clusters", response_model=Dict[str, Any])
async def get_product_clusters(
    eps: float = None,
    min_samples: int = None,
    db: Session = Depends(get_db)
):
    """
    Returns product clustering results.
    
    Parameters:
    - eps: DBSCAN epsilon parameter
    - min_samples: DBSCAN minimum samples parameter
    
    Returns:
    - Product clusters and statistics
    """
    try:
        results = analyze_product_clusters(db, eps, min_samples)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suppliers/clusters", response_model=Dict[str, Any])
async def get_supplier_clusters(
    eps: float = None,
    min_samples: int = None,
    db: Session = Depends(get_db)
):
    """
    Returns supplier clustering results.
    
    Parameters:
    - eps: DBSCAN epsilon parameter
    - min_samples: DBSCAN minimum samples parameter
    
    Returns:
    - Supplier clusters and statistics
    """
    try:
        results = analyze_supplier_clusters(db, eps, min_samples)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/countries/clusters", response_model=Dict[str, Any])
async def get_country_clusters(
    eps: float = None,
    min_samples: int = None,
    db: Session = Depends(get_db)
):
    """
    Returns country clustering results.
    
    Parameters:
    - eps: DBSCAN epsilon parameter
    - min_samples: DBSCAN minimum samples parameter
    
    Returns:
    - Country clusters and statistics
    """
    try:
        results = analyze_country_clusters(db, eps, min_samples)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 