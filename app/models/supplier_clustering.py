from sqlalchemy.orm import Session
from typing import Dict, Any
from ..core.clustering import ClusterAnalyzer
from ..core.preprocessing import DataPreprocessor


SUPPLIER_FEATURES = [
    'product_count', 'total_stock', 'total_on_order',
    'avg_product_price', 'unique_customers'
]


def get_supplier_features(db: Session):
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
    preprocessor = DataPreprocessor(db)
    return preprocessor.load_and_clean(query)


def analyze_supplier_clusters(db: Session, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
    df = get_supplier_features(db)
    analyzer = ClusterAnalyzer(feature_columns=SUPPLIER_FEATURES, name='supplier')
    return analyzer.analyze(df, eps, min_samples)
