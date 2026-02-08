from sqlalchemy.orm import Session
from typing import Dict, Any
from ..core.clustering import ClusterAnalyzer
from ..core.preprocessing import DataPreprocessor


PRODUCT_FEATURES = [
    'unit_price', 'units_in_stock', 'units_on_order', 'reorder_level',
    'order_count', 'total_quantity', 'unique_customers'
]


def get_product_features(db: Session):
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
    preprocessor = DataPreprocessor(db)
    return preprocessor.load_and_clean(query)


def analyze_product_clusters(db: Session, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
    df = get_product_features(db)
    analyzer = ClusterAnalyzer(feature_columns=PRODUCT_FEATURES, name='product')
    return analyzer.analyze(df, eps, min_samples)
