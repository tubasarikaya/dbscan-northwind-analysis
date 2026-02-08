from sqlalchemy.orm import Session
from typing import Dict, Any
from ..core.clustering import ClusterAnalyzer
from ..core.preprocessing import DataPreprocessor


CUSTOMER_FEATURES = ['order_count', 'total_quantity', 'avg_unit_price', 'unique_categories']


def get_customer_features(db: Session):
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
    preprocessor = DataPreprocessor(db)
    return preprocessor.load_and_clean(query)


def analyze_customer_clusters(db: Session, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
    df = get_customer_features(db)
    analyzer = ClusterAnalyzer(feature_columns=CUSTOMER_FEATURES, name='customer')
    return analyzer.analyze(df, eps, min_samples)
