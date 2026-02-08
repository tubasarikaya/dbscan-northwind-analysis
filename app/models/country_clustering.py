from sqlalchemy.orm import Session
from typing import Dict, Any
from ..core.clustering import ClusterAnalyzer
from ..core.preprocessing import DataPreprocessor


COUNTRY_FEATURES = ['order_count', 'total_quantity', 'avg_unit_price', 'unique_categories']


def get_country_features(db: Session):
    query = """
        SELECT 
            c.country,
            COUNT(DISTINCT o.order_id) as order_count,
            COALESCE(SUM(od.quantity), 0) as total_quantity,
            COALESCE(AVG(od.unit_price), 0) as avg_unit_price,
            COUNT(DISTINCT p.category_id) as unique_categories
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        LEFT JOIN order_details od ON o.order_id = od.order_id
        LEFT JOIN products p ON od.product_id = p.product_id
        GROUP BY c.country
    """
    preprocessor = DataPreprocessor(db)
    return preprocessor.load_and_clean(query)


def analyze_country_clusters(db: Session, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
    df = get_country_features(db)
    analyzer = ClusterAnalyzer(feature_columns=COUNTRY_FEATURES, name='country')
    return analyzer.analyze(df, eps, min_samples)
