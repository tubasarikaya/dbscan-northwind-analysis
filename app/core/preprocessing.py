import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError


class DataPreprocessor:
    def __init__(self, db: Session):
        self.db = db
        self.scaler = StandardScaler()
    
    def load_and_clean(self, query: str) -> pd.DataFrame:
        df = pd.read_sql(query, self.db.connection())
        
        if df.empty:
            raise ValueError("Query returned no data")
        
        if df.isnull().any().any():
            df = df.fillna(0)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        features = df[feature_columns]
        
        if features.isnull().any().any():
            features = features.fillna(0)
        
        scaled = self.scaler.fit_transform(features)
        return pd.DataFrame(scaled, columns=feature_columns, index=df.index)
