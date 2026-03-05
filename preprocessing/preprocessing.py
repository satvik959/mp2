"""
Data Preprocessing Module

Prepares network traffic dataset for machine learning models.
Handles encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class NetworkDataPreprocessor:
    """Preprocesses network traffic data for ML models."""
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler = StandardScaler()
        self.protocol_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.feature_columns: Optional[list] = None
        self.is_fitted = False
    
    def preprocess_data(self, dataset: pd.DataFrame, 
                       fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess network traffic dataset.
        
        Args:
            dataset: Raw dataset DataFrame
            fit: Whether to fit encoders/scalers (True for training, False for inference)
            
        Returns:
            Tuple of (X_scaled, y_encoded) - features and labels
        """
        print(f"🔧 Preprocessing {len(dataset)} records...")
        
        # Create a copy
        df = dataset.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode protocol
        if fit:
            df['protocol_encoded'] = self.protocol_encoder.fit_transform(df['protocol'])
        else:
            df['protocol_encoded'] = self.protocol_encoder.transform(df['protocol'])
        
        # Select features for ML
        self.feature_columns = [
            'packet_size',
            'packet_rate',
            'connection_count',
            'avg_packet_size',
            'protocol_encoded'
        ]
        
        # Extract features
        X = df[self.feature_columns].values
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise RuntimeError("Preprocessor not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        # Encode labels
        if 'label' in df.columns:
            if fit:
                y_encoded = self.label_encoder.fit_transform(df['label'])
            else:
                y_encoded = self.label_encoder.transform(df['label'])
        else:
            y_encoded = None
        
        print(f"✓ Preprocessed data shape: {X_scaled.shape}")
        print(f"   Features: {self.feature_columns}")
        
        return X_scaled, y_encoded
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        # Fill numeric columns with median
        numeric_cols = ['packet_size', 'packet_rate', 'connection_count', 'avg_packet_size']
        for col in numeric_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical with mode
        if 'protocol' in df.columns:
            df['protocol'].fillna(df['protocol'].mode()[0], inplace=True)
        
        return df
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Split data: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def get_label_mapping(self) -> dict:
        """Get mapping of encoded labels to original labels."""
        if not self.is_fitted:
            return {}
        return dict(enumerate(self.label_encoder.classes_))


# Test/Demo code
if __name__ == "__main__":
    print("=" * 60)
    print("  DATA PREPROCESSING MODULE - TEST")
    print("=" * 60)
    
    # Load processed dataset
    try:
        df = pd.read_csv('../data/processed_dataset.csv')
        print(f"\n📂 Loaded dataset: {df.shape}")
        
        # Initialize preprocessor
        preprocessor = NetworkDataPreprocessor()
        
        # Preprocess
        X, y = preprocessor.preprocess_data(df, fit=True)
        
        print(f"\n📊 Preprocessed Data:")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape if y is not None else 'None'}")
        print(f"   Label mapping: {preprocessor.get_label_mapping()}")
        
        # Split data
        if y is not None:
            X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
            
            print(f"\n✓ Data ready for training!")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Testing samples: {len(X_test)}")
        
    except FileNotFoundError:
        print("\n⚠️  Dataset not found. Run dataset_builder.py first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
