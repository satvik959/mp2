"""
Dataset Loader Module

Loads network traffic dataset from CSV file, validates structure,
and prepares data for preprocessing pipeline.
"""

import pandas as pd
import os
from typing import Optional, List


class DatasetLoader:
    """Handles loading and validation of network traffic datasets."""
    
    REQUIRED_COLUMNS = ['No', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
    
    def __init__(self, data_path: str = '../data/network_traffic_dataset.csv'):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Path to CSV dataset file
        """
        self.data_path = data_path
        self.dataframe: Optional[pd.DataFrame] = None
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load network traffic dataset from CSV file.
        
        Returns:
            Pandas DataFrame with network traffic data
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is invalid
        """
        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        
        print(f"📂 Loading dataset from: {self.data_path}")
        
        try:
            # Load CSV
            df = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(df)} packets")
            
            # Validate columns
            self._validate_columns(df)
            
            # Clean dataset
            df = self._clean_dataset(df)
            
            # Store internally
            self.dataframe = df
            
            print(f"✓ Dataset ready: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {str(e)}")
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that dataset has required columns.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"✓ Validated columns: {list(df.columns)}")
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset by removing corrupted or invalid rows.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)
        
        # Remove rows with missing critical fields
        df = df.dropna(subset=['Source', 'Destination', 'Protocol', 'Length'])
        
        # Remove duplicate packets
        df = df.drop_duplicates(subset=['No'], keep='first')
        
        # Ensure Length is numeric and positive
        df = df[pd.to_numeric(df['Length'], errors='coerce') > 0]
        
        # Ensure Time is numeric
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"⚠️  Removed {removed_rows} corrupted/invalid rows")
        
        return df
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.dataframe is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        df = self.dataframe
        
        return {
            'total_packets': len(df),
            'unique_sources': df['Source'].nunique(),
            'unique_destinations': df['Destination'].nunique(),
            'protocols': df['Protocol'].value_counts().to_dict(),
            'time_range': {
                'start': df['Time'].min(),
                'end': df['Time'].max(),
                'duration': df['Time'].max() - df['Time'].min()
            },
            'packet_sizes': {
                'min': df['Length'].min(),
                'max': df['Length'].max(),
                'mean': df['Length'].mean(),
                'median': df['Length'].median()
            }
        }
    
    def filter_by_protocol(self, protocols: List[str]) -> pd.DataFrame:
        """
        Filter dataset by specific protocols.
        
        Args:
            protocols: List of protocol names to keep
            
        Returns:
            Filtered DataFrame
        """
        if self.dataframe is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        filtered = self.dataframe[self.dataframe['Protocol'].isin(protocols)]
        print(f"📊 Filtered to {len(filtered)} packets with protocols: {protocols}")
        return filtered
    
    def filter_by_time_range(self, start_time: float, end_time: float) -> pd.DataFrame:
        """
        Filter dataset by time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Filtered DataFrame
        """
        if self.dataframe is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        filtered = self.dataframe[
            (self.dataframe['Time'] >= start_time) & 
            (self.dataframe['Time'] <= end_time)
        ]
        print(f"⏱️  Filtered to {len(filtered)} packets in time range [{start_time}, {end_time}]")
        return filtered
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get loaded DataFrame.
        
        Returns:
            Loaded DataFrame
        """
        if self.dataframe is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        return self.dataframe


def load_network_traffic(data_path: str = '../data/network_traffic_dataset.csv') -> pd.DataFrame:
    """
    Convenience function to quickly load network traffic dataset.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        Loaded and cleaned DataFrame
    """
    loader = DatasetLoader(data_path)
    return loader.load_dataset()


# Test/Demo code
if __name__ == "__main__":
    print("=" * 60)
    print("  NETWORK TRAFFIC DATASET LOADER - TEST")
    print("=" * 60)
    
    try:
        # Load dataset
        loader = DatasetLoader('../data/network_traffic_dataset.csv')
        df = loader.load_dataset()
        
        # Show summary
        print("\n📊 Dataset Summary:")
        summary = loader.get_summary()
        print(f"  Total packets: {summary['total_packets']}")
        print(f"  Unique sources: {summary['unique_sources']}")
        print(f"  Unique destinations: {summary['unique_destinations']}")
        print(f"  Time duration: {summary['time_range']['duration']:.2f} seconds")
        
        print("\n🔧 Protocol Distribution:")
        for protocol, count in list(summary['protocols'].items())[:10]:
            print(f"  {protocol:15s}: {count:4d} packets")
        
        print("\n📦 Packet Size Statistics:")
        print(f"  Min: {summary['packet_sizes']['min']:.0f} bytes")
        print(f"  Max: {summary['packet_sizes']['max']:.0f} bytes")
        print(f"  Mean: {summary['packet_sizes']['mean']:.2f} bytes")
        print(f"  Median: {summary['packet_sizes']['median']:.0f} bytes")
        
        # Show sample data
        print("\n📋 Sample Packets (first 5):")
        print(df.head())
        
        print("\n✓ Dataset loader test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
