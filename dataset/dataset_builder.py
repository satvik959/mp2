"""
Dataset Builder Module

Converts captured network packets into structured dataset for ML training.
Computes traffic statistics and aggregates features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class DatasetBuilder:
    """Builds ML-ready dataset from captured network traffic."""
    
    def __init__(self):
        """Initialize dataset builder."""
        self.dataset: pd.DataFrame = None
    
    def build_dataset(self, packet_df: pd.DataFrame, window_size: float = 1.0) -> pd.DataFrame:
        """
        Convert packet DataFrame into aggregated dataset with traffic statistics.
        
        Args:
            packet_df: DataFrame with raw packet data
            window_size: Time window for aggregation in seconds
            
        Returns:
            Processed dataset with computed features
        """
        print(f"🔨 Building dataset from {len(packet_df)} packets...")
        
        if packet_df.empty:
            print("⚠️  Empty packet DataFrame")
            return pd.DataFrame()
        
        # Sort by timestamp
        packet_df = packet_df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute additional features
        dataset_rows = []
        
        for idx, row in packet_df.iterrows():
            # Get time window
            current_time = row['timestamp']
            window_start = current_time - window_size
            
            # Get packets in current window
            window_packets = packet_df[
                (packet_df['timestamp'] >= window_start) & 
                (packet_df['timestamp'] <= current_time)
            ]
            
            # Compute traffic statistics
            packet_rate = len(window_packets) / window_size
            
            # Count connections
            connection_count = len(window_packets[['src_ip', 'dst_ip']].drop_duplicates())
            
            # Average packet size
            avg_packet_size = window_packets['packet_size'].mean()
            
            # Protocol distribution
            protocol_counts = window_packets['protocol'].value_counts()
            dominant_protocol = protocol_counts.index[0] if len(protocol_counts) > 0 else row['protocol']
            
            # Build row
            dataset_row = {
                'timestamp': row['timestamp'],
                'src_ip': row['src_ip'],
                'dst_ip': row['dst_ip'],
                'protocol': row['protocol'],
                'packet_size': row['packet_size'],
                'src_port': row.get('src_port'),
                'dst_port': row.get('dst_port'),
                'packet_rate': packet_rate,
                'connection_count': connection_count,
                'avg_packet_size': avg_packet_size,
                'dominant_protocol': dominant_protocol,
                'label': 'normal'  # Placeholder - to be labeled later
            }
            
            dataset_rows.append(dataset_row)
        
        # Create DataFrame
        dataset = pd.DataFrame(dataset_rows)
        
        print(f"✓ Built dataset with {len(dataset)} rows and {len(dataset.columns)} features")
        
        self.dataset = dataset
        return dataset
    
    def add_labels(self, dataset: pd.DataFrame, anomaly_ips: List[str] = None) -> pd.DataFrame:
        """
        Add anomaly labels to dataset based on known anomalous IPs.
        
        Args:
            dataset: Dataset DataFrame
            anomaly_ips: List of IP addresses known to be anomalous
            
        Returns:
            Dataset with updated labels
        """
        if anomaly_ips:
            dataset['label'] = dataset.apply(
                lambda row: 'attack' if row['src_ip'] in anomaly_ips or row['dst_ip'] in anomaly_ips else 'normal',
                axis=1
            )
            print(f"✓ Added labels: {dataset['label'].value_counts().to_dict()}")
        
        return dataset
    
    def save_dataset(self, dataset: pd.DataFrame, filepath: str = '../data/processed_dataset.csv') -> None:
        """
        Save processed dataset to CSV file.
        
        Args:
            dataset: Dataset to save
            filepath: Output file path
        """
        try:
            dataset.to_csv(filepath, index=False)
            print(f"💾 Dataset saved to: {filepath}")
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    def get_summary(self, dataset: pd.DataFrame) -> Dict:
        """
        Get summary statistics of dataset.
        
        Args:
            dataset: Dataset to summarize
            
        Returns:
            Dictionary with statistics
        """
        return {
            'total_rows': len(dataset),
            'time_range': {
                'start': dataset['timestamp'].min(),
                'end': dataset['timestamp'].max(),
                'duration': dataset['timestamp'].max() - dataset['timestamp'].min()
            },
            'protocols': dataset['protocol'].value_counts().to_dict(),
            'labels': dataset['label'].value_counts().to_dict(),
            'unique_sources': dataset['src_ip'].nunique(),
            'unique_destinations': dataset['dst_ip'].nunique(),
            'traffic_stats': {
                'avg_packet_rate': dataset['packet_rate'].mean(),
                'max_packet_rate': dataset['packet_rate'].max(),
                'avg_connection_count': dataset['connection_count'].mean()
            }
        }


# Test/Demo code
if __name__ == "__main__":
    print("=" * 60)
    print("  DATASET BUILDER MODULE - TEST")
    print("=" * 60)
    
    # Load sample packet data (from dataset_loader)
    from dataset_loader import load_network_traffic
    
    try:
        # Load raw packets
        print("\n📂 Loading sample network traffic...")
        raw_packets = load_network_traffic()
        
        # Create sample packet DataFrame format
        sample_packets = pd.DataFrame({
            'timestamp': raw_packets['Time'].values,
            'src_ip': raw_packets['Source'].values,
            'dst_ip': raw_packets['Destination'].values,
            'protocol': raw_packets['Protocol'].values,
            'packet_size': raw_packets['Length'].values,
            'src_port': None,
            'dst_port': None
        })
        
        # Build dataset
        builder = DatasetBuilder()
        dataset = builder.build_dataset(sample_packets, window_size=1.0)
        
        # Get summary
        summary = builder.get_summary(dataset)
        
        print("\n📊 Dataset Summary:")
        print(f"   Total rows: {summary['total_rows']}")
        print(f"   Duration: {summary['time_range']['duration']:.2f}s")
        print(f"   Unique sources: {summary['unique_sources']}")
        print(f"   Avg packet rate: {summary['traffic_stats']['avg_packet_rate']:.2f} packets/s")
        
        print("\n📋 Sample Dataset Rows:")
        print(dataset.head())
        
        # Save
        builder.save_dataset(dataset)
        
        print("\n✓ Dataset builder test completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
