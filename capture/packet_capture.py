"""
Packet Capture Module

Captures live network packets using Scapy and extracts relevant features.
"""

from scapy.all import sniff, IP, TCP, UDP, wrpcap
import pandas as pd
import time
from typing import List, Dict, Optional
from datetime import datetime


class PacketCapture:
    """Captures and processes network packets in real-time."""
    
    def __init__(self):
        """Initialize packet capture."""
        self.packets_data: List[Dict] = []
        self.start_time: Optional[float] = None
    
    def _extract_packet_info(self, packet) -> Optional[Dict]:
        """
        Extract relevant information from captured packet.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Dictionary with packet features or None if invalid
        """
        try:
            if IP in packet:
                # Get timestamp
                timestamp = time.time() if self.start_time is None else time.time() - self.start_time
                
                # Extract IP layer info
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                protocol = packet[IP].proto
                packet_length = len(packet)
                
                # Extract transport layer info
                src_port = None
                dst_port = None
                flags = None
                
                if TCP in packet:
                    protocol_name = 'TCP'
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                    flags = packet[TCP].flags
                elif UDP in packet:
                    protocol_name = 'UDP'
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                else:
                    protocol_name = f'IP-{protocol}'
                
                return {
                    'timestamp': timestamp,
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'protocol': protocol_name,
                    'packet_size': packet_length,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'flags': str(flags) if flags else None
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error extracting packet: {e}")
            return None
    
    def _packet_handler(self, packet):
        """
        Callback function for each captured packet.
        
        Args:
            packet: Scapy packet object
        """
        packet_info = self._extract_packet_info(packet)
        if packet_info:
            self.packets_data.append(packet_info)
    
    def capture_packets(self, duration: int = 60, interface: Optional[str] = None, 
                       count: Optional[int] = None) -> pd.DataFrame:
        """
        Capture network packets for specified duration.
        
        Args:
            duration: Capture duration in seconds (default: 60)
            interface: Network interface to capture on (None for default)
            count: Maximum number of packets to capture (None for unlimited)
            
        Returns:
            DataFrame with captured packet data
        """
        print(f"🔍 Starting packet capture...")
        print(f"   Duration: {duration}s")
        print(f"   Interface: {interface or 'default'}")
        
        self.packets_data = []
        self.start_time = time.time()
        
        try:
            # Capture packets
            sniff(
                iface=interface,
                prn=self._packet_handler,
                timeout=duration,
                count=count,
                store=False
            )
            
            print(f"✓ Captured {len(self.packets_data)} packets")
            
            # Convert to DataFrame
            if self.packets_data:
                df = pd.DataFrame(self.packets_data)
                return df
            else:
                print("⚠️  No packets captured")
                return pd.DataFrame()
                
        except PermissionError:
            print("❌ Permission denied. Run with sudo/administrator privileges.")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Capture error: {e}")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str = '../data/captured_packets.csv') -> None:
        """
        Save captured packets to CSV file.
        
        Args:
            df: DataFrame with packet data
            filepath: Output CSV file path
        """
        try:
            df.to_csv(filepath, index=False)
            print(f"💾 Saved to: {filepath}")
        except Exception as e:
            print(f"❌ Save error: {e}")


# Demo/Test code
if __name__ == "__main__":
    print("=" * 60)
    print("  PACKET CAPTURE MODULE - TEST")
    print("=" * 60)
    print("\n⚠️  Note: This requires root/admin privileges")
    print("   Run with: sudo python packet_capture.py\n")
    
    # Create capture instance
    capture = PacketCapture()
    
    # Capture for 10 seconds (adjust as needed)
    df = capture.capture_packets(duration=10)
    
    if not df.empty:
        print(f"\n📊 Captured Packets Summary:")
        print(f"   Total: {len(df)}")
        print(f"   Protocols: {df['protocol'].value_counts().to_dict()}")
        print(f"\n📋 Sample Data:")
        print(df.head())
        
        # Save to file
        capture.save_to_csv(df)
    else:
        print("\n⚠️  No packets captured. Check permissions and network activity.")
