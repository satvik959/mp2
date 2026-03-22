"""
Packet Capture Module

Captures live network packets using Scapy and extracts relevant features.
"""

from scapy.all import sniff, IP, TCP, UDP, wrpcap, conf
import pandas as pd
import time
from typing import List, Dict, Optional
from datetime import datetime
import signal

STOP_CAPTURE = False

def handle_interrupt(signum, frame):
    global STOP_CAPTURE
    print("\n🛑 Stopping capture...")
    STOP_CAPTURE = True

signal.signal(signal.SIGINT, handle_interrupt)
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
                    
                info = packet.summary()
                
                return {
                    'timestamp': timestamp,
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'protocol': protocol_name,
                    'packet_size': packet_length,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'flags': str(flags) if flags else None,
                    'Info': info
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
    
    def capture_packets(self, duration: int = 5, interface: Optional[str] = None) -> pd.DataFrame:
        print(f"🔍 Starting packet capture...")
        print(f"   Duration: {duration}s")
        print(f"   Interface: {interface or 'default'}")
        
        self.packets_data = []
        self.start_time = time.time()
        conf.use_pcap = True

        try:
            start = time.time()

            # 🔥 KEY FIX: loop with small timeout
            while time.time() - start < duration and not STOP_CAPTURE:
                sniff(
                    iface=interface,
                    prn=self._packet_handler,
                    timeout=1,      # ✅ VERY IMPORTANT
                    store=False,
                    filter="ip",
                    promisc=True
                )

            print(f"✓ Captured {len(self.packets_data)} packets")

            if self.packets_data:
                return pd.DataFrame(self.packets_data)
            else:
                print("⚠️  No packets captured")
                return pd.DataFrame()

        except KeyboardInterrupt:
            print("\n🛑 Capture interrupted!")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str = 'data/captured_packets.csv') -> None:
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
    import sys

    print("🚀 Starting Packet Capture (Replace Mode)...")

    capture = PacketCapture()

    try:
        while not STOP_CAPTURE:
            df = capture.capture_packets(duration=5, interface="en0")

            if not df.empty:
                df.to_csv("data/captured_packets.csv", index=False)
                print(f"💾 Replaced CSV with {len(df)} packets")
            else:
                print("⚠️ No packets captured in this cycle")
        print("✅ Capture fully stopped")

    except KeyboardInterrupt:
        print("\n🛑 Capture stopped by user")
        sys.exit(0)