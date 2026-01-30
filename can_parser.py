#!/usr/bin/env python3
"""
CAN Bus Message Parser and Visualizer
Parses CAN messages in all possible interpretations and creates graphs
"""

import struct
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import re


def parse_can_log(log_text: str) -> List[Dict]:
    """Parse CAN log entries from text"""
    entries = []
    
    # Split by lines and parse each entry
    lines = log_text.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
            
        # Extract fields using regex
        time_match = re.search(r"'action_time':\s*(\d+)", line)
        data_match = re.search(r"'data':\s*'([^']+)'", line)
        frame_match = re.search(r"'frame_name':\s*'([^']+)'", line)
        
        if time_match and data_match and frame_match:
            entries.append({
                'action_time': int(time_match.group(1)),
                'data': data_match.group(1),
                'frame_name': frame_match.group(1)
            })
    
    return entries


def hex_string_to_bytes(hex_string: str) -> bytes:
    """Convert hex string like 'AA E8 27 04' to bytes"""
    hex_values = hex_string.strip().split()
    return bytes([int(h, 16) for h in hex_values])


def extract_all_interpretations(data_bytes: bytes, timestamp: int) -> Dict[str, List[Tuple[int, float]]]:
    """Extract all possible interpretations from CAN data bytes"""
    results = {}
    
    # Skip header (AA E8) and footer (55) if present
    # Assuming format: AA E8 [frame_id] [data...] 55
    if len(data_bytes) >= 3 and data_bytes[0] == 0xAA and data_bytes[1] == 0xE8:
        payload = data_bytes[3:-1] if data_bytes[-1] == 0x55 else data_bytes[3:]
    else:
        payload = data_bytes
    
    # Individual bytes (unsigned)
    for i, byte_val in enumerate(payload):
        key = f"Byte_{i}_uint8"
        if key not in results:
            results[key] = []
        results[key].append((timestamp, byte_val))
    
    # Individual bytes (signed)
    for i, byte_val in enumerate(payload):
        key = f"Byte_{i}_int8"
        if key not in results:
            results[key] = []
        signed_val = struct.unpack('b', bytes([byte_val]))[0]
        results[key].append((timestamp, signed_val))
    
    # 16-bit combinations
    for i in range(len(payload) - 1):
        chunk = payload[i:i+2]
        
        # uint16 little-endian
        key = f"Bytes_{i}_{i+1}_uint16le"
        if key not in results:
            results[key] = []
        val = struct.unpack('<H', chunk)[0]
        results[key].append((timestamp, val))
        
        # uint16 big-endian
        key = f"Bytes_{i}_{i+1}_uint16be"
        if key not in results:
            results[key] = []
        val = struct.unpack('>H', chunk)[0]
        results[key].append((timestamp, val))
        
        # int16 little-endian
        key = f"Bytes_{i}_{i+1}_int16le"
        if key not in results:
            results[key] = []
        val = struct.unpack('<h', chunk)[0]
        results[key].append((timestamp, val))
        
        # int16 big-endian
        key = f"Bytes_{i}_{i+1}_int16be"
        if key not in results:
            results[key] = []
        val = struct.unpack('>h', chunk)[0]
        results[key].append((timestamp, val))
    
    # 32-bit combinations
    for i in range(len(payload) - 3):
        chunk = payload[i:i+4]
        
        # uint32 little-endian
        key = f"Bytes_{i}_{i+3}_uint32le"
        if key not in results:
            results[key] = []
        val = struct.unpack('<I', chunk)[0]
        results[key].append((timestamp, val))
        
        # uint32 big-endian
        key = f"Bytes_{i}_{i+3}_uint32be"
        if key not in results:
            results[key] = []
        val = struct.unpack('>I', chunk)[0]
        results[key].append((timestamp, val))
        
        # int32 little-endian
        key = f"Bytes_{i}_{i+3}_int32le"
        if key not in results:
            results[key] = []
        val = struct.unpack('<i', chunk)[0]
        results[key].append((timestamp, val))
        
        # int32 big-endian
        key = f"Bytes_{i}_{i+3}_int32be"
        if key not in results:
            results[key] = []
        val = struct.unpack('>i', chunk)[0]
        results[key].append((timestamp, val))
        
        # float32 little-endian
        key = f"Bytes_{i}_{i+3}_float32le"
        if key not in results:
            results[key] = []
        try:
            val = struct.unpack('<f', chunk)[0]
            if not np.isnan(val) and not np.isinf(val):
                results[key].append((timestamp, val))
        except:
            pass
        
        # float32 big-endian
        key = f"Bytes_{i}_{i+3}_float32be"
        if key not in results:
            results[key] = []
        try:
            val = struct.unpack('>f', chunk)[0]
            if not np.isnan(val) and not np.isinf(val):
                results[key].append((timestamp, val))
        except:
            pass
    
    # 64-bit combinations
    for i in range(len(payload) - 7):
        chunk = payload[i:i+8]
        
        # uint64 little-endian
        key = f"Bytes_{i}_{i+7}_uint64le"
        if key not in results:
            results[key] = []
        val = struct.unpack('<Q', chunk)[0]
        results[key].append((timestamp, val))
        
        # uint64 big-endian
        key = f"Bytes_{i}_{i+7}_uint64be"
        if key not in results:
            results[key] = []
        val = struct.unpack('>Q', chunk)[0]
        results[key].append((timestamp, val))
        
        # float64 little-endian
        key = f"Bytes_{i}_{i+7}_float64le"
        if key not in results:
            results[key] = []
        try:
            val = struct.unpack('<d', chunk)[0]
            if not np.isnan(val) and not np.isinf(val):
                results[key].append((timestamp, val))
        except:
            pass
        
        # float64 big-endian
        key = f"Bytes_{i}_{i+7}_float64be"
        if key not in results:
            results[key] = []
        try:
            val = struct.unpack('>d', chunk)[0]
            if not np.isnan(val) and not np.isinf(val):
                results[key].append((timestamp, val))
        except:
            pass
    
    return results


def filter_varying_signals(all_interpretations: Dict[str, List[Tuple[int, float]]]) -> Dict[str, List[Tuple[int, float]]]:
    """Filter out signals that don't change (constant values)"""
    varying = {}
    
    for key, values in all_interpretations.items():
        if len(values) < 2:
            continue
        
        # Check if values vary
        vals = [v[1] for v in values]
        if len(set(vals)) > 1:  # More than one unique value
            varying[key] = values
    
    return varying


def create_plots(interpretations: Dict[str, List[Tuple[int, float]]], frame_name: str, output_file: str):
    """Create plots for all interpretations"""
    if not interpretations:
        print(f"No varying signals found for frame {frame_name}")
        return
    
    num_plots = len(interpretations)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f'CAN Frame {frame_name} - All Interpretations', fontsize=16, fontweight='bold')
    
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx, (name, values) in enumerate(sorted(interpretations.items())):
        timestamps = [v[0] for v in values]
        vals = [v[1] for v in values]
        
        # Normalize timestamps to start from 0
        min_time = min(timestamps)
        timestamps_norm = [(t - min_time) / 1000.0 for t in timestamps]  # Convert to seconds
        
        ax = axes[idx]
        ax.plot(timestamps_norm, vals, marker='o', linestyle='-', markersize=4)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add value range info
        val_range = max(vals) - min(vals)
        ax.text(0.02, 0.98, f'Range: {val_range:.2f}', 
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()


def main():
    # Example log data
    log_data = """
{'action_time': 1769369151120, 'data': 'AA E8 27 04 FF 18 00 7D 13 92 05 03 03 FF 55', 'frame_name': '2704'}
{'action_time': 1769369151132, 'data': 'AA E8 27 17 FF 18 81 0E 69 27 E6 3F 02 28 55', 'frame_name': '2717'}
{'action_time': 1769369151623, 'data': 'AA E8 27 04 FF 18 00 7D 13 8F 05 03 03 FF 55', 'frame_name': '2704'}
{'action_time': 1769369151635, 'data': 'AA E8 27 17 FF 18 81 0E 69 27 E6 3F 02 28 55', 'frame_name': '2717'}
{'action_time': 1769369152127, 'data': 'AA E8 27 04 FF 18 00 7D 13 8D 05 04 03 FF 55', 'frame_name': '2704'}
{'action_time': 1769369152141, 'data': 'AA E8 27 17 FF 18 81 0E 69 27 E6 3F 02 28 55', 'frame_name': '2717'}
{'action_time': 1769369152632, 'data': 'AA E8 27 04 FF 18 00 7D 13 8B 05 04 03 FF 55', 'frame_name': '2704'}
"""
    
    print("Parsing CAN log data...")
    entries = parse_can_log(log_data)
    print(f"Found {len(entries)} log entries")
    
    # Filter only frame 2704
    frame_2704 = [e for e in entries if e['frame_name'] == '2704']
    print(f"Found {len(frame_2704)} entries for frame 2704")
    
    # Collect all interpretations
    all_interpretations = {}
    
    for entry in frame_2704:
        data_bytes = hex_string_to_bytes(entry['data'])
        timestamp = entry['action_time']
        
        interpretations = extract_all_interpretations(data_bytes, timestamp)
        
        # Merge into all_interpretations
        for key, values in interpretations.items():
            if key not in all_interpretations:
                all_interpretations[key] = []
            all_interpretations[key].extend(values)
    
    print(f"Generated {len(all_interpretations)} different interpretations")
    
    # Filter only varying signals
    varying_interpretations = filter_varying_signals(all_interpretations)
    print(f"Found {len(varying_interpretations)} varying signals")
    
    if varying_interpretations:
        # Print some statistics
        print("\nVarying signals found:")
        for name, values in sorted(varying_interpretations.items()):
            vals = [v[1] for v in values]
            print(f"  {name}: min={min(vals):.2f}, max={max(vals):.2f}, range={max(vals)-min(vals):.2f}")
        
        # Create plots
        create_plots(varying_interpretations, '2704', '/mnt/user-data/outputs/can_frame_2704_analysis.png')
    else:
        print("No varying signals found!")


if __name__ == "__main__":
    main()
