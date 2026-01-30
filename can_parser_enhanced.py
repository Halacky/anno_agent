#!/usr/bin/env python3
"""
Enhanced CAN Bus Message Parser and Visualizer
- Reads from input file or uses embedded test data
- Generates comprehensive analysis report
- Creates organized visualization plots
"""

import struct
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import re
import sys


def parse_can_log(log_text: str) -> List[Dict]:
    """Parse CAN log entries from text"""
    entries = []
    
    lines = log_text.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
            
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
    """Convert hex string to bytes"""
    hex_values = hex_string.strip().split()
    return bytes([int(h, 16) for h in hex_values])


def extract_all_interpretations(data_bytes: bytes, timestamp: int) -> Dict[str, List[Tuple[int, float]]]:
    """Extract all possible interpretations from CAN data bytes"""
    results = {}
    
    # Skip header (AA E8) and footer (55) if present
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
        
        for suffix, fmt in [('uint16le', '<H'), ('uint16be', '>H'), 
                            ('int16le', '<h'), ('int16be', '>h')]:
            key = f"Bytes_{i}_{i+1}_{suffix}"
            if key not in results:
                results[key] = []
            val = struct.unpack(fmt, chunk)[0]
            results[key].append((timestamp, val))
    
    # 32-bit combinations
    for i in range(len(payload) - 3):
        chunk = payload[i:i+4]
        
        for suffix, fmt in [('uint32le', '<I'), ('uint32be', '>I'),
                            ('int32le', '<i'), ('int32be', '>i'),
                            ('float32le', '<f'), ('float32be', '>f')]:
            key = f"Bytes_{i}_{i+3}_{suffix}"
            if key not in results:
                results[key] = []
            try:
                val = struct.unpack(fmt, chunk)[0]
                if 'float' in suffix and (np.isnan(val) or np.isinf(val)):
                    continue
                results[key].append((timestamp, val))
            except:
                pass
    
    # 64-bit combinations
    for i in range(len(payload) - 7):
        chunk = payload[i:i+8]
        
        for suffix, fmt in [('uint64le', '<Q'), ('uint64be', '>Q'),
                            ('float64le', '<d'), ('float64be', '>d')]:
            key = f"Bytes_{i}_{i+7}_{suffix}"
            if key not in results:
                results[key] = []
            try:
                val = struct.unpack(fmt, chunk)[0]
                if 'float' in suffix and (np.isnan(val) or np.isinf(val)):
                    continue
                results[key].append((timestamp, val))
            except:
                pass
    
    return results


def filter_varying_signals(all_interpretations: Dict[str, List[Tuple[int, float]]]) -> Dict[str, List[Tuple[int, float]]]:
    """Filter out constant signals"""
    varying = {}
    
    for key, values in all_interpretations.items():
        if len(values) < 2:
            continue
        
        vals = [v[1] for v in values]
        if len(set(vals)) > 1:
            varying[key] = values
    
    return varying


def categorize_signals(interpretations: Dict[str, List[Tuple[int, float]]]) -> Dict[str, Dict]:
    """Categorize signals by their characteristics"""
    categories = {
        'counters': {},  # Monotonically increasing/decreasing
        'digital': {},   # Only 0/1 or small discrete values
        'analog': {},    # Continuous-looking values
        'other': {}
    }
    
    for name, values in interpretations.items():
        vals = [v[1] for v in values]
        unique_vals = set(vals)
        
        # Check if counter (monotonic)
        is_increasing = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        is_decreasing = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        
        if is_increasing or is_decreasing:
            categories['counters'][name] = values
        elif len(unique_vals) <= 5 and all(v == int(v) for v in unique_vals):
            categories['digital'][name] = values
        elif 'float' in name or max(vals) - min(vals) > 100:
            categories['analog'][name] = values
        else:
            categories['other'][name] = values
    
    return categories


def create_summary_report(entries: List[Dict], varying: Dict, categories: Dict, 
                         output_file: str, frame_name: str):
    """Create a text summary report"""
    with open(output_file, 'w') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"CAN Frame {frame_name} Analysis Report\n")
        f.write(f"=" * 80 + "\n\n")
        
        f.write(f"Total messages analyzed: {len(entries)}\n")
        f.write(f"Time span: {entries[0]['action_time']} - {entries[-1]['action_time']}\n")
        f.write(f"Duration: {(entries[-1]['action_time'] - entries[0]['action_time'])/1000:.2f} seconds\n\n")
        
        # Raw data
        f.write("Raw Messages:\n")
        f.write("-" * 80 + "\n")
        for entry in entries:
            f.write(f"Time: {entry['action_time']}, Data: {entry['data']}\n")
        f.write("\n")
        
        # Signal summary
        f.write(f"Signal Summary:\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"Total varying signals found: {len(varying)}\n\n")
        
        for category, signals in categories.items():
            if signals:
                f.write(f"\n{category.upper()} ({len(signals)} signals):\n")
                for name, values in sorted(signals.items()):
                    vals = [v[1] for v in values]
                    f.write(f"  {name}:\n")
                    f.write(f"    Min: {min(vals):.6f}\n")
                    f.write(f"    Max: {max(vals):.6f}\n")
                    f.write(f"    Range: {max(vals) - min(vals):.6f}\n")
                    f.write(f"    Unique values: {len(set(vals))}\n")
                    f.write(f"    Values: {vals}\n")
                    f.write("\n")


def create_categorized_plots(categories: Dict, frame_name: str):
    """Create separate plots for each category"""
    
    for category, signals in categories.items():
        if not signals:
            continue
            
        num_plots = len(signals)
        if num_plots == 0:
            continue
            
        cols = min(4, num_plots)
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        fig.suptitle(f'CAN Frame {frame_name} - {category.upper()} Signals', 
                     fontsize=16, fontweight='bold')
        
        if num_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten() if rows > 1 or cols > 1 else axes
        
        for idx, (name, values) in enumerate(sorted(signals.items())):
            timestamps = [v[0] for v in values]
            vals = [v[1] for v in values]
            
            min_time = min(timestamps)
            timestamps_norm = [(t - min_time) / 1000.0 for t in timestamps]
            
            ax = axes[idx] if num_plots > 1 else axes[0]
            ax.plot(timestamps_norm, vals, marker='o', linestyle='-', 
                   markersize=6, linewidth=2)
            ax.set_title(name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Statistics box
            val_range = max(vals) - min(vals)
            stats_text = f'Min: {min(vals):.2f}\nMax: {max(vals):.2f}\nRange: {val_range:.2f}'
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Hide unused subplots
        for idx in range(num_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_file = f'/mnt/user-data/outputs/can_frame_{frame_name}_{category}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_file}")
        plt.close()


def main(input_file=None):
    # Default test data
    default_log = """
{'action_time': 1769369151120, 'data': 'AA E8 27 04 FF 18 00 7D 13 92 05 03 03 FF 55', 'frame_name': '2704'}
{'action_time': 1769369151132, 'data': 'AA E8 27 17 FF 18 81 0E 69 27 E6 3F 02 28 55', 'frame_name': '2717'}
{'action_time': 1769369151623, 'data': 'AA E8 27 04 FF 18 00 7D 13 8F 05 03 03 FF 55', 'frame_name': '2704'}
{'action_time': 1769369151635, 'data': 'AA E8 27 17 FF 18 81 0E 69 27 E6 3F 02 28 55', 'frame_name': '2717'}
{'action_time': 1769369152127, 'data': 'AA E8 27 04 FF 18 00 7D 13 8D 05 04 03 FF 55', 'frame_name': '2704'}
{'action_time': 1769369152141, 'data': 'AA E8 27 17 FF 18 81 0E 69 27 E6 3F 02 28 55', 'frame_name': '2717'}
{'action_time': 1769369152632, 'data': 'AA E8 27 04 FF 18 00 7D 13 8B 05 04 03 FF 55', 'frame_name': '2704'}
"""
    
    # Read input
    if input_file:
        with open(input_file, 'r') as f:
            log_data = f.read()
    else:
        log_data = default_log
    
    print("=" * 80)
    print("CAN Bus Message Analyzer")
    print("=" * 80)
    print(f"\nParsing CAN log data...")
    
    entries = parse_can_log(log_data)
    print(f"Found {len(entries)} total log entries")
    
    # Analyze each frame separately
    frame_names = set(e['frame_name'] for e in entries)
    
    for frame_name in sorted(frame_names):
        print(f"\n{'='*80}")
        print(f"Analyzing Frame: {frame_name}")
        print(f"{'='*80}")
        
        frame_entries = [e for e in entries if e['frame_name'] == frame_name]
        print(f"Messages: {len(frame_entries)}")
        
        # Collect interpretations
        all_interpretations = {}
        for entry in frame_entries:
            data_bytes = hex_string_to_bytes(entry['data'])
            timestamp = entry['action_time']
            interpretations = extract_all_interpretations(data_bytes, timestamp)
            
            for key, values in interpretations.items():
                if key not in all_interpretations:
                    all_interpretations[key] = []
                all_interpretations[key].extend(values)
        
        print(f"Generated {len(all_interpretations)} interpretations")
        
        # Filter varying signals
        varying = filter_varying_signals(all_interpretations)
        print(f"Varying signals: {len(varying)}")
        
        if varying:
            # Categorize signals
            categories = categorize_signals(varying)
            
            print(f"\nSignal Categories:")
            for cat, sigs in categories.items():
                print(f"  {cat}: {len(sigs)}")
            
            # Create report
            report_file = f'/mnt/user-data/outputs/can_frame_{frame_name}_report.txt'
            create_summary_report(frame_entries, varying, categories, report_file, frame_name)
            print(f"\nReport saved: {report_file}")
            
            # Create plots
            create_categorized_plots(categories, frame_name)
        else:
            print("No varying signals found!")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file)
