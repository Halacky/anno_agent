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


def create_individual_plots(categories: Dict, frame_name: str):
    """Create separate plot file for each signal"""
    from datetime import datetime
    
    plot_count = 0
    
    for category, signals in categories.items():
        if not signals:
            continue
            
        for name, values in sorted(signals.items()):
            timestamps = [v[0] for v in values]
            vals = [v[1] for v in values]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot data
            ax.plot(timestamps, vals, marker='o', linestyle='-', 
                   markersize=8, linewidth=2, color='#2E86AB')
            
            # Title
            ax.set_title(f'CAN Frame {frame_name} - {name}\n[{category.upper()}]', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Format X-axis with original timestamps
            # Create ticks every 10 seconds (10000 ms)
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_range = max_time - min_time
            
            # Generate ticks every 10 seconds
            if time_range > 0:
                tick_interval = 10000  # 10 seconds in milliseconds
                tick_positions = []
                tick_labels = []
                
                current_tick = min_time
                while current_tick <= max_time:
                    tick_positions.append(current_tick)
                    tick_labels.append(str(current_tick))
                    current_tick += tick_interval
                
                # If we don't have many ticks, add intermediate ones
                if len(tick_positions) < 3:
                    tick_positions = timestamps
                    tick_labels = [str(t) for t in timestamps]
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=90, ha='right')
            else:
                # If all timestamps are the same or very close
                ax.set_xticks(timestamps)
                ax.set_xticklabels([str(t) for t in timestamps], rotation=90, ha='right')
            
            # Statistics box
            val_range = max(vals) - min(vals)
            stats_text = (f'Min: {min(vals):.4f}\n'
                         f'Max: {max(vals):.4f}\n'
                         f'Range: {val_range:.4f}\n'
                         f'Samples: {len(vals)}')
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                           edgecolor='black', linewidth=1.5))
            
            # Improve layout
            plt.tight_layout()
            
            # Save to file
            # Clean filename from special characters
            safe_name = name.replace(':', '_').replace('/', '_')
            output_file = f'/mnt/user-data/outputs/can_{frame_name}_{category}_{safe_name}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plot_count += 1
            print(f"  Plot {plot_count}: {output_file}")
            plt.close()
    
    return plot_count


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
            
            # Create individual plots for each signal
            print(f"\nGenerating individual plots...")
            plot_count = create_individual_plots(categories, frame_name)
            print(f"Total plots created: {plot_count}")
        else:
            print("No varying signals found!")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file)
