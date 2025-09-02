#!/usr/bin/env python3
"""
Script to analyze router outputs from validation logs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def load_router_outputs(filepath):
    """Load router outputs from JSONL file."""
    outputs = []
    with open(filepath, 'r') as f:
        for line in f:
            outputs.append(json.loads(line.strip()))
    return outputs


def analyze_precision_evolution(outputs):
    """Analyze how precision selection evolves over epochs."""
    epochs = [entry['epoch'] for entry in outputs]
    num_layers = len(outputs[0]['layers'])
    
    # Extract precision for each layer across epochs
    layer_precisions = {}
    for layer_idx in range(num_layers):
        layer_precisions[layer_idx] = []
        for entry in outputs:
            precision = entry['layers'][layer_idx]['selected_precision']
            layer_precisions[layer_idx].append(precision)
    
    return epochs, layer_precisions


def plot_precision_evolution(epochs, layer_precisions, save_path=None):
    """Plot how precision selection evolves over training."""
    num_layers = len(layer_precisions)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Individual layer precision evolution
    for layer_idx in range(min(10, num_layers)):  # Limit to first 10 layers for readability
        precisions = layer_precisions[layer_idx]
        ax1.plot(epochs, precisions, marker='o', label=f'Layer {layer_idx}', alpha=0.7)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Selected Precision (bits)')
    ax1.set_title('Precision Evolution by Layer')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average precision evolution
    avg_precisions = []
    for epoch_idx in range(len(epochs)):
        epoch_precisions = [layer_precisions[layer_idx][epoch_idx] for layer_idx in range(num_layers)]
        avg_precisions.append(np.mean(epoch_precisions))
    
    ax2.plot(epochs, avg_precisions, marker='o', linewidth=2, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Precision (bits)')
    ax2.set_title('Average Precision Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Precision evolution plot saved: {save_path}")
    
    return fig


def analyze_precision_distribution(outputs):
    """Analyze precision distribution across layers and epochs."""
    final_epoch = outputs[-1]
    layers = final_epoch['layers']
    
    # Count precision usage
    precision_counts = {}
    for layer in layers:
        prec = layer['selected_precision']
        precision_counts[prec] = precision_counts.get(prec, 0) + 1
    
    print(f"\n📊 Final Epoch Precision Distribution:")
    print(f"{'Precision':<10} {'Count':<8} {'Percentage':<10}")
    print("-" * 30)
    
    total_layers = len(layers)
    for prec in sorted(precision_counts.keys()):
        count = precision_counts[prec]
        percentage = (count / total_layers) * 100
        print(f"{prec}-bit{'':<5} {count:<8} {percentage:.1f}%")
    
    return precision_counts


def analyze_layer_patterns(outputs):
    """Analyze which layers prefer which precisions."""
    num_layers = len(outputs[0]['layers'])
    num_epochs = len(outputs)
    
    # Calculate most common precision for each layer
    layer_preferences = {}
    
    for layer_idx in range(num_layers):
        layer_precisions = []
        for entry in outputs:
            precision = entry['layers'][layer_idx]['selected_precision']
            layer_precisions.append(precision)
        
        # Find most common precision
        from collections import Counter
        precision_counter = Counter(layer_precisions)
        most_common_prec, count = precision_counter.most_common(1)[0]
        consistency = (count / num_epochs) * 100
        
        layer_preferences[layer_idx] = {
            'preferred_precision': most_common_prec,
            'consistency': consistency,
            'all_precisions': layer_precisions
        }
    
    print(f"\n🎯 Layer Precision Preferences:")
    print(f"{'Layer':<8} {'Preferred':<12} {'Consistency':<12} {'Pattern'}")
    print("-" * 50)
    
    for layer_idx in range(min(20, num_layers)):  # Show first 20 layers
        pref = layer_preferences[layer_idx]
        preferred = f"{pref['preferred_precision']}-bit"
        consistency = f"{pref['consistency']:.1f}%"
        
        # Show recent pattern (last 5 epochs)
        recent_pattern = pref['all_precisions'][-5:] if len(pref['all_precisions']) >= 5 else pref['all_precisions']
        pattern_str = "→".join(map(str, recent_pattern))
        
        print(f"Layer {layer_idx:<3} {preferred:<12} {consistency:<12} {pattern_str}")
    
    return layer_preferences


def main():
    parser = argparse.ArgumentParser(description="Analyze router outputs from training")
    parser.add_argument("router_outputs_file", help="Path to validation_router_outputs.jsonl")
    parser.add_argument("--save-plots", help="Directory to save plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.router_outputs_file):
        print(f"❌ File not found: {args.router_outputs_file}")
        return
    
    print(f"🔍 Analyzing router outputs from: {args.router_outputs_file}")
    
    # Load data
    outputs = load_router_outputs(args.router_outputs_file)
    print(f"📊 Loaded data for {len(outputs)} epochs")
    
    # Analyze precision evolution
    epochs, layer_precisions = analyze_precision_evolution(outputs)
    
    # Create plots
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)
        plot_path = os.path.join(args.save_plots, 'precision_evolution.png')
        plot_precision_evolution(epochs, layer_precisions, plot_path)
    else:
        plot_precision_evolution(epochs, layer_precisions)
        plt.show()
    
    # Analyze distribution
    precision_dist = analyze_precision_distribution(outputs)
    
    # Analyze layer patterns
    layer_patterns = analyze_layer_patterns(outputs)
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
