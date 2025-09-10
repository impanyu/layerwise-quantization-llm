#!/usr/bin/env python3
"""
Standalone script to plot training history from saved training_history.json files.
Useful for analyzing training results after the fact.
"""

import argparse
import json
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Error: matplotlib and seaborn are required for plotting.")
    print("Install them with: pip install matplotlib seaborn")
    exit(1)


def load_training_history(history_path):
    """Load training history from JSON file."""
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_training_history(history, save_dir=None):
    """Plot comprehensive training and validation history."""
    if len(history['epoch']) == 0:
        print("No training history to plot")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Router Training History', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_total_loss'], 'o-', label='Training', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_total_loss'], 's-', label='Validation', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cross Entropy Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_ce_loss'], 'o-', label='Training', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_ce_loss'], 's-', label='Validation', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.set_title('Cross Entropy Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision Loss
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['train_precision_loss'], 'o-', label='Training', linewidth=2, markersize=4)
    ax3.plot(epochs, history['val_precision_loss'], 's-', label='Validation', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision Loss')
    ax3.set_title('Precision Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average Precision
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['train_avg_precision'], 'o-', label='Training', linewidth=2, markersize=4)
    ax4.plot(epochs, history['val_avg_precision'], 's-', label='Validation', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Average Precision (bits)')
    ax4.set_title('Average Precision')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    if save_dir:
        plot_path = os.path.join(save_dir, 'training_history_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {plot_path}")
        
        # Also save as PDF
        plot_pdf_path = os.path.join(save_dir, 'training_history_plot.pdf')
        plt.savefig(plot_pdf_path, bbox_inches='tight')
        print(f"Training history plot (PDF) saved: {plot_pdf_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Create a focused loss comparison plot
    plot_loss_comparison(history, save_dir)


def plot_loss_comparison(history, save_dir=None):
    """Create a focused plot comparing training vs validation loss."""
    if len(history['epoch']) == 0:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = history['epoch']
    
    # Plot total loss
    ax.plot(epochs, history['train_total_loss'], 'o-', 
            label='Training Total Loss', linewidth=2.5, markersize=5, color='#2E86AB')
    ax.plot(epochs, history['val_total_loss'], 's-', 
            label='Validation Total Loss', linewidth=2.5, markersize=5, color='#A23B72')
    
    # Add markers for best validation loss
    best_val_idx = np.argmin(history['val_total_loss'])
    best_val_loss = history['val_total_loss'][best_val_idx]
    best_epoch = epochs[best_val_idx]
    
    ax.scatter([best_epoch], [best_val_loss], color='red', s=100, 
              marker='*', zorder=5, label=f'Best Val Loss (Epoch {best_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for best validation loss
    ax.annotate(f'Best: {best_val_loss:.4f}', 
               xy=(best_epoch, best_val_loss), 
               xytext=(best_epoch + 0.5, best_val_loss + 0.1),
               arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
               fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Save the focused loss plot
    if save_dir:
        loss_plot_path = os.path.join(save_dir, 'loss_comparison_plot.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"Loss comparison plot saved: {loss_plot_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training history from saved JSON files")
    parser.add_argument("history_path", type=str, help="Path to training_history.json file")
    parser.add_argument("--save_dir", type=str, default=None, 
                       help="Directory to save plots (if not provided, plots will be displayed)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history_path):
        print(f"Error: History file not found: {args.history_path}")
        return
    
    # Load training history
    print(f"Loading training history from: {args.history_path}")
    history = load_training_history(args.history_path)
    
    # Create plots
    plot_training_history(history, args.save_dir)
    
    print("Plotting completed!")


if __name__ == "__main__":
    main()
