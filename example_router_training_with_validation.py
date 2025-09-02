#!/usr/bin/env python3
"""
Example script demonstrating router training with validation
"""

import os
import sys
from layerwise.train_layerwise_router import RouterTrainer

def main():
    """Example of router training with validation."""
    
    # Example configuration - replace with your actual model path
    model_path = "path/to/your/quantized/model"  # Replace with actual path
    
    # Check if model path exists (for demo purposes)
    if not os.path.exists(model_path):
        print(f"⚠️  Model path '{model_path}' does not exist.")
        print("Please update the model_path variable with your actual quantized model path.")
        return
    
    # Create trainer with validation settings
    trainer = RouterTrainer(
        model_path=model_path,
        dataset='c4',
        seq_len=512,
        num_examples=1000,          # Total examples (training + validation)
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=5,
        weight_ce=0.7,
        weight_precision=0.3,
        save_dir='router_checkpoints_with_validation',
        validation_split=0.2,       # 20% for validation, 80% for training
        # validation_examples=200,  # Alternatively, specify exact number
        precisions=[4, 6, 8],       # Custom precision subset
        random_state=42
    )
    
    print("🚀 Starting router training with validation...")
    print(f"📊 Training examples: {len(trainer.train_input_ids)}")
    print(f"📊 Validation examples: {len(trainer.val_input_ids)}")
    print(f"📊 Validation split: {trainer.validation_split}")
    print(f"📊 Available precisions: {trainer.model.precisions}")
    
    # Start training
    trainer.train()
    
    print("\n✅ Training completed!")
    print(f"📁 Results saved in: {trainer.save_dir}")
    print("📋 Files generated:")
    print(f"   - training_history.json: Complete training and validation metrics")
    print(f"   - training_summary.txt: Human-readable summary")
    print(f"   - best_router_checkpoint.pt: Best model based on validation loss")
    print(f"   - router_checkpoint_epoch_*.pt: Checkpoints for each epoch")

def plot_training_history(checkpoint_dir):
    """
    Example function to plot training history.
    Requires matplotlib: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import json
        
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        if not os.path.exists(history_path):
            print(f"History file not found: {history_path}")
            return
            
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = history['epoch']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total Loss
        ax1.plot(epochs, history['train_total_loss'], label='Training', marker='o')
        ax1.plot(epochs, history['val_total_loss'], label='Validation', marker='s')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # CE Loss
        ax2.plot(epochs, history['train_ce_loss'], label='Training', marker='o')
        ax2.plot(epochs, history['val_ce_loss'], label='Validation', marker='s')
        ax2.set_title('Cross Entropy Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('CE Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Precision Loss
        ax3.plot(epochs, history['train_precision_loss'], label='Training', marker='o')
        ax3.plot(epochs, history['val_precision_loss'], label='Validation', marker='s')
        ax3.set_title('Precision Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision Loss')
        ax3.legend()
        ax3.grid(True)
        
        # Average Precision
        ax4.plot(epochs, history['train_avg_precision'], label='Training', marker='o')
        ax4.plot(epochs, history['val_avg_precision'], label='Validation', marker='s')
        ax4.set_title('Average Precision')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Avg Precision (bits)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(checkpoint_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved: {plot_path}")
        
        # Optionally show plot
        # plt.show()
        
    except ImportError:
        print("📈 To plot training curves, install matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error plotting training curves: {e}")

if __name__ == "__main__":
    main()
    
    # Uncomment to plot training curves after training
    # plot_training_history('router_checkpoints_with_validation')
