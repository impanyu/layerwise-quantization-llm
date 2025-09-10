#!/usr/bin/env python3
"""
Example script showing how to evaluate and compare router vs quantized models.

This script demonstrates the usage of evaluate_router_vs_quantized.py with
different evaluation scenarios.
"""

import subprocess
import os
import sys

def run_evaluation(checkpoint_dir=None, **kwargs):
    """Run the evaluation script with given parameters."""
    cmd = [sys.executable, "evaluate_router_vs_quantized.py"]
    
    if checkpoint_dir:
        cmd.append(checkpoint_dir)
    
    # Add optional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Evaluation completed successfully!")
        print(result.stdout)
    else:
        print("❌ Evaluation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    print("Router vs Quantized Models Evaluation Examples")
    print("=" * 60)
    
    # Default checkpoint directory (same as training default)
    default_checkpoint_dir = "router_checkpoints"
    
    # Example 1: Basic evaluation with default checkpoint directory
    print("\n📊 Example 1: Basic evaluation with default checkpoint directory")
    print("-" * 50)
    if not os.path.exists(default_checkpoint_dir):
        print(f"⚠️  Default checkpoint directory not found: {default_checkpoint_dir}")
        print("Make sure you have trained a router model first using train_layerwise_router.py")
        print("Or specify a custom checkpoint directory in the examples below.")
    
    success = run_evaluation(
        # checkpoint_dir=None,  # Uses default "router_checkpoints"
        num_examples=100,  # Small number for quick testing
        batch_size=2
    )
    
    if not success:
        print("Skipping other examples due to failure.")
        return
    
    # Example 2: Comprehensive evaluation with default directory
    print("\n📊 Example 2: Comprehensive evaluation")
    print("-" * 50)
    run_evaluation(
        num_examples=500,  # More examples for better statistics
        batch_size=4,
        seq_len=256,  # Shorter sequences for faster evaluation
        dataset="c4"
    )
    
    # Example 3: CPU evaluation (useful for debugging or when GPU unavailable)
    print("\n📊 Example 3: CPU evaluation (no plots)")
    print("-" * 50)
    run_evaluation(
        num_examples=50,  # Very small for CPU
        batch_size=1,
        device="cpu",
        no_plots=True
    )
    
    # Example 4: Custom checkpoint and output directories
    print("\n📊 Example 4: Custom checkpoint and output directories")
    print("-" * 50)
    custom_checkpoint_dir = "my_router_checkpoints"
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    run_evaluation(
        checkpoint_dir=custom_checkpoint_dir,  # Custom checkpoint directory
        num_examples=200,
        output_dir=output_dir
    )
    
    print("\n🎉 All evaluation examples completed!")
    print("\nGenerated files:")
    print("- router_vs_quantized_results.json    # Detailed numerical results")
    print("- router_vs_quantized_comparison.png  # Main comparison plot")
    print("- router_vs_quantized_comparison.pdf  # Publication-ready plot")
    print("- loss_focused_comparison.png         # Focused loss comparison")

if __name__ == "__main__":
    main()
