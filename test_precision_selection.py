#!/usr/bin/env python3
"""
Test script to demonstrate precision selection strategies
"""

import torch

def test_precision_selection():
    """Test different precision selection strategies."""
    
    # Example: 3 precisions [4-bit, 8-bit, 16-bit] with indices [0, 1, 2]
    precisions = [4, 8, 16]
    
    # Simulate router outputs for a batch of 4 samples
    # Each sample selects a different precision
    router_outputs = torch.tensor([
        [0.8, 0.1, 0.1],  # Sample 0: prefers 4-bit (index 0)
        [0.1, 0.8, 0.1],  # Sample 1: prefers 8-bit (index 1)
        [0.1, 0.1, 0.8],  # Sample 2: prefers 16-bit (index 2)
        [0.7, 0.2, 0.1],  # Sample 3: prefers 4-bit (index 0)
    ])  # [4, 3]
    
    # Get precision indices for each sample
    precision_indices = torch.argmax(router_outputs, dim=-1)  # [0, 1, 2, 0]
    
    print("Router outputs:")
    print(router_outputs)
    print(f"Precision indices: {precision_indices}")
    print()
    
    # Strategy 1: Mode (most frequent)
    mode_idx = torch.mode(precision_indices)[0].item()
    mode_precision = precisions[mode_idx]
    print(f"Mode strategy:")
    print(f"  Most frequent precision index: {mode_idx}")
    print(f"  Selected precision: {mode_precision}-bit")
    print(f"  Quality: Lower (uses most common precision)")
    print()
    
    # Strategy 2: Max (highest precision)
    max_idx = torch.max(precision_indices).item()
    max_precision = precisions[max_idx]
    print(f"Max strategy:")
    print(f"  Highest precision index: {max_idx}")
    print(f"  Selected precision: {max_precision}-bit")
    print(f"  Quality: Higher (uses highest precision)")
    print()
    
    # Strategy 3: Min (lowest precision)
    min_idx = torch.min(precision_indices).item()
    min_precision = precisions[min_idx]
    print(f"Min strategy:")
    print(f"  Lowest precision index: {min_idx}")
    print(f"  Selected precision: {min_precision}-bit")
    print(f"  Quality: Lowest (uses lowest precision)")
    print()
    
    # Strategy 4: Mean (average precision)
    mean_idx = torch.round(precision_indices.float().mean()).long().item()
    mean_precision = precisions[mean_idx]
    print(f"Mean strategy:")
    print(f"  Average precision index: {mean_idx}")
    print(f"  Selected precision: {mean_precision}-bit")
    print(f"  Quality: Medium (uses average precision)")
    print()
    
    # Compare quality vs efficiency
    print("Quality vs Efficiency Trade-off:")
    print(f"  Mode ({mode_precision}-bit):   Efficiency: High, Quality: Low")
    print(f"  Mean ({mean_precision}-bit):   Efficiency: Medium, Quality: Medium")
    print(f"  Max ({max_precision}-bit):     Efficiency: Low, Quality: High")
    print(f"  Min ({min_precision}-bit):     Efficiency: Highest, Quality: Lowest")
    
    print("\n✓ For inference, Max strategy is best for quality!")
    print("✓ For training, mixed precision is used anyway.")

if __name__ == "__main__":
    test_precision_selection()
