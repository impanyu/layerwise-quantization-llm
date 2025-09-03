#!/usr/bin/env python3
"""
Example demonstrating how to switch between softmax and sparsemax activation functions
"""

from layerwise.LayerwiseQuantizeForCausalLM import LayerwiseQuantizeForCausalLM

def main():
    """Example of loading model with different activation functions."""
    
    model_path = "path/to/your/quantized/model"  # Replace with actual path
    
    print("🔄 Router Activation Function Examples")
    print("=" * 50)
    
    # Example 1: Default behavior (softmax)
    print("\n1. Loading model with Softmax (default):")
    try:
        model_softmax = LayerwiseQuantizeForCausalLM.from_quantized(
            quant_model_path=model_path,
            precisions=[4, 6, 8],
            use_sparsemax=False  # Default: False
        )
        print(f"   ✅ Model loaded with softmax activation")
        print(f"   📊 Router activation: {'Sparsemax' if model_softmax.use_sparsemax else 'Softmax'}")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
    
    # Example 2: Using sparsemax for sparse precision selection
    print("\n2. Loading model with Sparsemax:")
    try:
        model_sparsemax = LayerwiseQuantizeForCausalLM.from_quantized(
            quant_model_path=model_path,
            precisions=[4, 6, 8],
            use_sparsemax=True  # Enable sparsemax
        )
        print(f"   ✅ Model loaded with sparsemax activation")
        print(f"   📊 Router activation: {'Sparsemax' if model_sparsemax.use_sparsemax else 'Softmax'}")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
    
    print("\n🔍 Key Differences:")
    print("   • Softmax: Produces smooth probability distributions over precisions")
    print("   • Sparsemax: Produces sparse distributions (some precisions get exactly 0 probability)")
    print("   • Sparsemax can lead to more decisive precision selection")
    print("   • Softmax provides smoother gradients during training")
    
    print("\n💡 Usage Guidelines:")
    print("   • Start with softmax (default) for stable training")
    print("   • Try sparsemax if you want more decisive precision selection")
    print("   • Sparsemax may help with interpretability and efficiency")
    print("   • Both implementations are available for easy switching")


def compare_router_outputs():
    """Example comparing router outputs between softmax and sparsemax."""
    import torch
    from layerwise.LayerwiseQuantizeForCausalLM import Router, sparsemax
    import torch.nn.functional as F
    
    print("\n🧪 Comparing Router Outputs")
    print("=" * 30)
    
    # Create example router logits
    logits = torch.tensor([[2.0, 1.0, 0.5, 3.0, 0.1, 1.5]], dtype=torch.float32)
    precisions = [3, 4, 5, 6, 7, 8]
    
    print(f"Input logits: {logits[0].tolist()}")
    print(f"Precisions:   {precisions}")
    
    # Apply softmax
    softmax_output = F.softmax(logits, dim=-1)
    print(f"\nSoftmax output:   {[f'{x:.3f}' for x in softmax_output[0]]}")
    print(f"Selected precision (argmax): {precisions[torch.argmax(softmax_output).item()]}-bit")
    
    # Apply sparsemax
    sparsemax_output = sparsemax(logits, dim=-1)
    print(f"Sparsemax output: {[f'{x:.3f}' for x in sparsemax_output[0]]}")
    print(f"Selected precision (argmax): {precisions[torch.argmax(sparsemax_output).item()]}-bit")
    
    # Count non-zero elements
    softmax_nonzero = torch.sum(softmax_output > 1e-6).item()
    sparsemax_nonzero = torch.sum(sparsemax_output > 1e-6).item()
    
    print(f"\nSparsity comparison:")
    print(f"  Softmax non-zero elements:   {softmax_nonzero}/{len(precisions)}")
    print(f"  Sparsemax non-zero elements: {sparsemax_nonzero}/{len(precisions)}")


if __name__ == "__main__":
    main()
    compare_router_outputs()
