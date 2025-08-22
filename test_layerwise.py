#!/usr/bin/env python3
"""
Test script for LayerwiseQuantizeForCausalLM
"""

import torch
from layerwise.LayerwiseQuantizeForCausalLM import LayerwiseQuantizeForCausalLM

def test_layerwise_model():
    """Test the layerwise quantization model."""
    
    # Test with a small model (you'll need to replace with actual model path)
    model_path = "path/to/your/model"  # Replace with actual model path
    
    try:
        # Initialize the layerwise model
        model = LayerwiseQuantizeForCausalLM.from_quantized(
            quant_model_path=model_path,
            precisions=[4, 8, 16],  # Test with 4, 8, and 16 bit precisions
            trust_remote_code=True
        )
        
        print("✓ Model initialized successfully")
        
        # Test input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        # Test train_forward
        print("Testing train_forward...")
        train_output = model.train_forward(input_ids, attention_mask)
        print(f"✓ Train forward output shape: {train_output.logits.shape}")
        
        # Test infer_forward
        print("Testing infer_forward...")
        infer_output = model.infer_forward(input_ids, attention_mask)
        print(f"✓ Infer forward output shape: {infer_output.logits.shape}")
        
        # Test regular forward (should use infer_forward)
        print("Testing regular forward...")
        regular_output = model(input_ids, attention_mask)
        print(f"✓ Regular forward output shape: {regular_output.logits.shape}")
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_layerwise_model()
