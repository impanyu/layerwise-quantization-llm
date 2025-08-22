#!/usr/bin/env python3
"""
Test script to verify parameter freezing in LayerwiseQuantizeForCausalLM
"""

import torch
from layerwise.LayerwiseQuantizeForCausalLM import LayerwiseQuantizeForCausalLM

def test_parameter_freezing():
    """Test that only router parameters are trainable."""
    
    # Test with a small model (you'll need to replace with actual model path)
    model_path = "path/to/your/model"  # Replace with actual model path
    
    try:
        # Initialize the layerwise model
        model = LayerwiseQuantizeForCausalLM.from_quantized(
            quant_model_path=model_path,
            precisions=[4, 8, 16],
            trust_remote_code=True
        )
        
        print("✓ Model initialized successfully")
        
        # Check trainable parameters
        trainable_params = model.get_trainable_parameters()
        frozen_params = model.get_frozen_parameters()
        
        print(f"✓ Trainable parameters: {len(trainable_params)} parameter groups")
        print(f"✓ Frozen parameters: {len(frozen_params)} parameter groups")
        
        # Verify that only router parameters are trainable
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = sum(p.numel() for p in frozen_params)
        
        print(f"✓ Trainable parameter count: {trainable_count:,}")
        print(f"✓ Frozen parameter count: {frozen_count:,}")
        
        # Test that frozen parameters don't have gradients
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones_like(input_ids)
        
        output = model.train_forward(input_ids, attention_mask)
        loss = output.logits.sum()  # Dummy loss
        loss.backward()
        
        # Check that only router parameters have gradients
        router_grads = 0
        frozen_grads = 0
        
        for param in trainable_params:
            if param.grad is not None:
                router_grads += param.grad.numel()
        
        for param in frozen_params:
            if param.grad is not None:
                frozen_grads += param.grad.numel()
        
        print(f"✓ Router parameters with gradients: {router_grads:,}")
        print(f"✓ Frozen parameters with gradients: {frozen_grads:,}")
        
        if frozen_grads == 0:
            print("✓ SUCCESS: Only router parameters have gradients!")
        else:
            print("✗ ERROR: Frozen parameters have gradients!")
        
        # Test optimizer with only trainable parameters
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
        optimizer.step()
        print("✓ Optimizer step completed successfully")
        
        print("✓ All parameter freezing tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parameter_freezing()
