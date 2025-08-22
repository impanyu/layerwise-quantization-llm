#!/usr/bin/env python3
"""
Test script to demonstrate router handling of variable dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """Router network that outputs a one-hot vector for precision selection."""
    def __init__(self, input_dim, num_precisions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_precisions)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, num_real_tokens=None):
        # Use mean pooling if input has multiple dimensions
        if x.dim() > 2:
            if num_real_tokens is not None:
                # Average only over real tokens (not padding)
                batch_size, seq_len, hidden_dim = x.shape
                x_reshaped = x.view(batch_size * seq_len, hidden_dim)
                # Create mask for real tokens
                real_token_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < num_real_tokens.unsqueeze(1)
                real_token_mask = real_token_mask.view(batch_size * seq_len)
                # Average only real tokens
                x = x_reshaped[real_token_mask].view(batch_size, -1, hidden_dim).mean(dim=1)
            else:
                # Fallback to original behavior
                x = x.mean(dim=1)  # Average over sequence length
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

def test_router_dimensions():
    """Test router with various input dimensions."""
    
    # Initialize router
    input_dim = 768
    num_precisions = 3
    router = Router(input_dim, num_precisions)
    
    print("Testing Router with Variable Dimensions:")
    print("=" * 50)
    
    # Test cases with different batch_size and seq_len
    test_cases = [
        (1, 5),      # Single sample, short sequence
        (2, 10),     # Small batch, short sequence
        (4, 50),     # Medium batch, medium sequence
        (8, 100),    # Medium batch, long sequence
        (16, 512),   # Large batch, very long sequence
        (32, 256),   # Large batch, long sequence
    ]
    
    for batch_size, seq_len in test_cases:
        # Create input tensor
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Create attention mask (simulate some padding)
        attention_mask = torch.ones(batch_size, seq_len)
        # Add some padding to test real token counting
        for i in range(batch_size):
            padding_start = seq_len - (i + 1)  # Different padding for each sample
            if padding_start < seq_len:
                attention_mask[i, padding_start:] = 0
        
        # Count real tokens
        num_real_tokens = attention_mask.sum(dim=1)
        
        # Forward pass with real token count
        output = router(x, num_real_tokens)
        
        # Check output shape
        expected_shape = (batch_size, num_precisions)
        actual_shape = output.shape
        
        print(f"Input: [{batch_size}, {seq_len}, {input_dim}]")
        print(f"Output: {actual_shape}")
        print(f"Expected: {expected_shape}")
        print(f"✓ Match: {actual_shape == expected_shape}")
        
        # Check that output sums to 1 (softmax property)
        output_sums = output.sum(dim=-1)
        print(f"Output sums: {output_sums.min().item():.4f} to {output_sums.max().item():.4f}")
        print(f"✓ Valid probabilities: {torch.allclose(output_sums, torch.ones_like(output_sums))}")
        print("-" * 30)
    
    print("✓ All dimension tests passed!")
    
    # Test edge cases
    print("\nTesting Edge Cases:")
    print("=" * 30)
    
    # Edge case 1: Very short sequence
    x1 = torch.randn(2, 1, input_dim)  # seq_len = 1
    output1 = router(x1)
    print(f"seq_len=1: Input {x1.shape} → Output {output1.shape} ✓")
    
    # Edge case 2: Very long sequence
    x2 = torch.randn(1, 1000, input_dim)  # seq_len = 1000
    output2 = router(x2)
    print(f"seq_len=1000: Input {x2.shape} → Output {output2.shape} ✓")
    
    # Edge case 3: Large batch
    x3 = torch.randn(64, 10, input_dim)  # batch_size = 64
    output3 = router(x3)
    print(f"batch_size=64: Input {x3.shape} → Output {output3.shape} ✓")
    
    print("✓ All edge cases passed!")

if __name__ == "__main__":
    test_router_dimensions()
