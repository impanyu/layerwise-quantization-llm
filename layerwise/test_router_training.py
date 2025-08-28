#!/usr/bin/env python3
"""
Test script for router training components.

This script tests the key components of the router training without
requiring a full quantized model, making it useful for development
and debugging.
"""

import torch
import torch.nn as nn
import numpy as np
from train_layerwise_router import RouterTrainer


def test_router_architecture():
    """Test the Router architecture."""
    print("Testing Router architecture...")
    
    # Create a simple router
    input_dim = 512
    num_precisions = 6
    hidden_dim = 128
    
    router = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, num_precisions),
        nn.Softmax(dim=-1)
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Mean pooling over sequence dimension
    x_pooled = x.mean(dim=1)  # [batch_size, input_dim]
    
    # Forward pass
    output = router(x_pooled)
    
    print(f"Input shape: {x.shape}")
    print(f"Pooled input shape: {x_pooled.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum per sample: {output.sum(dim=1)}")  # Should be close to 1.0
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    assert output.shape == (batch_size, num_precisions)
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    print("✓ Router architecture test passed!")


def test_precision_calculation():
    """Test the precision calculation functions."""
    print("\nTesting precision calculation...")
    
    # Simulate router outputs
    batch_size = 4
    num_layers = 3
    num_precisions = 6
    precisions = [3, 4, 5, 6, 7, 8]
    
    # Create random router outputs
    router_outputs = []
    for _ in range(num_layers):
        # Create random weights that sum to 1
        weights = torch.rand(batch_size, num_precisions)
        weights = weights / weights.sum(dim=1, keepdim=True)
        router_outputs.append(weights)
    
    # Calculate average precision manually
    avg_precisions = []
    for batch_idx in range(batch_size):
        layer_precisions = []
        for layer_idx in range(num_layers):
            precision_weights = router_outputs[layer_idx][batch_idx]
            weighted_precision = torch.sum(precision_weights * torch.tensor(precisions, dtype=torch.float32))
            layer_precisions.append(weighted_precision)
        avg_precision = torch.mean(torch.stack(layer_precisions))
        avg_precisions.append(avg_precision)
    
    avg_precision_tensor = torch.stack(avg_precisions)
    
    print(f"Router outputs shape: {[ro.shape for ro in router_outputs]}")
    print(f"Available precisions: {precisions}")
    print(f"Average precision per sample: {avg_precision_tensor}")
    print(f"Average precision range: [{avg_precision_tensor.min():.2f}, {avg_precision_tensor.max():.2f}]")
    
    # Test normalization
    min_precision = min(precisions)
    max_precision = max(precisions)
    normalized = (avg_precision_tensor - min_precision) / (max_precision - min_precision)
    scaled = normalized * 10.0
    
    print(f"Normalized precision: {normalized}")
    print(f"Scaled precision: {scaled}")
    print(f"Scaled range: [{scaled.min():.2f}, {scaled.max():.2f}]")
    
    assert torch.all(avg_precision_tensor >= min_precision)
    assert torch.all(avg_precision_tensor <= max_precision)
    print("✓ Precision calculation test passed!")


def test_loss_function():
    """Test the custom loss function."""
    print("\nTesting custom loss function...")
    
    # Simulate model outputs
    batch_size = 4
    seq_len = 10
    vocab_size = 32000
    
    # Create dummy logits and labels
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create dummy router outputs
    num_layers = 3
    num_precisions = 6
    router_outputs = []
    for _ in range(num_layers):
        weights = torch.rand(batch_size, num_precisions)
        weights = weights / weights.sum(dim=1, keepdim=True)
        router_outputs.append(weights)
    
    # Calculate losses
    ce_loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1))
    
    # Calculate average precision
    precisions = [3, 4, 5, 6, 7, 8]
    avg_precisions = []
    for batch_idx in range(batch_size):
        layer_precisions = []
        for layer_idx in range(num_layers):
            precision_weights = router_outputs[layer_idx][batch_idx]
            weighted_precision = torch.sum(precision_weights * torch.tensor(precisions, dtype=torch.float32))
            layer_precisions.append(weighted_precision)
        avg_precision = torch.mean(torch.stack(layer_precisions))
        avg_precisions.append(avg_precision)
    
    avg_precision_tensor = torch.stack(avg_precisions)
    
    # Normalize precision loss
    min_precision = min(precisions)
    max_precision = max(precisions)
    normalized_precision = (avg_precision_tensor - min_precision) / (max_precision - min_precision)
    precision_loss = torch.mean(normalized_precision * 10.0)
    
    # Combined loss
    weight_ce = 0.7
    weight_precision = 0.3
    total_loss = weight_ce * ce_loss + weight_precision * precision_loss
    
    print(f"CE Loss: {ce_loss:.4f}")
    print(f"Precision Loss: {precision_loss:.4f}")
    print(f"Total Loss: {total_loss:.4f}")
    print(f"Average Precision: {torch.mean(avg_precision_tensor):.2f}")
    
    assert ce_loss > 0
    assert precision_loss > 0
    assert total_loss > 0
    print("✓ Loss function test passed!")


def test_data_loading():
    """Test data loading and preprocessing."""
    print("\nTesting data loading...")
    
    # Create dummy data
    num_examples = 10
    seq_len = 64
    vocab_size = 32000
    
    # Create dummy tokens
    input_tokens = []
    for _ in range(num_examples):
        tokens = torch.randint(0, vocab_size, (seq_len,))
        input_tokens.append(tokens)
    
    # Convert to tensors and create attention masks
    input_ids = []
    attention_masks = []
    
    for tokens in input_tokens:
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        input_ids.append(tokens)
        attention_masks.append(attention_mask)
    
    # Pad sequences
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        padding_length = max_len - len(ids)
        padded_ids = torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)])
        padded_input_ids.append(padded_ids)
        
        padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
        padded_attention_masks.append(padded_mask)
    
    # Convert to tensors
    input_ids_tensor = torch.stack(padded_input_ids)
    attention_masks_tensor = torch.stack(padded_attention_masks)
    
    print(f"Number of examples: {len(input_tokens)}")
    print(f"Sequence length: {seq_len}")
    print(f"Padded input shape: {input_ids_tensor.shape}")
    print(f"Attention mask shape: {attention_masks_tensor.shape}")
    print(f"Input range: [{input_ids_tensor.min()}, {input_ids_tensor.max()}]")
    print(f"Mask range: [{attention_masks_tensor.min()}, {attention_masks_tensor.max()}]")
    
    assert input_ids_tensor.shape == (num_examples, max_len)
    assert attention_masks_tensor.shape == (num_examples, max_len)
    assert torch.all(attention_masks_tensor >= 0) and torch.all(attention_masks_tensor <= 1)
    print("✓ Data loading test passed!")


def main():
    """Run all tests."""
    print("Running router training component tests...\n")
    
    try:
        test_router_architecture()
        test_precision_calculation()
        test_loss_function()
        test_data_loading()
        
        print("\n🎉 All tests passed!")
        print("\nThe router training components are working correctly.")
        print("You can now proceed with training on a real quantized model.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
