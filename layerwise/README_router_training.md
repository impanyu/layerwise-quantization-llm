# Router Training for LayerwiseQuantizeForCausalLM

This directory contains the implementation for training the router network in `LayerwiseQuantizeForCausalLM`. The router learns to adaptively select precision levels for each layer based on the input characteristics.

## Overview

The router training program implements:

1. **Custom Loss Function**: Combines cross-entropy loss with a precision penalty
2. **Dataset Handling**: Uses the same dataset loading mechanism as `quantize.py`
3. **Router Training**: Only trains the router parameters while keeping the original model frozen
4. **Precision Normalization**: Normalizes precision loss to be comparable to cross-entropy

## Files

- `train_layerwise_router.py`: Main training program
- `example_train_router.py`: Example script for quick testing
- `README_router_training.md`: This documentation file

## Requirements

Make sure you have:
1. A quantized model created using the main quantization pipeline
2. The required dependencies installed (torch, transformers, datasets, etc.)
3. Sufficient GPU memory for training

## Usage

### Basic Usage

```bash
python train_layerwise_router.py /path/to/quantized/model \
    --dataset c4 \
    --seq_len 512 \
    --num_examples 1000 \
    --batch_size 4 \
    --num_epochs 10 \
    --weight_ce 0.7 \
    --weight_precision 0.3
```

### Example Usage (Faster Training)

```bash
python example_train_router.py \
    --model_path /path/to/quantized/model \
    --seq_len 256 \
    --num_examples 500 \
    --batch_size 2 \
    --num_epochs 5
```

## Parameters

### Required Parameters
- `model_path`: Path to the quantized model directory

### Optional Parameters
- `--dataset`: Dataset to use (default: "c4")
  - Options: "c4", "wikitext2", "ptb", "pileval"
- `--seq_len`: Sequence length (default: 512)
- `--num_examples`: Number of examples to use (default: 1000)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Number of training epochs (default: 10)
- `--weight_ce`: Weight for cross-entropy loss (default: 0.7)
- `--weight_precision`: Weight for precision loss (default: 0.3)
- `--save_dir`: Directory to save checkpoints (default: "router_checkpoints")
- `--device`: Device to use (default: auto-detect)
- `--random_state`: Random seed (default: 42)
- `--trust_remote_code`: Trust remote code flag
- `--precisions`: Specific precisions to use (default: all available)

## Loss Function

The training uses a custom loss function that combines:

1. **Cross-Entropy Loss**: Standard language modeling loss
2. **Precision Loss**: Penalty for using higher precision (to encourage efficiency)

```
Total Loss = weight_ce * CE_Loss + weight_precision * Precision_Loss
```

The precision loss is normalized to be comparable to cross-entropy (typically 0-10 range).

## Training Process

1. **Model Loading**: Loads the quantized model and freezes all parameters except routers
2. **Data Preparation**: Uses the same dataset loading as `quantize.py`
3. **Forward Pass**: Uses `train_forward` method with mixed precision
4. **Loss Calculation**: Computes combined loss with precision penalty
5. **Backward Pass**: Updates only router parameters
6. **Checkpointing**: Saves model checkpoints and training history

## Output Files

The training creates several output files in the save directory:

- `router_checkpoint_epoch_N.pt`: Checkpoint for each epoch
- `best_router_checkpoint.pt`: Best model based on loss
- `training_history.json`: Training metrics over time

## Loading Trained Router

After training, you can load the trained router:

```python
from LayerwiseQuantizeForCausalLM import LayerwiseQuantizeForCausalLM

# Load the model
model = LayerwiseQuantizeForCausalLM.from_quantized(
    quant_model_path="/path/to/quantized/model",
    trust_remote_code=True
)

# Load the trained router
checkpoint = torch.load("router_checkpoints/best_router_checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
model.eval()
outputs = model.infer_forward(input_ids, attention_mask)
```

## Training Tips

1. **Start Small**: Use smaller datasets and shorter sequences for initial testing
2. **Memory Management**: Reduce batch size if you encounter memory issues
3. **Loss Weights**: Adjust `weight_ce` and `weight_precision` to balance accuracy vs efficiency
4. **Learning Rate**: Start with 1e-4 and adjust based on training stability
5. **Monitoring**: Check the training history JSON file for detailed metrics

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Model Not Found**: Ensure the quantized model path is correct
3. **Dataset Issues**: Check if the dataset is available and accessible
4. **Precision Range**: Ensure the specified precisions are supported by the model

### Debug Mode

For debugging, you can add more verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example Training Session

Here's what a typical training session looks like:

```
[12:34:56 | INFO] Starting router training...
[12:34:56 | INFO] Training configuration:
[12:34:56 | INFO]   - Dataset: c4
[12:34:56 | INFO]   - Sequence length: 512
[12:34:56 | INFO]   - Number of examples: 1000
[12:34:56 | INFO]   - Batch size: 4
[12:34:56 | INFO]   - Learning rate: 0.0001
[12:34:56 | INFO]   - Number of epochs: 10
[12:34:56 | INFO]   - CE weight: 0.7
[12:34:56 | INFO]   - Precision weight: 0.3
[12:34:56 | INFO]   - Available precisions: [3, 4, 5, 6, 7, 8]

[12:34:57 | INFO] Loading model from /path/to/quantized/model
[12:34:58 | INFO] Model loaded successfully. Device: cuda
[12:34:58 | INFO] Available precisions: [3, 4, 5, 6, 7, 8]
[12:34:58 | INFO] Number of trainable parameters: 123456

[12:34:59 | INFO] Loading dataset: c4
[12:34:59 | INFO] Fetching dataset: c4
[12:35:00 | INFO] Sampling 1000 samples of length 512 from c4...
[12:35:01 | INFO] Dataset loaded: 1000 samples, batch size: 4

[12:35:01 | INFO] Optimizer setup: AdamW with lr=0.0001

[12:35:01 | INFO] Epoch 1/10
Epoch 1/10: 100%|██████████| 250/250 [02:30<00:00, Loss: 3.2456, CE: 2.1234, Prec: 1.1222, AvgPrec: 5.67]
[12:37:31 | INFO] Epoch 1 Results:
[12:37:31 | INFO]   - Total Loss: 3.2456
[12:37:31 | INFO]   - CE Loss: 2.1234
[12:37:31 | INFO]   - Precision Loss: 1.1222
[12:37:31 | INFO]   - Average Precision: 5.67
[12:37:32 | INFO] Checkpoint saved: router_checkpoints/router_checkpoint_epoch_1.pt
[12:37:32 | INFO] New best model saved: router_checkpoints/best_router_checkpoint.pt

...

[12:45:00 | INFO] Training completed!
[12:45:00 | INFO] Best loss: 2.9876
[12:45:00 | INFO] Checkpoints saved in: router_checkpoints
```

## Performance Considerations

- **Training Time**: Depends on model size, dataset size, and hardware
- **Memory Usage**: Scales with batch size and sequence length
- **GPU Requirements**: Recommended to use GPU for faster training
- **Storage**: Checkpoints can be large, ensure sufficient disk space

## Future Improvements

Potential enhancements for the router training:

1. **Curriculum Learning**: Start with simpler examples and gradually increase difficulty
2. **Multi-Objective Optimization**: Use Pareto optimization for accuracy-efficiency trade-off
3. **Adaptive Learning Rate**: Implement learning rate scheduling
4. **Regularization**: Add dropout or other regularization techniques
5. **Validation**: Add validation set for better model selection
