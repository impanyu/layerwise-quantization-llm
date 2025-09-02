# Router Training with Validation

This document describes the new validation features added to the router training system.

## Overview

The router training system now includes comprehensive validation functionality to monitor training progress and prevent overfitting. The validation system splits your data into training and validation sets, evaluates the model on both sets during training, and saves detailed metrics for analysis.

## New Features

### 1. Automatic Data Splitting
- **Validation Split**: Automatically splits your data into training and validation sets
- **Flexible Configuration**: Choose percentage split or specify exact number of validation examples
- **Consistent Seeding**: Uses the same random seed for reproducible splits

### 2. Validation During Training
- **Epoch-by-Epoch Evaluation**: Validates the model after each training epoch
- **No Gradient Updates**: Validation runs in `torch.no_grad()` mode for efficiency
- **Same Metrics**: Tracks the same loss components on validation data

### 3. Enhanced History Tracking
- **Separate Metrics**: Tracks training and validation metrics separately
- **JSON Export**: Saves complete history to `training_history.json`
- **Human-Readable Summary**: Creates `training_summary.txt` with key information

### 4. Best Model Selection
- **Validation-Based**: Selects best model based on validation loss (not training loss)
- **Automatic Saving**: Saves best checkpoint as `best_router_checkpoint.pt`
- **Overfitting Prevention**: Helps identify when to stop training

## Usage

### Basic Usage with Default Split

```python
from layerwise.train_layerwise_router import RouterTrainer

trainer = RouterTrainer(
    model_path="path/to/quantized/model",
    num_examples=1000,
    validation_split=0.2,  # 20% for validation
    num_epochs=10
)

trainer.train()
```

### Advanced Configuration

```python
trainer = RouterTrainer(
    model_path="path/to/quantized/model",
    dataset='c4',
    seq_len=512,
    num_examples=1000,           # Training examples
    validation_examples=200,     # Explicit validation count (overrides split)
    batch_size=4,
    learning_rate=1e-4,
    num_epochs=10,
    weight_ce=0.7,
    weight_precision=0.3,
    validation_split=0.2,        # Used if validation_examples is None
    save_dir='my_checkpoints',
    random_state=42
)
```

### Command Line Usage

```bash
python layerwise/train_layerwise_router.py \
    path/to/model \
    --num_examples 1000 \
    --validation_split 0.2 \
    --num_epochs 10 \
    --batch_size 4 \
    --learning_rate 1e-4
```

## Output Files

After training, you'll find these files in your checkpoint directory:

### 1. `training_history.json`
Complete training and validation metrics for each epoch:
```json
{
  "epoch": [1, 2, 3, ...],
  "train_total_loss": [2.45, 2.31, 2.18, ...],
  "train_ce_loss": [2.20, 2.10, 2.05, ...],
  "train_precision_loss": [0.25, 0.21, 0.13, ...],
  "train_avg_precision": [6.2, 6.4, 6.6, ...],
  "val_total_loss": [2.52, 2.38, 2.25, ...],
  "val_ce_loss": [2.28, 2.15, 2.12, ...],
  "val_precision_loss": [0.24, 0.23, 0.13, ...],
  "val_avg_precision": [6.1, 6.3, 6.5, ...]
}
```

### 2. `training_summary.txt`
Human-readable summary with configuration and final results:
```
Router Training Summary
==================================================

Configuration:
  - Model: path/to/model
  - Dataset: c4
  - Training examples: 800
  - Validation examples: 200
  - Validation split: 0.2
  - Available precisions: [3, 4, 5, 6, 7, 8]

Final Results:
  - Final training loss: 2.18
  - Final validation loss: 2.25
  - Best validation loss: 2.23
  - Final training avg precision: 6.6
  - Final validation avg precision: 6.5
```

### 3. `validation_router_outputs.jsonl`
Detailed router outputs for each layer during validation (one entry per epoch):
```json
{
  "epoch": 1,
  "layers": [
    {
      "layer": 0,
      "selected_precision": 6,
      "precision_idx": 3,
      "router_output": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    },
    // ... more layers
  ],
  "summary": {
    "selected_precisions": [6, 4, 8, 6, ...],
    "avg_precision": 6.2,
    "precision_distribution": {"4": 5, "6": 10, "8": 3}
  }
}
```

### 4. Checkpoint Files
- `best_router_checkpoint.pt`: Best model based on validation loss
- `router_checkpoint_epoch_N.pt`: Checkpoint for each epoch

## Monitoring Training

### During Training
The training logs now show both training and validation metrics:

```
Epoch 1/10
[10:30:15 | INFO] Epoch 1 Results:
[10:30:15 | INFO]   Training:
[10:30:15 | INFO]     - Total Loss: 2.45
[10:30:15 | INFO]     - CE Loss: 2.20
[10:30:15 | INFO]     - Precision Loss: 0.25
[10:30:15 | INFO]     - Average Precision: 6.2
[10:30:15 | INFO]   Validation:
[10:30:15 | INFO]     - Total Loss: 2.52
[10:30:15 | INFO]     - CE Loss: 2.28
[10:30:15 | INFO]     - Precision Loss: 0.24
[10:30:15 | INFO]     - Average Precision: 6.1
```

### Plotting Training Curves
Use the provided example script to visualize training progress:

```python
# See example_router_training_with_validation.py
plot_training_history('router_checkpoints')
```

### Analyzing Router Outputs
Use the provided analysis script to understand router behavior:

```bash
python analyze_router_outputs.py router_checkpoints/validation_router_outputs.jsonl --save-plots ./plots
```

This will show:
- How precision selection evolves over training
- Which layers prefer which precisions
- Precision distribution across layers
- Layer-wise consistency patterns

## Best Practices

### 1. Validation Split Size
- **Small datasets**: Use 15-20% for validation
- **Large datasets**: 10-15% is usually sufficient
- **Very large datasets**: Even 5-10% can be effective

### 2. Monitoring for Overfitting
- **Watch the gap**: Large differences between training and validation loss indicate overfitting
- **Early stopping**: Stop training when validation loss stops improving
- **Learning rate**: Reduce if validation loss starts increasing

### 3. Data Quality
- **Consistent splits**: Use the same `random_state` for reproducible experiments
- **Representative data**: Ensure validation set represents your target distribution
- **Sufficient size**: Validation set should be large enough for reliable estimates

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_split` | float | 0.2 | Fraction of data for validation (0.0-1.0) |
| `validation_examples` | int | None | Explicit number of validation examples |
| `num_examples` | int | 1000 | Total examples for training (if validation_examples is None) |

If `validation_examples` is specified, it overrides `validation_split`.

## Migration from Old Code

If you have existing training scripts, the new validation features are **backward compatible**. Your existing code will work without changes, but you can opt-in to validation by setting `validation_split` > 0 or providing `validation_examples`.

```python
# Old code (still works)
trainer = RouterTrainer(model_path, num_examples=1000)

# New code with validation
trainer = RouterTrainer(model_path, num_examples=1000, validation_split=0.2)
```
