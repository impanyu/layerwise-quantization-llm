# Router Model Evaluation

This directory contains tools for evaluating and comparing router model performance against fixed precision quantized models.

## Overview

The evaluation system allows you to:
1. Load a trained router model from checkpoints
2. Evaluate the router in inference mode (dynamic precision selection)
3. Compare against fixed precision models (3-8 bits)
4. Generate comprehensive comparison plots and reports

## Files

### Core Scripts

- **`evaluate_router_vs_quantized.py`** - Main evaluation script
- **`example_evaluate_router.py`** - Usage examples and demonstrations
- **`plot_training_history.py`** - Standalone plotting for training history
- **`train_layerwise_router.py`** - Router training script (with plotting)

### Support Files

- **`README_evaluation.md`** - This documentation
- **`LayerwiseQuantizeForCausalLM.py`** - Core model implementation

## Quick Start

### 1. Basic Evaluation (Default Directory)

```bash
# Evaluate router vs fixed precision models using default "router_checkpoints" directory
python evaluate_router_vs_quantized.py --num_examples 500
```

### 2. Custom Checkpoint Directory

```bash
# Evaluate with custom checkpoint directory
python evaluate_router_vs_quantized.py my_router_checkpoints/ --num_examples 500
```

### 3. Comprehensive Custom Evaluation

```bash
# Evaluate with all custom settings
python evaluate_router_vs_quantized.py my_checkpoints/ \
    --dataset c4 \
    --seq_len 256 \
    --num_examples 1000 \
    --batch_size 8 \
    --output_dir evaluation_results/
```

### 4. Quick Testing

```bash
# Fast evaluation for testing (uses default checkpoint directory)
python evaluate_router_vs_quantized.py \
    --num_examples 50 \
    --batch_size 1 \
    --device cpu \
    --no_plots
```

## Usage Examples

### Command Line Arguments

```bash
python evaluate_router_vs_quantized.py <checkpoint_dir> [options]

Required:
  checkpoint_dir        Directory containing router checkpoints

Options:
  --dataset DATASET     Dataset to use (default: c4)
  --seq_len LENGTH      Sequence length (default: 512)
  --num_examples N      Number of examples (default: 500)
  --batch_size N        Batch size (default: 4)
  --device DEVICE       Device: cuda/cpu (default: auto)
  --random_state SEED   Random seed (default: 42)
  --output_dir DIR      Output directory (default: checkpoint_dir)
  --no_plots           Skip generating plots
```

### Programmatic Usage

```python
from evaluate_router_vs_quantized import RouterEvaluator

evaluator = RouterEvaluator(
    checkpoint_dir="router_checkpoints/",
    num_examples=500,
    batch_size=4
)

evaluator.evaluate_all_models()
evaluator.save_results()
evaluator.plot_comparison()
evaluator.print_summary()
```

## Output Files

After evaluation, the following files are generated in the output directory:

### Results
- **`router_vs_quantized_results.json`** - Complete numerical results with metadata
- **Terminal output** - Summary table and best model identification

### Plots
- **`router_vs_quantized_comparison.png`** - 4-panel comparison (Total Loss, CE Loss, Precision Loss, Avg Precision)
- **`router_vs_quantized_comparison.pdf`** - Publication-ready PDF version
- **`loss_focused_comparison.png`** - Focused comparison of Total vs CE Loss

## Understanding Results

### Metrics Compared

1. **Total Loss** - Combined CE + Precision loss (lower is better)
2. **Cross Entropy Loss** - Language modeling performance (lower is better)
3. **Precision Loss** - Regularization term penalizing high precision (lower is better)
4. **Average Precision** - Mean bit-width used across layers

### Model Types

- **Router (Best)** - Trained router model using inference mode (one-hot precision selection)
- **X-bit Fixed** - Model with all layers set to X-bit precision (3, 4, 5, 6, 7, 8 bits)

### Interpretation

- **Best CE Loss** - Model with best language modeling performance
- **Best Total Loss** - Model with best trade-off between performance and efficiency
- **Router Performance** - Shows how well adaptive precision compares to fixed choices

## Expected Results

Typical evaluation scenarios:

### Scenario 1: Router Wins
- Router achieves lower total loss than any fixed precision
- Demonstrates value of adaptive precision selection
- Router selects different precisions for different layers optimally

### Scenario 2: Fixed Precision Wins
- A specific fixed precision (e.g., 6-bit) performs best
- May indicate router needs more training or different hyperparameters
- Could suggest optimal fixed precision for this model/dataset

### Scenario 3: Trade-off Analysis
- Router has slightly higher CE loss but much lower precision loss
- Shows efficiency gains from adaptive precision
- Useful for understanding performance vs. efficiency trade-offs

## Troubleshooting

### Common Issues

1. **Checkpoint not found**
   ```
   FileNotFoundError: Best checkpoint not found
   ```
   - Ensure `best_router_checkpoint.pt` exists in checkpoint directory
   - Check that training completed successfully

2. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce `--batch_size` or `--num_examples`
   - Use `--device cpu` for CPU evaluation
   - Reduce `--seq_len`

3. **Import errors**
   ```
   ModuleNotFoundError: No module named 'LayerwiseQuantizeForCausalLM'
   ```
   - Run from the correct directory (layerwise/ or project root)
   - Ensure all dependencies are installed

### Performance Tips

- **Fast testing**: Use `--num_examples 50 --batch_size 1` for quick verification
- **Comprehensive eval**: Use `--num_examples 1000+` for reliable comparisons
- **Memory efficiency**: Reduce batch size before reducing number of examples
- **CPU evaluation**: Useful for debugging, but much slower

## Integration with Training

The evaluation script automatically reads configuration from training checkpoints:
- Model path and tokenizer settings
- Available precisions
- Loss function weights (CE vs Precision)
- Training hyperparameters

This ensures evaluation uses exactly the same setup as training for fair comparison.

## Advanced Usage

### Custom Evaluation Datasets

To evaluate on different datasets, ensure they're supported by `get_tokens()` function:

```python
# Supported datasets include: c4, wikitext, ptb, etc.
evaluator = RouterEvaluator(
    checkpoint_dir="router_checkpoints/",
    dataset="wikitext",  # Custom dataset
    num_examples=500
)
```

### Batch Processing Multiple Checkpoints

```bash
# Evaluate multiple checkpoint directories
for checkpoint_dir in router_checkpoints_*/; do
    echo "Evaluating $checkpoint_dir"
    python evaluate_router_vs_quantized.py "$checkpoint_dir" \
        --output_dir "evaluation_results/$(basename $checkpoint_dir)"
done
```

### Plotting Existing Results

```python
# Plot previously saved results
import json
import matplotlib.pyplot as plt
from evaluate_router_vs_quantized import RouterEvaluator

# Load saved results
with open('router_vs_quantized_results.json', 'r') as f:
    data = json.load(f)

# Create evaluator just for plotting
evaluator = RouterEvaluator("dummy")  # Won't load model
evaluator.results = data['results']
evaluator.plot_comparison()
```

This evaluation framework provides comprehensive insights into router model performance and helps validate the effectiveness of adaptive precision selection!
