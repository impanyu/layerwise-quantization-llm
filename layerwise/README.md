# Layerwise Quantization for Causal Language Models

This module implements layerwise quantization with router networks for causal language models. The implementation adds fully connected neural network routers before each transformer layer to dynamically select quantization precision.

## Key Features

1. **Router Networks**: Each transformer layer has an associated router that outputs a one-hot vector indicating which quantization precision to use.

2. **Training Mode**: `train_forward()` method that mixes outputs from different quantization precisions based on router weights.

3. **Inference Mode**: `infer_forward()` method that selects the precision with maximum router weight for efficient inference.

4. **Parameter Freezing**: Original LM parameters are automatically frozen - only router parameters are trained.

5. **Compatible Imports**: Updated import statements to work with the any_precision module structure.

## Architecture

### Router Network
- **Input**: Hidden states from previous layer (or embeddings for the first router)
- **Output**: P-dimensional one-hot vector where P is the number of available precisions
- **Architecture**: 3-layer fully connected network with ReLU activation and dropout
- **Final Layer**: Softmax to ensure one-hot output

### Layer Processing
1. **Embedding Layer**: Token embeddings are processed normally
2. **Router Selection**: Router network determines precision for current layer
3. **Mixed Precision**: During training, outputs from all precisions are mixed based on router weights
4. **Single Precision**: During inference, only the highest-weight precision is used

## Usage

### Initialization
```python
from layerwise.LayerwiseQuantizeForCausalLM import LayerwiseQuantizeForCausalLM

model = LayerwiseQuantizeForCausalLM.from_quantized(
    quant_model_path="path/to/quantized/model",
    precisions=[4, 8, 16],  # Available precisions
    trust_remote_code=True
)
```

### Training
```python
# Use train_forward for training with mixed precision
output = model.train_forward(input_ids, attention_mask)
loss = criterion(output.logits, labels)
loss.backward()

# Only router parameters will be updated
optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
optimizer.step()
```

### Inference
```python
# Use infer_forward for efficient inference
output = model.infer_forward(input_ids, attention_mask)

# Or use regular forward (defaults to infer_forward)
output = model(input_ids, attention_mask)
```

### Generation
```python
# Generate text using the layerwise quantization
generated = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7
)
```

## Methods

### `train_forward(input_ids, attention_mask=None, **kwargs)`
- Processes input through each transformer layer
- For each layer, computes outputs with all available precisions
- Mixes outputs based on router network weights
- Returns mixed precision logits

### `infer_forward(input_ids, attention_mask=None, **kwargs)`
- Processes input through each transformer layer
- For each layer, selects precision with maximum router weight
- Uses the highest precision among all batches for optimal quality
- Uses only the selected precision for efficiency
- Returns single precision logits

### `generate(*args, **kwargs)`
- Modified to use `infer_forward` for generation
- Temporarily replaces model's forward method during generation
- Restores original forward method after generation

### `get_trainable_parameters()`
- Returns only router parameters that will be trained
- Useful for optimizer initialization

### `get_frozen_parameters()`
- Returns frozen original LM parameters
- Useful for verification

### `unfreeze_original_parameters()`
- Unfreezes original LM parameters if needed
- Use with caution as it will train the entire model

## Router Network Details

Each router consists of:
- **Input Layer**: Linear transformation from hidden_dim to 128
- **Hidden Layer**: Linear transformation from 128 to 128 with ReLU and dropout
- **Output Layer**: Linear transformation from 128 to num_precisions
- **Activation**: Softmax to produce probability distribution

The router takes the mean over the sequence dimension if input has multiple dimensions.

## Compatibility

- Compatible with LLaMA, Mistral, OPT, and other transformer-based models
- Uses the same architecture configurations as the original any_precision module
- Maintains compatibility with existing quantization infrastructure
