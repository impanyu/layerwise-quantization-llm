import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

try:
    # Relative imports (when used as module)
    from ..any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear
    from ..any_precision.analyzer.analyzer import get_analyzer
except ImportError:
    # Absolute imports (when run as script or imported differently)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear
    from any_precision.analyzer.analyzer import get_analyzer


def replace_module_by_name(layer, module_name, new_module):
    levels = module_name.split('.')
    module = layer
    for level in levels[:-1]:
        module = getattr(module, level) if not level.isdigit() else module[int(level)]
    setattr(module, levels[-1], new_module)


def sparsemax(input, dim=-1):
    """Sparsemax activation function.
    
    Sparsemax is a sparse alternative to softmax that can produce exactly zero probabilities.
    This encourages the router to select fewer precisions, leading to more efficient inference.
    
    Reference: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
    https://arxiv.org/abs/1602.02068
    
    Args:
        input: Input tensor of any shape
        dim: Dimension along which sparsemax will be computed
        
    Returns:
        Tensor of same shape as input with sparsemax applied along dim
    """
    # Translate input by max for numerical stability
    input_shifted = input - torch.max(input, dim=dim, keepdim=True)[0]
    
    # Sort input in descending order
    zs_sorted, _ = torch.sort(input_shifted, dim=dim, descending=True)
    
    # Calculate cumulative sums
    range_tensor = torch.arange(1, input.size(dim) + 1, dtype=input.dtype, device=input.device)
    if dim != -1 and dim != input.dim() - 1:
        # Reshape range_tensor to broadcast correctly
        shape = [1] * input.dim()
        shape[dim] = input.size(dim)
        range_tensor = range_tensor.view(shape)
    
    cumsum = torch.cumsum(zs_sorted, dim=dim)
    
    # Find the threshold k
    condition = 1 + range_tensor * zs_sorted > cumsum
    k = torch.sum(condition, dim=dim, keepdim=True)
    
    # Calculate tau (threshold)
    k_broadcast = k.expand_as(cumsum)
    tau = (torch.gather(cumsum, dim, k - 1) - 1) / k.float()
    
    # Apply sparsemax transformation
    output = torch.clamp(input_shifted - tau, min=0)
    
    return output





class Router(nn.Module):
    """Router network that outputs a one-hot vector for precision selection.
    
    Uses softmax activation by default. Sparsemax implementation is available
    in the sparsemax() function above if sparse outputs are needed in the future.
    """
    def __init__(self, input_dim, num_precisions, hidden_dim=128, dtype=None, use_sparsemax=False):
        super().__init__()
        # Always use float32 for router parameters to avoid numerical instability
        # Ignore the dtype parameter - router always uses float32 internally
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        self.fc3 = nn.Linear(hidden_dim, num_precisions, dtype=torch.float32)
        self.dropout = nn.Dropout(0.1)
        self.use_sparsemax = use_sparsemax
        
        # Initialize weights to prevent extreme values
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
            nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
            nn.init.xavier_uniform_(self.fc3.weight, gain=0.1)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)
            if self.fc3.bias is not None:
                nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x, num_real_tokens=None):
        # Use mean pooling if input has multiple dimensions
        if x.dim() > 2:
            # x should be (batch_size, seq_len, hidden_dim)
            x = x.mean(dim=1)  # Average over sequence length -> (batch_size, hidden_dim)
        
        # Store original dtype to convert back at the end
        original_dtype = x.dtype
        
        # Check for NaN in input and handle gracefully
        if torch.isnan(x).any():
            print(f"WARNING: NaN detected in router input, using uniform distribution")
            print(f"Input stats: min={x.min()}, max={x.max()}, mean={x.mean()}")
            # Return uniform distribution over precisions instead of propagating NaN
            # Make sure it requires grad to maintain gradient flow
            batch_size = x.shape[0]
            num_precisions = self.fc3.out_features
            uniform_output = torch.full((batch_size, num_precisions), 1.0/num_precisions, 
                                      device=x.device, dtype=original_dtype, requires_grad=True)
            return uniform_output
        
        # Convert to float32 for computation (router weights are already float32)
        x = x.float()
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Check for extreme values before activation
        if torch.isinf(x).any() or x.abs().max() > 50:
            print(f"WARNING: Extreme values before activation: min={x.min()}, max={x.max()}")
        
        # Apply activation function based on configuration
        if self.use_sparsemax:
            # Sparsemax for sparse precision selection
            x = sparsemax(x, dim=-1)
        else:
            # Softmax for smooth precision distribution (default)
            x = F.softmax(x, dim=-1)
        
        # Final safety check
        if torch.isnan(x).any():
            print(f"WARNING: NaN detected in router output, using uniform distribution")
            batch_size = x.shape[0]
            num_precisions = x.shape[1]
            uniform_output = torch.full((batch_size, num_precisions), 1.0/num_precisions, 
                                      device=x.device, dtype=original_dtype, requires_grad=True)
            return uniform_output
        
        # Convert back to original dtype (float16) for compatibility with model
        return x.to(original_dtype)


class LayerwiseQuantizeForCausalLM(nn.Module):
    def __init__(
            self,
            model_path,
            config,
            precisions=None,
            torch_dtype=torch.float16,
            fuse_layers=False,
            trust_remote_code=True,
            use_sparsemax=False,
    ):
        super().__init__()

        self.config = config
        self.use_sparsemax = use_sparsemax

        self.supported_bits = list(range(self.config.anyprec['seed_precision'],
                                         self.config.anyprec['parent_precision'] + 1))
        if precisions is None:
            self.precisions = self.supported_bits
        else:
            assert len(precisions) == len(set(precisions)), "Precisions must be unique"
            assert all(bit in self.supported_bits for bit in precisions), \
                f"Supported bits {precisions} must be a subset of model supported bits {self.supported_bits}"
            self.precisions = precisions

        self.precision = max(self.precisions)

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                # attn_implementation="flash_attention_2",
            )

        self.analyzer = get_analyzer(self.model)

        self.ap_linears = []
        # Replace to AnyPrecisionLinear layers
        self._load_quantized_modules()

        # Initialize routers
        self._initialize_routers()

        # Freeze original LM parameters
        self._freeze_original_parameters()

        self.tie_weights()

        device_map = {key: 'cpu' for key in self.model.state_dict().keys()}

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=model_path,
            device_map=device_map,
            no_split_module_classes=[self.layer_type],
            dtype=torch_dtype,
        )

        # Dispath to devices
        if fuse_layers:
            self.fuse_layers()

        self.prune_precisions()

    def _initialize_routers(self):
        """Initialize router networks for each layer."""
        self.routers = nn.ModuleList()
        layers = self.get_model_layers()
        
        # Router for embedding layer (before first transformer layer)
        embedding_dim = self.model.config.hidden_size
        model_dtype = next(self.model.parameters()).dtype
        self.embedding_router = Router(embedding_dim, len(self.precisions), use_sparsemax=self.use_sparsemax)
        
        # Routers for each transformer layer
        for _ in layers:
            self.routers.append(Router(embedding_dim, len(self.precisions), use_sparsemax=self.use_sparsemax))

    def _freeze_original_parameters(self):
        """Freeze all parameters from the original language model."""
        # Freeze the main model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Freeze AnyPrecisionLinear parameters (quantized weights and LUTs)
        for ap_linear in self.ap_linears:
            for param in ap_linear.parameters():
                param.requires_grad = False
            # Also freeze the buffers (quantized weights and LUTs)
            for buffer in ap_linear.buffers():
                buffer.requires_grad = False
        
        print(f"✓ Frozen {sum(p.numel() for p in self.model.parameters())} original LM parameters")
        print(f"✓ Frozen {sum(p.numel() for p in self.ap_linears[0].parameters()) * len(self.ap_linears)} AP linear parameters")
        print(f"✓ Only router parameters ({sum(p.numel() for p in self.routers.parameters())} parameters) will be trained")

    def get_trainable_parameters(self):
        """Return trainable parameters (only routers)."""
        trainable_params = []
        for router in self.routers:
            trainable_params.extend(router.parameters())
        trainable_params.extend(self.embedding_router.parameters())
        return trainable_params

    def get_frozen_parameters(self):
        """Return frozen parameters (original LM + AP linear layers)."""
        frozen_params = []
        frozen_params.extend(self.model.parameters())
        for ap_linear in self.ap_linears:
            frozen_params.extend(ap_linear.parameters())
        return frozen_params

    def unfreeze_original_parameters(self):
        """Unfreeze original LM parameters if needed."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        for ap_linear in self.ap_linears:
            for param in ap_linear.parameters():
                param.requires_grad = True
            for buffer in ap_linear.buffers():
                buffer.requires_grad = True
        
        print("✓ Unfrozen original LM parameters")

    def train_forward(self, input_ids, return_router_outputs=False, **kwargs):
        """Forward pass during training with mixed precision based on router outputs."""
        batch_size, seq_len = input_ids.shape
        
        # Get embedding output
        embeddings = self.model.get_input_embeddings()(input_ids)
        current_input = embeddings
        # Check for NaN in embeddings
        if torch.isnan(current_input).any():
            print(f"ERROR: NaN detected in initial embeddings!")
            # Reset to a safe state
            current_input = torch.zeros_like(current_input)
        
        # Process through each transformer layer with its own router
        layers = self.get_model_layers()
        router_outputs = []
        
        for layer_idx, layer in enumerate(layers):
            # Check for NaN in current input
            if torch.isnan(current_input).any():
                print(f"ERROR: NaN detected in layer {layer_idx} input, stopping training")
                break
                
            # For fixed-length sequences, all tokens are real
            num_real_tokens = torch.full((batch_size,), seq_len, dtype=torch.long, device=input_ids.device)
            
            # Get router output for this layer
            layer_router_output = self.routers[layer_idx](current_input, num_real_tokens)
            router_outputs.append(layer_router_output)
            
            # Initialize layer output
            layer_output = torch.zeros_like(current_input)
            
            # For each precision, compute layer output and mix
            for i, precision in enumerate(self.precisions):
                self.set_precision(precision)
                
                # Forward through this specific layer
                # Option 1: Gradient checkpointing (recommended for most cases)
                from torch.utils.checkpoint import checkpoint
                
                def layer_forward_fn(x):
                    """Wrapper function for gradient checkpointing."""
                    return layer(x)
                
                layer_output_precision = checkpoint(
                    layer_forward_fn, 
                    current_input, 
                    use_reentrant=False  # Use newer, more memory-efficient checkpointing
                )
                
                # Option 2: Custom autograd function (maximum memory efficiency)
                # Uncomment below and comment above to use custom autograd:
                # 
                # class LayerForwardInputGradOnly(torch.autograd.Function):
                #     @staticmethod
                #     def forward(ctx, input_tensor, layer_module):
                #         ctx.layer_module = layer_module
                #         ctx.save_for_backward(input_tensor)
                #         with torch.no_grad():
                #             return layer_module(input_tensor)
                #     
                #     @staticmethod
                #     def backward(ctx, grad_output):
                #         input_tensor, = ctx.saved_tensors
                #         layer_module = ctx.layer_module
                #         input_for_grad = input_tensor.detach().requires_grad_(True)
                #         with torch.enable_grad():
                #             output = layer_module(input_for_grad)
                #         grad_input = torch.autograd.grad(output, input_for_grad, grad_output)[0]
                #         return grad_input, None
                # 
                # layer_output_precision = LayerForwardInputGradOnly.apply(current_input, layer)
                
      
                # Mix based on router weights
                precision_weight = layer_router_output[:, i:i+1].unsqueeze(-1)
                if isinstance(layer_output_precision, tuple):
                    layer_output += precision_weight * layer_output_precision[0]
                else:
                    layer_output += precision_weight * layer_output_precision
            
            current_input = layer_output
        
        # Final output processing through lm_head
        final_output = self.model.lm_head(current_input)
        
        if return_router_outputs:
            return type('Outputs', (), {'logits': final_output, 'router_outputs': router_outputs})()
        else:
            return type('Outputs', (), {'logits': final_output})()

    def infer_forward(self, input_ids, return_router_outputs=False, **kwargs):
        """Forward pass during inference using the precision with maximum router weight."""
        batch_size, seq_len = input_ids.shape
        
        # Get embedding output
        embeddings = self.model.get_input_embeddings()(input_ids)
        current_input = embeddings
        
        # Process through each transformer layer
        layers = self.get_model_layers()
        router_outputs = [] if return_router_outputs else None
        
        for layer_idx, layer in enumerate(layers):
            # For fixed-length sequences, all tokens are real
            num_real_tokens = torch.full((batch_size,), seq_len, dtype=torch.long, device=input_ids.device)
            
            # Get router output for this layer
            layer_router_output = self.routers[layer_idx](current_input, num_real_tokens)
            layer_precision_idx = torch.argmax(layer_router_output, dim=-1)
            # Use maximum precision among all batches for highest quality
            layer_precision_idx = torch.max(layer_precision_idx).item()
            layer_precision = self.precisions[layer_precision_idx]
            
            # Convert selected precision to one-hot vector for consistency with train_forward
            if return_router_outputs:
                # Create one-hot vector representing the selected precision
                one_hot_output = torch.zeros_like(layer_router_output)
                one_hot_output[:, layer_precision_idx] = 1.0
                router_outputs.append(one_hot_output)
            
            # Set precision and forward through layer
            self.set_precision(layer_precision)
            with torch.no_grad():
                # For fixed-length sequences, let the model handle causal masking internally
                layer_output = layer(current_input)
            
            if isinstance(layer_output, tuple):
                current_input = layer_output[0]
            else:
                current_input = layer_output
        
        # Final output through lm_head
        final_output = self.model.lm_head(current_input)
        
        if return_router_outputs:
            return type('Outputs', (), {'logits': final_output, 'router_outputs': router_outputs})()
        else:
            return type('Outputs', (), {'logits': final_output})()

    def forward(self, input_ids, **kwargs):
        # Default to infer_forward
        return self.infer_forward(input_ids, **kwargs)

    def generate(self, *args, **kwargs):
        """Modified generate method to use infer_forward."""
        if 'precision' in kwargs:
            prev_precision = self.precision
            precision = kwargs.pop('precision')
            self.set_precision(precision)
        else:
            prev_precision = self.precision

        with torch.inference_mode():
            # Use the original model's generate method but with our custom forward
            # We need to temporarily replace the forward method
            original_forward = self.model.forward
            self.model.forward = self.infer_forward
            
            try:
                results = self.model.generate(*args, **kwargs)
            finally:
                # Restore original forward method
                self.model.forward = original_forward

        self.set_precision(prev_precision)
        return results

    @staticmethod
    def _load_config(
            model_path,
            trust_remote_code=True,
    ):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return config

    @classmethod
    def from_quantized(
            cls,
            quant_model_path,
            trust_remote_code=True,
            fuse_layers=False,
            precisions=None,
            use_sparsemax=False
    ):
        config = cls._load_config(quant_model_path, trust_remote_code)

        ap_model = cls(
            model_path=quant_model_path,
            precisions=precisions,
            config=config,
            fuse_layers=fuse_layers,
            trust_remote_code=trust_remote_code,
            use_sparsemax=use_sparsemax,
        )

        return ap_model

    def _load_quantized_modules(self):
        # Get blocks of model
        layers = self.analyzer.get_layers()

        for layer in tqdm(layers, desc="Loading AP Layers"):
            # Get every linear layer in a block
            named_linears = self.analyzer.get_modules(layer)

            # Replace nn.Linear with AnyPrecisionLinear
            for name, module in named_linears.items():
                wqlinear = AnyPrecisionLinear(
                    module.in_features, module.out_features,
                    self.supported_bits,
                    bias=module.bias is not None,
                    precisions=self.precisions,
                    device=module.weight.device,
                )
                self.ap_linears.append(wqlinear)
                replace_module_by_name(layer, name, wqlinear)

            torch.cuda.empty_cache()
            gc.collect()

    def prune_precisions(self):
        for ap_linear in self.ap_linears:
            ap_linear.prune_precisions()

        torch.cuda.empty_cache()
        gc.collect()

    def set_precision(self, precision):
        for ap_linear in self.ap_linears:
            ap_linear.set_precision(precision)
        self.precision = precision

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def get_model_layers(self):
        module = self.model
        for attrib_name in self.config.anyprec['arch_config']['model_name'].split('.'):
            module = getattr(module, attrib_name)
        return getattr(module, self.config.anyprec['arch_config']['layers_name'])

    def fuse_layers(self):
        if 'fuse_target_layers' not in self.config:
            raise NotImplementedError("This model does not support layer fusion")
        # TODO implement layer fusion
        pass

    @property
    def layer_type(self):
        for layer in self.get_model_layers():
            layer_class_name = layer.__class__.__name__
            if layer_class_name.endswith("DecoderLayer"):
                return layer_class_name
        return None

    @property
    def device(self):
        return self.model.device
