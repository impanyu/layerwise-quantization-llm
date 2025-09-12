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

    def train_forward(self, return_router_outputs=False, **model_kwargs):
        """Train forward: mimic HF forward path, but mix per-layer outputs by router weights.

        Accepts the same kwargs as `self.model.forward` (e.g., input_ids, attention_mask, labels, ...)
        and returns the same HF output object. Optionally also returns router_outputs.
        """
        layers = self.get_model_layers()
        original_forwards = [layer.forward for layer in layers]
        router_outputs = []

        attention_mask = model_kwargs.get('attention_mask', None)

        # Precompute num_real_tokens per sample if attention_mask is provided
        def compute_num_real_tokens(hidden_states):
            batch_size, seq_len = hidden_states.shape[:2]
            if attention_mask is not None:
                return attention_mask.to(hidden_states.device).sum(dim=1)
            return torch.full((batch_size,), seq_len, dtype=torch.long, device=hidden_states.device)

        # Define mixed forward wrapper per layer
        def make_mixed_forward(layer_idx, orig_fwd):
            def mixed_forward(hidden_states, *args, **kwargs):
                num_real_tokens = compute_num_real_tokens(hidden_states)
                layer_router_output = self.routers[layer_idx](hidden_states, num_real_tokens)
                if return_router_outputs:
                    router_outputs.append(layer_router_output)

                mixed_hidden = torch.zeros_like(hidden_states)
                cached_first_out = None

                # Use memory-efficient custom autograd approach
                # This avoids gradient checkpointing shape mismatches while preserving gradients
                
                for i, precision in enumerate(self.precisions):
                    # Set precision for this forward pass
                    self.set_precision(precision)
                    
                    # Define the custom autograd function outside the loop to avoid closure issues
                    def create_precision_autograd_fn(ap_linears_ref):
                        class PrecisionForward(torch.autograd.Function):
                            @staticmethod
                            def forward(ctx, hs, layer_fn, target_precision, ap_linears):
                                # Store what we need for backward
                                ctx.layer_fn = layer_fn
                                ctx.target_precision = target_precision
                                ctx.ap_linears = ap_linears
                                ctx.save_for_backward(hs)
                                
                                # Forward pass without storing intermediate activations
                                with torch.no_grad():
                                    # Ensure precision is set
                                    for ap_linear in ap_linears:
                                        ap_linear.set_precision(target_precision)
                                    output = layer_fn(hs, *args, **kwargs)
                                return output
                            
                            @staticmethod
                            def backward(ctx, grad_output):
                                # Retrieve stored data
                                hs, = ctx.saved_tensors
                                layer_fn = ctx.layer_fn
                                target_precision = ctx.target_precision
                                ap_linears = ctx.ap_linears
                                
                                # Create fresh input with gradients
                                hs_grad = hs.detach().requires_grad_(True)
                                
                                # Recompute with gradients enabled
                                with torch.enable_grad():
                                    # Set precision for recomputation
                                    for ap_linear in ap_linears:
                                        ap_linear.set_precision(target_precision)
                                    output = layer_fn(hs_grad, *args, **kwargs)
                                
                                # Get the right output tensor
                                if isinstance(output, tuple):
                                    output_tensor = output[0]
                                else:
                                    output_tensor = output
                                
                                # Compute gradients only for input
                                grad_input = torch.autograd.grad(
                                    outputs=output_tensor,
                                    inputs=hs_grad,
                                    grad_outputs=grad_output if not isinstance(grad_output, tuple) else grad_output[0],
                                    retain_graph=False,
                                    create_graph=False,
                                    only_inputs=True
                                )[0]
                                
                                return grad_input, None, None, None
                        
                        return PrecisionForward
                    
                    # Create the autograd function class
                    PrecisionForward = create_precision_autograd_fn(self.ap_linears)
                    
                    # Use custom autograd function
                    out_i = PrecisionForward.apply(hidden_states, orig_fwd, precision, self.ap_linears)

                    if isinstance(out_i, tuple):
                        hs_i = out_i[0]
                        if cached_first_out is None:
                            cached_first_out = out_i
                    else:
                        hs_i = out_i

                    # Get the weight for this precision
                    weight = layer_router_output[:, i:i+1].unsqueeze(-1)
                    
                    # Accumulate weighted output
                    mixed_hidden = mixed_hidden + weight * hs_i

                # Rebuild return matching original
                if isinstance(cached_first_out, tuple):
                    rebuilt = (mixed_hidden,) + tuple(cached_first_out[1:])
                    return rebuilt
                return mixed_hidden
            return mixed_forward

        # Patch forwards
        for idx, layer in enumerate(layers):
            layer.forward = make_mixed_forward(idx, original_forwards[idx])

        try:
            outputs = self.model.forward(**model_kwargs)
        finally:
            # Restore forwards
            for layer, orig in zip(layers, original_forwards):
                layer.forward = orig

        if return_router_outputs:
            # Attach for convenience while preserving HF output
            if isinstance(outputs, tuple):
                return type('Outputs', (), {'logits': outputs[0], 'router_outputs': router_outputs})()
            if hasattr(outputs, 'logits'):
                obj = outputs
                obj.router_outputs = router_outputs
                return obj
            return type('Outputs', (), {'logits': outputs, 'router_outputs': router_outputs})()
        return outputs

    def infer_forward(self, return_router_outputs=False, **model_kwargs):
        """Inference forward: mimic HF forward path, but select one precision per layer via router.

        Accepts the same kwargs as `self.model.forward` and returns the same HF output object.
        """
        layers = self.get_model_layers()
        original_forwards = [layer.forward for layer in layers]
        collected_router = [] if return_router_outputs else None

        attention_mask = model_kwargs.get('attention_mask', None)

        def compute_num_real_tokens(hidden_states):
            batch_size, seq_len = hidden_states.shape[:2]
            if attention_mask is not None:
                return attention_mask.to(hidden_states.device).sum(dim=1)
            return torch.full((batch_size,), seq_len, dtype=torch.long, device=hidden_states.device)

        def make_selected_forward(layer_idx, orig_fwd):
            def selected_forward(hidden_states, *args, **kwargs):
                num_real_tokens = compute_num_real_tokens(hidden_states)
                layer_router_output = self.routers[layer_idx](hidden_states, num_real_tokens)
                # Choose one precision for whole batch (argmax of batch-average)
                avg_weights = layer_router_output.mean(dim=0)
                sel_idx = torch.argmax(avg_weights).item()
                sel_precision = self.precisions[sel_idx]

                if return_router_outputs:
                    one_hot = torch.zeros_like(layer_router_output)
                    one_hot[:, sel_idx] = 1.0
                    collected_router.append(one_hot)

                self.set_precision(sel_precision)
                with torch.no_grad():
                    return orig_fwd(hidden_states, *args, **kwargs)
            return selected_forward

        # Patch forwards
        for idx, layer in enumerate(layers):
            layer.forward = make_selected_forward(idx, original_forwards[idx])

        try:
            outputs = self.model.forward(**model_kwargs)
        finally:
            for layer, orig in zip(layers, original_forwards):
                layer.forward = orig

        if return_router_outputs:
            if isinstance(outputs, tuple):
                return type('Outputs', (), {'logits': outputs[0], 'router_outputs': collected_router})()
            if hasattr(outputs, 'logits'):
                obj = outputs
                obj.router_outputs = collected_router
                return obj
            return type('Outputs', (), {'logits': outputs, 'router_outputs': collected_router})()
        return outputs

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
