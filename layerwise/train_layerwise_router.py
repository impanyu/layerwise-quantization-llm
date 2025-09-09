import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import json
from datetime import datetime

try:
    # Relative imports (when run as module)
    from .LayerwiseQuantizeForCausalLM import LayerwiseQuantizeForCausalLM
    from ..any_precision.quantization.datautils import get_tokens
except ImportError:
    # Absolute imports (when run as script)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from layerwise.LayerwiseQuantizeForCausalLM import LayerwiseQuantizeForCausalLM
    from any_precision.quantization.datautils import get_tokens


# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


class RouterTrainer:
    def __init__(
        self,
        model_path,
        dataset='c4',
        seq_len=512,
        num_examples=1000,
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=10,
        weight_ce=0.7,
        weight_precision=0.3,
        save_dir='router_checkpoints',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        random_state=42,
        trust_remote_code=True,
        precisions=None,
        validation_split=0.2,
        validation_examples=None
    ):
        self.model_path = model_path
        self.dataset = dataset
        self.seq_len = seq_len
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_ce = weight_ce
        self.weight_precision = weight_precision
        self.save_dir = save_dir
        self.device = device
        self.random_state = random_state
        self.trust_remote_code = trust_remote_code
        self.precisions = precisions
        self.validation_split = validation_split
        self.validation_examples = validation_examples
        
        # Validate weights
        assert abs(self.weight_ce + self.weight_precision - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {self.weight_ce + self.weight_precision}"
        
        # Setup
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_total_loss': [],
            'train_ce_loss': [],
            'train_precision_loss': [],
            'train_avg_precision': [],
            'val_total_loss': [],
            'val_ce_loss': [],
            'val_precision_loss': [],
            'val_avg_precision': []
        }
    
    def setup_model(self):
        """Load the layerwise quantized model."""
        logging.info(f"Loading model from {self.model_path}")
        
        self.model = LayerwiseQuantizeForCausalLM.from_quantized(
            quant_model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            precisions=self.precisions
        )
        
        self.model.to(self.device)
        self.model.train()  # Set to training mode
        
        # Get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=self.trust_remote_code
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info(f"Model loaded successfully. Device: {self.device}")
        logging.info(f"Available precisions: {self.model.precisions}")
        logging.info(f"Number of trainable parameters: {sum(p.numel() for p in self.model.get_trainable_parameters())}")
    
    def setup_data(self):
        """Setup training and validation data using the same mechanism as quantize.py."""
        logging.info(f"Loading dataset: {self.dataset}")
        
        # Calculate number of training and validation examples
        if self.validation_examples is None:
            val_examples = int(self.num_examples * self.validation_split)
            train_examples = self.num_examples
        else:
            train_examples = self.num_examples
            val_examples = self.validation_examples
        
        # Load training data from 'train' split
        logging.info(f"Loading {train_examples} training examples from 'train' split")
        train_tokens = get_tokens(
            self.dataset, 
            'train', 
            self.tokenizer, 
            self.seq_len, 
            train_examples, 
            seed=self.random_state
        )
        
        # Load validation data from 'validation' split
        logging.info(f"Loading {val_examples} validation examples from 'validation' split")
        val_tokens = get_tokens(
            self.dataset, 
            'validation', 
            self.tokenizer, 
            self.seq_len, 
            val_examples, 
            seed=self.random_state
        )
        
        # Process training data
        self.train_input_ids, self.train_attention_masks = self._process_tokens(train_tokens)
        
        # Process validation data
        self.val_input_ids, self.val_attention_masks = self._process_tokens(val_tokens)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(self.train_input_ids, self.train_attention_masks)
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        val_dataset = TensorDataset(self.val_input_ids, self.val_attention_masks)
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=False
        )
        
        logging.info(f"Training dataset: {len(self.train_input_ids)} samples")
        logging.info(f"Validation dataset: {len(self.val_input_ids)} samples")
        logging.info(f"Batch size: {self.batch_size}")
    
    def _process_tokens(self, tokens_list):
        """Process a list of token sequences into padded tensors."""
        input_ids = []
        attention_masks = []
        
        for tokens in tokens_list:
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones(len(tokens), dtype=torch.long)
            input_ids.append(tokens)
            attention_masks.append(attention_mask)
        
        # Pad sequences to the same length
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            # Pad input_ids
            padding_length = max_len - len(ids)
            padded_ids = torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)])
            padded_input_ids.append(padded_ids)
            
            # Pad attention_mask
            padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
            padded_attention_masks.append(padded_mask)
        
        # Convert to tensors
        input_ids_tensor = torch.stack(padded_input_ids)
        attention_masks_tensor = torch.stack(padded_attention_masks)
        
        return input_ids_tensor, attention_masks_tensor
    
    def setup_optimizer(self):
        """Setup optimizer for router parameters only."""
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = optim.AdamW(trainable_params, lr=self.learning_rate)
        
        logging.info(f"Optimizer setup: AdamW with lr={self.learning_rate}")
    
    def calculate_average_precision(self, router_outputs):
        """Calculate the average precision used across all layers."""
        # router_outputs is a list of router outputs for each layer
        # Each router output has shape [batch_size, num_precisions]
        
        batch_size = router_outputs[0].shape[0]
        num_layers = len(router_outputs)
        
        # For each sample in the batch, calculate the average precision
        avg_precisions = []
        
        for batch_idx in range(batch_size):
            layer_precisions = []
            for layer_idx in range(num_layers):
                # Get the precision weights for this layer and sample
                precision_weights = router_outputs[layer_idx][batch_idx]  # [num_precisions]
                
                # Calculate weighted average precision
                weighted_precision = torch.sum(precision_weights * torch.tensor(self.model.precisions, device=self.device, dtype=torch.float32))
                layer_precisions.append(weighted_precision)
            
            # Average across layers
            avg_precision = torch.mean(torch.stack(layer_precisions))
            avg_precisions.append(avg_precision)
        
        return torch.stack(avg_precisions)  # [batch_size]
    
    def normalize_precision_loss(self, avg_precision):
        """Normalize precision loss to be comparable to cross entropy."""
        # Cross entropy typically ranges from 0 to ~10 for language models
        # We normalize precision to a similar range
        # Assuming precision range is from min_precision to max_precision
        min_precision = min(self.model.precisions)
        max_precision = max(self.model.precisions)
        
        # Normalize to [0, 1] then scale to typical CE range
        normalized = (avg_precision - min_precision) / (max_precision - min_precision)
        # Scale to typical CE range (0-10)
        scaled = normalized * 200.0
        
        
        return scaled
    
    def custom_loss(self, outputs, labels, router_outputs):
        """Custom loss function combining cross entropy and precision penalty."""
        # Debug: Check for NaN in outputs
        if torch.isnan(outputs.logits).any():
            print("ERROR: NaN detected in model outputs.logits")
            print(f"Logits shape: {outputs.logits.shape}")
            print(f"Logits min: {outputs.logits.min()}, max: {outputs.logits.max()}")
        
        # Cross entropy loss
        ce_loss = nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        
        # Debug: Check CE loss
        if torch.isnan(ce_loss):
            print("ERROR: NaN detected in CE loss")
            print(f"CE loss: {ce_loss}")
        
        # Debug: Check router outputs
        for i, router_output in enumerate(router_outputs):
            if torch.isnan(router_output).any():
                print(f"ERROR: NaN detected in router_outputs[{i}]")
                print(f"Router output shape: {router_output.shape}")
                print(f"Router output: {router_output}")
        
        # Calculate average precision
        avg_precision = self.calculate_average_precision(router_outputs)
        
        # Debug: Check avg precision
        if torch.isnan(avg_precision).any():
            print("ERROR: NaN detected in avg_precision")
            print(f"avg_precision: {avg_precision}")
        
        # Normalize precision loss
        normalized_precision = self.normalize_precision_loss(avg_precision)
        precision_loss = torch.mean(normalized_precision)
        

        # Debug: Check precision loss
        if torch.isnan(precision_loss):
            print("ERROR: NaN detected in precision_loss")
            print(f"precision_loss: {precision_loss}")
            print(f"normalized_precision: {normalized_precision}")
        
        # Weighted combination with router regularization
        total_loss = self.weight_ce * ce_loss + self.weight_precision * precision_loss 
        # Debug: Check total loss
        if torch.isnan(total_loss):
            print("ERROR: NaN detected in total_loss")
            print(f"total_loss: {total_loss}")
            print(f"ce_loss: {ce_loss}, precision_loss: {precision_loss}")
            print(f"weights: ce={self.weight_ce}, precision={self.weight_precision}")
        
        return total_loss, ce_loss, precision_loss, torch.mean(avg_precision)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_precision_loss = 0.0
        total_avg_precision = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, (input_ids, attention_mask) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            # attention_mask no longer needed for fixed-length sequences
            
            # Forward pass with router outputs collection
            outputs = self.model.train_forward(
                input_ids=input_ids,
                return_router_outputs=True
            )
            
            router_outputs = outputs.router_outputs
            
            # Calculate loss
            loss, ce_loss, precision_loss, avg_precision = self.custom_loss(
                outputs, input_ids, router_outputs
            )
            
            # Check if loss requires grad
            if not loss.requires_grad:
                print(f"WARNING: Loss doesn't require grad. CE requires grad: {ce_loss.requires_grad}, Precision requires grad: {precision_loss.requires_grad}")
                print("Skipping backward pass")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            #torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=0.1)
            
            # Check for NaN in gradients before optimizer step
            has_nan_grad = False
            nan_param_names = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    nan_param_names.append(name)
            
            if has_nan_grad:
                print(f"ERROR: NaN detected in gradients for parameters: {nan_param_names[:5]}...")  # Show first 5
                print(f"Loss values: total={loss.item():.6f}, ce={ce_loss.item():.6f}, prec={precision_loss.item():.6f}")
                print(f"Batch {batch_idx}: Skipping optimizer step due to NaN gradients")
                continue
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_precision_loss += precision_loss.item()
            total_avg_precision += avg_precision.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Prec': f'{precision_loss.item():.4f}',
                'AvgPrec': f'{avg_precision.item():.2f}'
            })
        
        # Calculate averages with safety check
        if num_batches == 0:
            logging.error("No valid training batches processed! All gradients were NaN.")
            logging.error("This indicates a serious numerical instability issue.")
            return float('inf'), float('inf'), float('inf'), 0.0
        
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_precision_loss = total_precision_loss / num_batches
        avg_precision = total_avg_precision / num_batches
        
        return avg_loss, avg_ce_loss, avg_precision_loss, avg_precision
    
    def validate_epoch(self, epoch):
        """Validate for one epoch using inference mode."""
        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_precision_loss = 0.0
        total_avg_precision = 0.0
        num_batches = 0
        sample_router_outputs = None
        
        # Randomly select a batch index to sample router outputs from
        total_val_batches = len(self.val_dataloader)
        random_batch_idx = np.random.randint(0, total_val_batches) if total_val_batches > 0 else 0
        
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask) in enumerate(self.val_dataloader):
                input_ids = input_ids.to(self.device)
                
                # Use infer_forward with router outputs for realistic validation
                outputs = self.model.infer_forward(
                    input_ids=input_ids,
                    return_router_outputs=True
                )
                
                router_outputs = outputs.router_outputs
                
                # Sample router outputs from the randomly selected batch for logging
                if batch_idx == random_batch_idx:
                    sample_router_outputs = self._sample_router_outputs_for_logging(router_outputs, epoch, batch_idx)
                
                # Calculate loss using the same custom loss function
                # Router outputs are now one-hot vectors representing selected precisions
                loss, ce_loss, precision_loss, avg_precision = self.custom_loss(
                    outputs, input_ids, router_outputs
                )
                
                # Update statistics
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_precision_loss += precision_loss.item()
                total_avg_precision += avg_precision.item()
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_ce_loss = total_ce_loss / num_batches if num_batches > 0 else 0.0
        avg_precision_loss = total_precision_loss / num_batches if num_batches > 0 else 0.0
        avg_precision = total_avg_precision / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_ce_loss, avg_precision_loss, avg_precision, sample_router_outputs
    
    def _sample_router_outputs_for_logging(self, router_outputs, epoch, batch_idx=None):
        """Sample and format router outputs for logging."""
        num_layers = len(router_outputs)
        sampled_outputs = []
        
        # Take the first sample from the batch (index 0) for each layer
        for layer_idx in range(num_layers):
            layer_output = router_outputs[layer_idx][0]  # Shape: [num_precisions]
            
            # Find the selected precision (argmax of one-hot vector)
            selected_precision_idx = torch.argmax(layer_output).item()
            selected_precision = self.model.precisions[selected_precision_idx]
            
            # Create a formatted output for this layer
            layer_info = {
                'layer': layer_idx,
                'selected_precision': selected_precision,
                'precision_idx': selected_precision_idx,
                'router_output': layer_output.cpu().numpy().tolist()
            }
            sampled_outputs.append(layer_info)
        
        # Log the router outputs
        self._log_router_outputs(epoch, sampled_outputs, batch_idx)
        
        return sampled_outputs
    
    def _log_router_outputs(self, epoch, sampled_outputs, batch_idx=None):
        """Log router outputs in a readable format."""
        batch_info = f" (Batch {batch_idx})" if batch_idx is not None else ""
        logging.info(f"\n--- Validation Router Outputs (Epoch {epoch+1}{batch_info}) ---")
        
        # Create a summary line showing selected precisions for all layers
        selected_precisions = [layer['selected_precision'] for layer in sampled_outputs]
        logging.info(f"Selected precisions by layer: {selected_precisions}")
        
        # Log detailed information for each layer
        for layer_info in sampled_outputs:
            layer_idx = layer_info['layer']
            selected_prec = layer_info['selected_precision']
            precision_idx = layer_info['precision_idx']
            
            # Format router output (show which precision was selected)
            router_str = f"[{', '.join(['1.0' if i == precision_idx else '0.0' for i in range(len(self.model.precisions))])}]"
            
            logging.info(f"  Layer {layer_idx:2d}: precision {selected_prec}-bit (idx {precision_idx}) -> {router_str}")
        
        logging.info("--- End Router Outputs ---\n")
        
        # Also save to file for later analysis
        self._save_router_outputs_to_file(epoch, sampled_outputs, batch_idx)
    
    def _save_router_outputs_to_file(self, epoch, sampled_outputs, batch_idx=None):
        """Save router outputs to a JSON file for analysis."""
        router_log_path = os.path.join(self.save_dir, 'validation_router_outputs.jsonl')
        
        # Create entry for this epoch
        epoch_entry = {
            'epoch': epoch + 1,
            'sampled_batch': batch_idx,
            'layers': sampled_outputs,
            'summary': {
                'selected_precisions': [layer['selected_precision'] for layer in sampled_outputs],
                'avg_precision': np.mean([layer['selected_precision'] for layer in sampled_outputs]),
                'precision_distribution': {
                    str(prec): sum(1 for layer in sampled_outputs if layer['selected_precision'] == prec)
                    for prec in self.model.precisions
                }
            }
        }
        
        # Append to JSONL file (one JSON object per line)
        with open(router_log_path, 'a') as f:
            json.dump(epoch_entry, f)
            f.write('\n')
        
        if epoch == 0:  # Only log on first epoch to avoid spam
            logging.info(f"Router outputs being saved to: {router_log_path}")
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'model_path': self.model_path,
                'dataset': self.dataset,
                'seq_len': self.seq_len,
                'num_examples': self.num_examples,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_ce': self.weight_ce,
                'weight_precision': self.weight_precision,
                'precisions': self.model.precisions,
                'random_state': self.random_state,
                'validation_split': self.validation_split,
                'validation_examples': self.validation_examples
            }
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'router_checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        logging.info(f"Training history saved: {history_path}")
        
        # Also save a summary
        summary_path = os.path.join(self.save_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Router Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Model: {self.model_path}\n")
            f.write(f"  - Dataset: {self.dataset}\n")
            f.write(f"  - Sequence length: {self.seq_len}\n")
            f.write(f"  - Training examples: {len(self.train_input_ids)}\n")
            f.write(f"  - Validation examples: {len(self.val_input_ids)}\n")
            f.write(f"  - Batch size: {self.batch_size}\n")
            f.write(f"  - Learning rate: {self.learning_rate}\n")
            f.write(f"  - Epochs: {self.num_epochs}\n")
            f.write(f"  - Validation split: {self.validation_split}\n")
            f.write(f"  - CE weight: {self.weight_ce}\n")
            f.write(f"  - Precision weight: {self.weight_precision}\n")
            f.write(f"  - Available precisions: {self.model.precisions}\n\n")
            
            if len(self.train_history['epoch']) > 0:
                f.write("Final Results:\n")
                f.write(f"  - Final training loss: {self.train_history['train_total_loss'][-1]:.4f}\n")
                f.write(f"  - Final validation loss: {self.train_history['val_total_loss'][-1]:.4f}\n")
                f.write(f"  - Best validation loss: {min(self.train_history['val_total_loss']):.4f}\n")
                f.write(f"  - Final training avg precision: {self.train_history['train_avg_precision'][-1]:.2f}\n")
                f.write(f"  - Final validation avg selected precision: {self.train_history['val_avg_precision'][-1]:.2f}\n")
                f.write(f"\nNote: Validation uses inference mode with one-hot precision selection\n")
        
        logging.info(f"Training summary saved: {summary_path}")
    
    def train(self):
        """Main training loop."""
        logging.info("Starting router training...")
        logging.info(f"Training configuration:")
        logging.info(f"  - Dataset: {self.dataset}")
        logging.info(f"  - Sequence length: {self.seq_len}")
        logging.info(f"  - Number of examples: {self.num_examples}")
        logging.info(f"  - Batch size: {self.batch_size}")
        logging.info(f"  - Learning rate: {self.learning_rate}")
        logging.info(f"  - Number of epochs: {self.num_epochs}")
        logging.info(f"  - CE weight: {self.weight_ce}")
        logging.info(f"  - Precision weight: {self.weight_precision}")
        logging.info(f"  - Validation split: {self.validation_split}")
        if self.validation_examples is not None:
            logging.info(f"  - Validation examples: {self.validation_examples}")
        logging.info(f"  - Available precisions: {self.model.precisions}")
        
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logging.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train for one epoch
            train_avg_loss, train_avg_ce_loss, train_avg_precision_loss, train_avg_precision = self.train_epoch(epoch)
            
            # Validate for one epoch
            val_avg_loss, val_avg_ce_loss, val_avg_precision_loss, val_avg_precision, sample_router_outputs = self.validate_epoch(epoch)
            
            # Log metrics
            logging.info(f"Epoch {epoch+1} Results:")
            logging.info(f"  Training:")
            logging.info(f"    - Total Loss: {train_avg_loss:.4f}")
            logging.info(f"    - CE Loss: {train_avg_ce_loss:.4f}")
            logging.info(f"    - Precision Loss: {train_avg_precision_loss:.4f}")
            logging.info(f"    - Average Precision: {train_avg_precision:.2f}")
            logging.info(f"  Validation (inference mode):")
            logging.info(f"    - Total Loss: {val_avg_loss:.4f}")
            logging.info(f"    - CE Loss: {val_avg_ce_loss:.4f}")
            logging.info(f"    - Precision Loss: {val_avg_precision_loss:.4f}")
            logging.info(f"    - Avg Selected Precision: {val_avg_precision:.2f}")
            
            # Save to history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_total_loss'].append(train_avg_loss)
            self.train_history['train_ce_loss'].append(train_avg_ce_loss)
            self.train_history['train_precision_loss'].append(train_avg_precision_loss)
            self.train_history['train_avg_precision'].append(train_avg_precision)
            self.train_history['val_total_loss'].append(val_avg_loss)
            self.train_history['val_ce_loss'].append(val_avg_ce_loss)
            self.train_history['val_precision_loss'].append(val_avg_precision_loss)
            self.train_history['val_avg_precision'].append(val_avg_precision)
            
            # Save checkpoint
            metrics = {
                'train_total_loss': train_avg_loss,
                'train_ce_loss': train_avg_ce_loss,
                'train_precision_loss': train_avg_precision_loss,
                'train_avg_precision': train_avg_precision,
                'val_total_loss': val_avg_loss,
                'val_ce_loss': val_avg_ce_loss,
                'val_precision_loss': val_avg_precision_loss,
                'val_avg_precision': val_avg_precision
            }
            self.save_checkpoint(epoch, metrics)
            
            # Save best model based on validation loss
            if val_avg_loss < best_loss:
                best_loss = val_avg_loss
                best_checkpoint_path = os.path.join(self.save_dir, 'best_router_checkpoint.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': metrics,
                    'config': {
                        'model_path': self.model_path,
                        'dataset': self.dataset,
                        'seq_len': self.seq_len,
                        'num_examples': self.num_examples,
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate,
                        'weight_ce': self.weight_ce,
                        'weight_precision': self.weight_precision,
                        'precisions': self.model.precisions,
                        'random_state': self.random_state,
                        'validation_split': self.validation_split,
                        'validation_examples': self.validation_examples
                    }
                }, best_checkpoint_path)
                logging.info(f"New best model saved: {best_checkpoint_path}")
        
        # Save final training history
        self.save_training_history()
        
        logging.info("\nTraining completed!")
        logging.info(f"Best validation loss: {best_loss:.4f}")
        logging.info(f"Checkpoints saved in: {self.save_dir}")
        logging.info(f"Training history saved as: {os.path.join(self.save_dir, 'training_history.json')}")
        logging.info(f"Router outputs saved as: {os.path.join(self.save_dir, 'validation_router_outputs.jsonl')}")


def main():
    parser = argparse.ArgumentParser(description="Train router for LayerwiseQuantizeForCausalLM")
    parser.add_argument("model_path", type=str, help="Path to the quantized model")
    parser.add_argument("--dataset", type=str, default="c4", help="Dataset to use for training")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--weight_ce", type=float, default=0.5, help="Weight for cross entropy loss")
    parser.add_argument("--weight_precision", type=float, default=0.5, help="Weight for precision loss")
    parser.add_argument("--save_dir", type=str, default="router_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--precisions", type=int, nargs="+", default=None, help="Precisions to use")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--validation_examples", type=int, default=None, help="Number of validation examples (overrides validation_split)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate weights
    if abs(args.weight_ce + args.weight_precision - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {args.weight_ce + args.weight_precision}")
    
    # Create trainer
    trainer = RouterTrainer(
        model_path=args.model_path,
        dataset=args.dataset,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_ce=args.weight_ce,
        weight_precision=args.weight_precision,
        save_dir=args.save_dir,
        device=args.device,
        random_state=args.random_state,
        trust_remote_code=args.trust_remote_code,
        precisions=args.precisions,
        validation_split=args.validation_split,
        validation_examples=args.validation_examples
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
