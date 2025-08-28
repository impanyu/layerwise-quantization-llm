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

from LayerwiseQuantizeForCausalLM import LayerwiseQuantizeForCausalLM
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
        precisions=None
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
            'total_loss': [],
            'ce_loss': [],
            'precision_loss': [],
            'avg_precision': []
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
        """Setup training data using the same mechanism as quantize.py."""
        logging.info(f"Loading dataset: {self.dataset}")
        
        # Get tokens using the same function as quantize.py
        input_tokens = get_tokens(
            self.dataset, 
            'train', 
            self.tokenizer, 
            self.seq_len, 
            self.num_examples, 
            seed=self.random_state
        )
        
        # Convert to tensors and create attention masks
        input_ids = []
        attention_masks = []
        
        for tokens in input_tokens:
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
        self.input_ids = torch.stack(padded_input_ids)
        self.attention_masks = torch.stack(padded_attention_masks)
        
        # Create dataset and dataloader
        dataset = TensorDataset(self.input_ids, self.attention_masks)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        logging.info(f"Dataset loaded: {len(self.input_ids)} samples, batch size: {self.batch_size}")
    
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
        scaled = normalized * 10.0
        
        return scaled
    
    def custom_loss(self, outputs, labels, router_outputs):
        """Custom loss function combining cross entropy and precision penalty."""
        # Cross entropy loss
        ce_loss = nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        
        # Calculate average precision
        avg_precision = self.calculate_average_precision(router_outputs)
        
        # Normalize precision loss
        normalized_precision = self.normalize_precision_loss(avg_precision)
        precision_loss = torch.mean(normalized_precision)
        
        # Weighted combination
        total_loss = self.weight_ce * ce_loss + self.weight_precision * precision_loss
        
        return total_loss, ce_loss, precision_loss, torch.mean(avg_precision)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_precision_loss = 0.0
        total_avg_precision = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, (input_ids, attention_mask) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass with router outputs collection
            outputs = self.model.train_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_router_outputs=True
            )
            
            router_outputs = outputs.router_outputs
            
            # Calculate loss
            loss, ce_loss, precision_loss, avg_precision = self.custom_loss(
                outputs, input_ids, router_outputs
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
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
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_precision_loss = total_precision_loss / num_batches
        avg_precision = total_avg_precision / num_batches
        
        return avg_loss, avg_ce_loss, avg_precision_loss, avg_precision
    
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
                'random_state': self.random_state
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
        logging.info(f"  - Available precisions: {self.model.precisions}")
        
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logging.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train for one epoch
            avg_loss, avg_ce_loss, avg_precision_loss, avg_precision = self.train_epoch(epoch)
            
            # Log metrics
            logging.info(f"Epoch {epoch+1} Results:")
            logging.info(f"  - Total Loss: {avg_loss:.4f}")
            logging.info(f"  - CE Loss: {avg_ce_loss:.4f}")
            logging.info(f"  - Precision Loss: {avg_precision_loss:.4f}")
            logging.info(f"  - Average Precision: {avg_precision:.2f}")
            
            # Save to history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['total_loss'].append(avg_loss)
            self.train_history['ce_loss'].append(avg_ce_loss)
            self.train_history['precision_loss'].append(avg_precision_loss)
            self.train_history['avg_precision'].append(avg_precision)
            
            # Save checkpoint
            metrics = {
                'total_loss': avg_loss,
                'ce_loss': avg_ce_loss,
                'precision_loss': avg_precision_loss,
                'avg_precision': avg_precision
            }
            self.save_checkpoint(epoch, metrics)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
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
                        'random_state': self.random_state
                    }
                }, best_checkpoint_path)
                logging.info(f"New best model saved: {best_checkpoint_path}")
        
        # Save final training history
        self.save_training_history()
        
        logging.info("\nTraining completed!")
        logging.info(f"Best loss: {best_loss:.4f}")
        logging.info(f"Checkpoints saved in: {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train router for LayerwiseQuantizeForCausalLM")
    parser.add_argument("model_path", type=str, help="Path to the quantized model")
    parser.add_argument("--dataset", type=str, default="c4", help="Dataset to use for training")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--weight_ce", type=float, default=0.7, help="Weight for cross entropy loss")
    parser.add_argument("--weight_precision", type=float, default=0.3, help="Weight for precision loss")
    parser.add_argument("--save_dir", type=str, default="router_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--precisions", type=int, nargs="+", default=None, help="Precisions to use")
    
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
        precisions=args.precisions
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
