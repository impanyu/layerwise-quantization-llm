#!/usr/bin/env python3
"""
Evaluate and compare router model performance against fixed precision quantized models.

This script:
1. Loads the best router checkpoint from training
2. Evaluates the router model in inference mode
3. Evaluates the same model with fixed precisions (3-8 bits)
4. Compares total loss, CE loss, precision loss, and average precision
5. Generates comparison plots

Usage:
    python evaluate_router_vs_quantized.py <checkpoint_dir> [options]
"""

import argparse
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import json
try:
    import matplotlib
    # Set matplotlib backend before importing pyplot to avoid GUI issues
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
from transformers import AutoTokenizer

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


class RouterEvaluator:
    def __init__(
        self,
        checkpoint_dir,
        dataset='c4',
        seq_len=512,
        num_examples=500,
        batch_size=4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        random_state=42,
        output_dir=None
    ):
        self.checkpoint_dir = checkpoint_dir
        self.dataset = dataset
        self.seq_len = seq_len
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.output_dir = output_dir or checkpoint_dir
        
        # Results storage
        self.results = {}
        
        # Load checkpoint and setup
        self.load_best_checkpoint()
        self.setup_data()
        self.setup_loss_function()
    
    def load_best_checkpoint(self):
        """Load the best checkpoint and extract configuration."""
        best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_router_checkpoint.pt')
        
        if not os.path.exists(best_checkpoint_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint_path}")
        
        logging.info(f"Loading best checkpoint: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        
        # Extract configuration
        self.config = checkpoint['config']
        self.model_path = self.config['model_path']
        self.precisions = self.config['precisions']
        self.weight_ce = self.config['weight_ce']
        self.weight_precision = self.config['weight_precision']
        
        # Load model
        self.model = LayerwiseQuantizeForCausalLM.from_quantized(
            quant_model_path=self.model_path,
            trust_remote_code=self.config.get('trust_remote_code', True),
            precisions=self.precisions
        )
        
        # Load trained router weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=self.config.get('trust_remote_code', True)
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info(f"Model loaded successfully")
        logging.info(f"Available precisions: {self.precisions}")
        logging.info(f"Best checkpoint from epoch: {checkpoint['epoch'] + 1}")
        logging.info(f"Best validation loss: {checkpoint['metrics']['val_total_loss']:.4f}")
    
    def setup_data(self):
        """Setup evaluation data."""
        logging.info(f"Loading evaluation dataset: {self.dataset}")
        
        # Use validation split for evaluation
        tokens = get_tokens(
            self.dataset, 
            'validation', 
            self.tokenizer, 
            self.seq_len, 
            self.num_examples, 
            seed=self.random_state
        )
        
        # Process tokens
        input_ids, attention_masks = self._process_tokens(tokens)
        
        # Create dataloader
        dataset = TensorDataset(input_ids, attention_masks)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=False
        )
        
        logging.info(f"Evaluation dataset: {len(input_ids)} samples")
    
    def _process_tokens(self, tokens_list):
        """Process a list of token sequences into padded tensors."""
        input_ids = []
        attention_masks = []
        
        for tokens in tokens_list:
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
        
        return torch.stack(padded_input_ids), torch.stack(padded_attention_masks)
    
    def setup_loss_function(self):
        """Setup the same loss function used during training."""
        self.ce_loss_fn = nn.CrossEntropyLoss()
    
    def calculate_average_precision(self, router_outputs):
        """Calculate the average precision used across all layers."""
        batch_size = router_outputs[0].shape[0]
        num_layers = len(router_outputs)
        
        avg_precisions = []
        
        for batch_idx in range(batch_size):
            layer_precisions = []
            for layer_idx in range(num_layers):
                precision_weights = router_outputs[layer_idx][batch_idx]
                weighted_precision = torch.sum(precision_weights * torch.tensor(self.precisions, device=self.device, dtype=torch.float32))
                layer_precisions.append(weighted_precision)
            
            avg_precision = torch.mean(torch.stack(layer_precisions))
            avg_precisions.append(avg_precision)
        
        return torch.stack(avg_precisions)
    
    def normalize_precision_loss(self, avg_precision):
        """Normalize precision loss to be comparable to cross entropy."""
        min_precision = min(self.precisions)
        max_precision = max(self.precisions)
        
        # Add small epsilon to prevent division by zero
        range_precision = max_precision - min_precision
        epsilon = 1e-8
        
        # Normalize to [0, 1] then scale
        normalized = (avg_precision - min_precision) / (range_precision + epsilon)
        # Use the same scaling as training (reduced to 10.0)
        scaled = normalized * 10.0
        
        # Clamp to prevent extreme values
        scaled = torch.clamp(scaled, min=0.0, max=20.0)
        
        return scaled
    
    def custom_loss(self, outputs, labels, router_outputs):
        """Custom loss function matching the training setup."""
        # Cross entropy loss
        ce_loss = self.ce_loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        
        # Calculate average precision
        avg_precision = self.calculate_average_precision(router_outputs)
        
        # Normalize precision loss
        normalized_precision = self.normalize_precision_loss(avg_precision)
        precision_loss = torch.mean(normalized_precision)
        
        # Weighted combination
        total_loss = self.weight_ce * ce_loss + self.weight_precision * precision_loss
        
        return total_loss, ce_loss, precision_loss, torch.mean(avg_precision)
    
    def evaluate_router_model(self):
        """Evaluate the trained router model in inference mode."""
        logging.info("Evaluating router model (inference mode)...")
        
        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_precision_loss = 0.0
        total_avg_precision = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(self.dataloader, desc="Router evaluation"):
                input_ids = input_ids.to(self.device)
                
                # Use inference mode with router outputs
                outputs = self.model.infer_forward(
                    input_ids=input_ids,
                    return_router_outputs=True
                )
                
                router_outputs = outputs['router_outputs']
                
                # Calculate loss (router outputs are one-hot vectors from inference)
                loss, ce_loss, precision_loss, avg_precision = self.custom_loss(
                    outputs, input_ids, router_outputs
                )
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_precision_loss += precision_loss.item()
                total_avg_precision += avg_precision.item()
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_precision_loss = total_precision_loss / num_batches
        avg_avg_precision = total_avg_precision / num_batches
        
        self.results['Router (Best)'] = {
            'total_loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'precision_loss': avg_precision_loss,
            'avg_precision': avg_avg_precision,
            'model_type': 'router'
        }
        
        logging.info(f"Router results - Total: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, "
                    f"Prec: {avg_precision_loss:.4f}, AvgPrec: {avg_avg_precision:.2f}")
    
    def evaluate_fixed_precision_model(self, precision):
        """Evaluate the model with a fixed precision."""
        logging.info(f"Evaluating fixed precision model ({precision}-bit)...")
        
        self.model.eval()
        
        # Set all layers to the specified precision
        self.model.set_precision(precision)
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_precision_loss = 0.0
        total_avg_precision = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(self.dataloader, desc=f"{precision}-bit evaluation"):
                input_ids = input_ids.to(self.device)
                
                # Use training forward to get router outputs (but they'll be uniform for fixed precision)
                outputs = self.model.train_forward(
                    input_ids=input_ids,
                    return_router_outputs=True
                )
                
                router_outputs = outputs.router_outputs
                
                # Calculate loss
                loss, ce_loss, precision_loss, avg_precision = self.custom_loss(
                    outputs, input_ids, router_outputs
                )
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_precision_loss += precision_loss.item()
                total_avg_precision += avg_precision.item()
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_precision_loss = total_precision_loss / num_batches
        avg_avg_precision = total_avg_precision / num_batches
        
        self.results[f'{precision}-bit Fixed'] = {
            'total_loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'precision_loss': avg_precision_loss,
            'avg_precision': avg_avg_precision,
            'model_type': 'fixed',
            'precision': precision
        }
        
        logging.info(f"{precision}-bit results - Total: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, "
                    f"Prec: {avg_precision_loss:.4f}, AvgPrec: {avg_avg_precision:.2f}")
    
    def evaluate_all_models(self):
        """Evaluate router model and all fixed precision models."""
        # Evaluate router model
        self.evaluate_router_model()
        
        # Evaluate fixed precision models
        for precision in self.precisions:
            self.evaluate_fixed_precision_model(precision)
    
    def save_results(self):
        """Save evaluation results to JSON."""
        results_path = os.path.join(self.output_dir, 'router_vs_quantized_results.json')
        
        # Add metadata
        results_with_metadata = {
            'metadata': {
                'checkpoint_dir': self.checkpoint_dir,
                'model_path': self.model_path,
                'dataset': self.dataset,
                'seq_len': self.seq_len,
                'num_examples': self.num_examples,
                'batch_size': self.batch_size,
                'precisions': self.precisions,
                'weight_ce': self.weight_ce,
                'weight_precision': self.weight_precision
            },
            'results': self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        logging.info(f"Results saved to: {results_path}")
    
    def plot_comparison(self):
        """Create comparison plots."""
        if not PLOTTING_AVAILABLE:
            logging.warning("Plotting libraries not available. Install matplotlib and seaborn to generate plots.")
            return
            
        if not self.results:
            logging.warning("No results to plot")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        metrics = ['total_loss', 'ce_loss', 'precision_loss', 'avg_precision']
        metric_labels = ['Total Loss', 'Cross Entropy Loss', 'Precision Loss', 'Average Precision (bits)']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Router vs Fixed Precision Models Comparison', fontsize=16, fontweight='bold')
        
        # Colors: router in red, fixed precisions in blue gradient
        colors = ['#E74C3C'] + [plt.cm.Blues(i/len(self.precisions)) for i in range(len(self.precisions))]
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            values = [self.results[name][metric] for name in model_names]
            bars = ax.bar(model_names, values, color=colors[:len(model_names)])
            
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}' if metric != 'avg_precision' else f'{value:.2f}',
                       ha='center', va='bottom', fontsize=9)
            
            # Highlight best performance
            if metric in ['total_loss', 'ce_loss', 'precision_loss']:
                best_idx = np.argmin(values)
            else:  # avg_precision - depends on what's better (lower or higher precision)
                best_idx = 0  # Router model for now
            
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(self.output_dir, 'router_vs_quantized_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Comparison plot saved: {plot_path}")
        
        plot_pdf_path = os.path.join(self.output_dir, 'router_vs_quantized_comparison.pdf')
        plt.savefig(plot_pdf_path, bbox_inches='tight')
        logging.info(f"Comparison plot (PDF) saved: {plot_pdf_path}")
        
        plt.close()
        
        # Create a focused loss comparison
        self._plot_loss_focus()
    
    def _plot_loss_focus(self):
        """Create a focused plot comparing just the losses."""
        if not PLOTTING_AVAILABLE:
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        model_names = list(self.results.keys())
        total_losses = [self.results[name]['total_loss'] for name in model_names]
        ce_losses = [self.results[name]['ce_loss'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, total_losses, width, label='Total Loss', alpha=0.8, color='#3498DB')
        bars2 = ax.bar(x + width/2, ce_losses, width, label='CE Loss', alpha=0.8, color='#E74C3C')
        
        ax.set_xlabel('Model Configuration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Total Loss vs Cross Entropy Loss Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save focused plot
        focused_plot_path = os.path.join(self.output_dir, 'loss_focused_comparison.png')
        plt.savefig(focused_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Focused loss comparison saved: {focused_plot_path}")
        
        plt.close()
    
    def print_summary(self):
        """Print a summary of results."""
        print("\n" + "="*80)
        print("ROUTER VS QUANTIZED MODELS EVALUATION SUMMARY")
        print("="*80)
        
        # Find best performing models for each metric
        metrics = ['total_loss', 'ce_loss', 'precision_loss', 'avg_precision']
        
        for metric in metrics:
            values = [(name, self.results[name][metric]) for name in self.results.keys()]
            
            if metric == 'avg_precision':
                # For average precision, we might want the router's adaptive choice
                best_name, best_value = 'Router (Best)', self.results['Router (Best)'][metric]
            else:
                # For losses, lower is better
                best_name, best_value = min(values, key=lambda x: x[1])
            
            print(f"\nBest {metric.replace('_', ' ').title()}: {best_name} ({best_value:.4f})")
        
        print(f"\nDetailed Results:")
        print("-" * 80)
        for name, result in self.results.items():
            print(f"{name:15s} | Total: {result['total_loss']:.4f} | "
                  f"CE: {result['ce_loss']:.4f} | Prec: {result['precision_loss']:.4f} | "
                  f"AvgPrec: {result['avg_precision']:.2f}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate router vs fixed precision quantized models")
    parser.add_argument("checkpoint_dir", type=str, nargs='?', default="router_checkpoints", help="Directory containing router checkpoints (default: router_checkpoints)")
    parser.add_argument("--dataset", type=str, default="c4", help="Dataset to use for evaluation")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num_examples", type=int, default=500, help="Number of examples for evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to checkpoint_dir)")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create evaluator
    evaluator = RouterEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        dataset=args.dataset,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        device=args.device,
        random_state=args.random_state,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    logging.info("Starting evaluation...")
    evaluator.evaluate_all_models()
    
    # Save results
    evaluator.save_results()
    
    # Generate plots
    if not args.no_plots:
        if not PLOTTING_AVAILABLE:
            logging.warning("Plotting libraries not available. Install matplotlib and seaborn to generate plots.")
        else:
            try:
                logging.info("Generating comparison plots...")
                evaluator.plot_comparison()
            except Exception as e:
                logging.warning(f"Failed to generate plots: {e}")
    
    # Print summary
    evaluator.print_summary()
    
    logging.info("Evaluation completed!")


if __name__ == "__main__":
    main()
