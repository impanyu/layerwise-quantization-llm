#!/usr/bin/env python3
"""
Example script for training the router in LayerwiseQuantizeForCausalLM.

This script demonstrates how to use the RouterTrainer class to train
the router network for adaptive precision selection.

Usage:
    python example_train_router.py --model_path /path/to/quantized/model
"""

import argparse
import logging
from train_layerwise_router import RouterTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description="Example: Train router for LayerwiseQuantizeForCausalLM")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the quantized model (e.g., cache/packed/anyprec-Llama-2-7b-chat-hf-w8_orig3-gc1-c4_s100_blk512)")
    parser.add_argument("--dataset", type=str, default="c4", 
                       help="Dataset to use for training (c4, wikitext2, ptb, pileval)")
    parser.add_argument("--seq_len", type=int, default=256, 
                       help="Sequence length (shorter for faster training)")
    parser.add_argument("--num_examples", type=int, default=500, 
                       help="Number of examples to use (smaller for faster training)")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Batch size (smaller for memory constraints)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, 
                       help="Number of training epochs")
    parser.add_argument("--weight_ce", type=float, default=0.7, 
                       help="Weight for cross entropy loss")
    parser.add_argument("--weight_precision", type=float, default=0.3, 
                       help="Weight for precision loss")
    parser.add_argument("--save_dir", type=str, default="example_router_checkpoints", 
                       help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--random_state", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--trust_remote_code", action="store_true", 
                       help="Trust remote code")
    parser.add_argument("--precisions", type=int, nargs="+", default=None, 
                       help="Precisions to use (e.g., 3 4 5 6 7 8)")
    
    args = parser.parse_args()
    
    # Validate weights
    if abs(args.weight_ce + args.weight_precision - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {args.weight_ce + args.weight_precision}")
    
    logging.info("Starting example router training...")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Sequence length: {args.seq_len}")
    logging.info(f"Number of examples: {args.num_examples}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Number of epochs: {args.num_epochs}")
    logging.info(f"CE weight: {args.weight_ce}")
    logging.info(f"Precision weight: {args.weight_precision}")
    if args.precisions:
        logging.info(f"Precisions: {args.precisions}")
    
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
    
    logging.info("Example training completed!")
    logging.info(f"Checkpoints saved in: {args.save_dir}")
    logging.info("You can now load the trained router using:")
    logging.info(f"  checkpoint = torch.load('{args.save_dir}/best_router_checkpoint.pt')")
    logging.info(f"  model.load_state_dict(checkpoint['model_state_dict'])")


if __name__ == "__main__":
    main()
