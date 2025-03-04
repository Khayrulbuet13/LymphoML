import argparse
import os
import torch
from logger import logging
from config import load_config_from_json
from trainer import get_trainer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model using the specified configuration.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    parser.add_argument('--gpu', type=str, help='GPU ID to use (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_json(args.config)
    
    # Override GPU ID if specified
    if args.gpu is not None:
        config.gpu_id = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get the appropriate trainer based on the configuration
    trainer = get_trainer(config)
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
