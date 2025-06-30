# SiT/loco_framework_2stage/finetune_encoder.py

import argparse
import importlib
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the main training function from the one-stage framework
from loco_framework.train import main as one_stage_trainer

def main():
    """
    A wrapper script for Stage 1: Fine-tuning the encoder.
    This script adapts the configuration from the two-stage setup and
    calls the robust, one-stage training script to perform the actual training.
    """
    parser = argparse.ArgumentParser('Wrapper for Stage 1: SiT Encoder Fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to the two-stage configuration module.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint file to resume training from.')
    args = parser.parse_args()

    # --- 1. Load the Two-Stage Configuration ---
    try:
        config_module = importlib.import_module(args.config)
        config = config_module.get_config()
    except ImportError as e:
        print(f"Error: Could not import config module '{args.config}'.")
        raise e

    # --- 2. Adapt the Configuration for the One-Stage Trainer ---
    # The one-stage trainer expects parameters in the `train` and `logging` dicts.
    # We will override the base config with the specific parameters from our `finetune` block.
    
    ft_config = config['finetune']
    
    config['train']['epochs'] = ft_config['epochs']
    config['train']['batch_size'] = ft_config['batch_size']
    config['train']['learning_rate'] = ft_config['learning_rate']
    config['dataset']['mask_ratio'] = ft_config['mask_ratio']
    
    # Set the final output name within the one-stage trainer's expected structure
    config['train']['final_checkpoint_name'] = ft_config['output_checkpoint_name']
    
    # --- 3. Set up Logging ---
    # Set up logging specifically for this stage
    output_dir = Path(config['logging']['output_dir'])
    # The one-stage trainer will create the dir, but we set up the logger here
    # Note: The one-stage trainer will also set up its own logging.
    # This ensures we have a dedicated log for the wrapper's actions if needed.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Wrapper: Preparing to call one-stage trainer for fine-tuning. ---")

    # --- 4. Call the Main Training Function ---
    # Pass the adapted config and the resume path to the imported trainer.
    # Note: The one-stage trainer needs to be modified to accept the resume_from path.
    # We will assume this modification for this wrapper.
    one_stage_trainer(config, args.resume_from)


if __name__ == '__main__':
    main()