# SiT/configs/loco_breakfast_box_2stage.py

"""
Configuration for the TWO-STAGE framework:
1. Fine-tune a SiT encoder on 'breakfast_box'.
2. Extract CLS embeddings and use a density model for anomaly detection.
"""

from .base_config import get_base_config

def get_config():
    """Returns the configuration for this specific experiment."""
    
    config = get_base_config()

    # -----------------------------------------------------------------------------
    # General Experiment Settings
    # -----------------------------------------------------------------------------
    config['dataset']['category'] = 'breakfast_box'
    config['dataset']['target_size'] = 768  # High-resolution processing
    config['logging']['output_dir'] = './outputs/loco_breakfast_box_2stage'

    # -----------------------------------------------------------------------------
    # STAGE 1: Fine-tuning Configuration
    # -----------------------------------------------------------------------------
    config['finetune'] = {
        'epochs': 1000,
        'batch_size': 2,
        'learning_rate': 1e-5,
        'mask_ratio': 0.6,
        # --- UPDATED: More descriptive name for the final artifact ---
        'output_checkpoint_name': 'stage1_finetuned_model.pth',
    }

    # -----------------------------------------------------------------------------
    # STAGE 2: Evaluation Configuration
    # -----------------------------------------------------------------------------
    config['evaluate'] = {
        'embedding_filename': 'embeddings.pt',
        'density_model': 'autoencoder',
        'batch_size': 32, 
        
        'autoencoder': {
            'embedding_dim': 384,
            'hidden_dim': 128,
            'bottleneck_dim': 32,
            'epochs': 3000,
            'learning_rate': 1e-3,
            'batch_size': 64,
        }
    }

    return config