# SiT/configs/base_config.py

"""
Base configuration for MVTec LOCO AD experiments.
This file defines a comprehensive set of parameters and their default values.
Specific experiment configs will import and override these settings.
"""

def get_base_config():
    """Returns the base configuration dictionary."""
    
    config = {}

    # -----------------------------------------------------------------------------
    # Model Configuration
    # -----------------------------------------------------------------------------
    config['model'] = {
        'name': 'vit_small',  # choices: 'vit_tiny', 'vit_small', 'vit_base'
        'patch_size': 16,
        'pretrained_weights': './models/SiT_Small_ImageNet.pth',
        'drop_path_rate': 0.1,
    }

    # -----------------------------------------------------------------------------
    # Dataset Configuration
    # -----------------------------------------------------------------------------
    config['dataset'] = {
        'name': 'MVTecLOCO',
        'path': '../data/mvtec_loco_anomaly_detection/',
        'category': 'breakfast_box',
        'image_size': 224, # Legacy, will be superseded by target_size
        'target_size': 224, # New parameter for high-resolution processing
        # Transformation/Augmentation parameters for the reconstruction task
        'mask_ratio': 0.6,    # Percentage of image to mask for reconstruction
        'mask_align_patches': True, # Align masking with patch boundaries
    }

    # -----------------------------------------------------------------------------
    # Training Configuration
    # -----------------------------------------------------------------------------
    config['train'] = {
        'task': 'reconstruction', # Future options: 'contrastive', 'combined'
        'epochs': 100,
        'batch_size': 32,
        'num_workers': 8,
        'optimizer': 'adamw', # 'adamw', 'sgd'
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'lr_scheduler': 'cosine', # 'cosine', 'step'
        'warmup_epochs': 10,
        'min_lr': 1e-6,
        'use_fp16': True, # Enable mixed-precision training
        'clip_grad': 3.0, # Gradient clipping value
    }

    # -----------------------------------------------------------------------------
    # Evaluation Configuration
    # -----------------------------------------------------------------------------
    config['eval'] = {
        'batch_size': 32,
    }

    # -----------------------------------------------------------------------------
    # Logging and Checkpointing
    # -----------------------------------------------------------------------------
    config['logging'] = {
        'output_dir': './outputs/base_experiment',
        'log_freq': 50,         # Log training status every N batches
        'save_ckpt_freq': 50,   # Save a checkpoint every N epochs
    }

    # -----------------------------------------------------------------------------
    # System Configuration
    # -----------------------------------------------------------------------------
    config['system'] = {
        'seed': 42,
        'dist_url': 'env://', # For distributed training
    }

    return config