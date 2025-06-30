# SiT/configs/loco_breakfast_box_recon.py

"""
Configuration for fine-tuning SiT on MVTec LOCO 'breakfast_box'
for the reconstruction-based anomaly detection task.
"""

from .base_config import get_base_config

def get_config():
    """Returns the configuration for this specific experiment."""
    
    config = get_base_config()

    # --- Override model configuration ---
    config['model']['name'] = 'vit_small'
    config['model']['pretrained_weights'] = './models/SiT_Small_ImageNet.pth'
    config['model']['patch_size'] = 16

    # --- Override dataset configuration ---
    config['dataset']['name'] = 'MVTecLOCO'
    config['dataset']['path'] = '../data/mvtec_loco_anomaly_detection/'
    config['dataset']['category'] = 'breakfast_box'
    # Set the target size for high-resolution processing
    config['dataset']['target_size'] = 512 
    config['dataset']['mask_ratio'] = 0.6
    
    # --- Override training configuration ---
    config['train']['task'] = 'reconstruction'
    config['train']['epochs'] = 200
    # IMPORTANT: Reduce batch size for larger images to fit in memory
    config['train']['batch_size'] = 4 
    config['train']['learning_rate'] = 1e-5
    config['train']['weight_decay'] = 0.05
    config['train']['warmup_epochs'] = 5

    # --- Override evaluation configuration ---
    # Also reduce eval batch size
    config['eval']['batch_size'] = 4

    # --- Override logging and checkpointing configuration ---
    config['logging']['output_dir'] = './outputs/loco_breakfast_box_recon_512' # New output dir
    config['logging']['save_ckpt_freq'] = 50

    return config