# SiT/loco_framework/utils.py

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F_trans
from torchvision.transforms import InterpolationMode


class ResizeAndPad:
    """
    A callable class to resize an image to a target size while maintaining aspect ratio,
    then pad it to a square.

    Args:
        target_size (int): The target size for the longest dimension and the final square size.
        interpolation (InterpolationMode): Desired interpolation mode for resizing.
        fill (int): Pixel value for padding.
    """
    def __init__(self, target_size, interpolation=InterpolationMode.BICUBIC, fill=0):
        self.target_size = target_size
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img):
        # Get original image size
        w, h = img.size
        
        # Determine the new size, maintaing aspect ratio
        if w > h:
            new_w = self.target_size
            new_h = int(h * (self.target_size / w))
        else:
            new_h = self.target_size
            new_w = int(w * (self.target_size / h))
            
        # Resize the image
        img = F_trans.resize(img, (new_h, new_w), self.interpolation)
        
        # Pad the image to make it a square
        padding_left = (self.target_size - new_w) // 2
        padding_right = self.target_size - new_w - padding_left
        padding_top = (self.target_size - new_h) // 2
        padding_bottom = self.target_size - new_h - padding_top
        
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        
        return F_trans.pad(img, padding, self.fill, 'constant')


def setup_logging(log_path: Path):
    """
    Sets up the logging to write to a file and to the console.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def save_config(config: dict, path: Path):
    """Saves the configuration dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_reconstruction(original: torch.Tensor, masked: torch.Tensor, reconstructed: torch.Tensor, path: Path, n_samples: int = 8):
    """Saves a grid of images to visualize the training reconstruction quality."""
    n_samples = min(n_samples, original.size(0))
    original_samples = original[:n_samples].cpu()
    masked_samples = masked[:n_samples].cpu()
    reconstructed_samples = reconstructed[:n_samples].cpu()
    images_to_show = torch.cat([original_samples, masked_samples, reconstructed_samples])
    grid = torchvision.utils.make_grid(images_to_show, nrow=n_samples, normalize=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, path)


def visualize_evaluation(original: torch.Tensor, reconstructed: torch.Tensor, anomaly_map: torch.Tensor, path: Path, n_samples: int = 8):
    """Saves a grid of images to visualize evaluation results."""
    n_samples = min(n_samples, original.size(0))
    
    original_samples = original[:n_samples].cpu()
    reconstructed_samples = reconstructed[:n_samples].cpu()
    anomaly_map_samples = anomaly_map[:n_samples].cpu()

    normalized_anomaly_maps = []
    for am in anomaly_map_samples:
        am = am - am.min()
        am = am / am.max()
        normalized_anomaly_maps.append(am)
    
    anomaly_map_samples = torch.stack(normalized_anomaly_maps)
    
    # --- CORRECTED LINE ---
    # Expand the single-channel anomaly map to 3 channels to match RGB images
    anomaly_map_rgb = anomaly_map_samples.expand(-1, 3, -1, -1)

    images_to_show = torch.cat([original_samples, reconstructed_samples, anomaly_map_rgb])
    grid = torchvision.utils.make_grid(images_to_show, nrow=n_samples, normalize=True)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, path)