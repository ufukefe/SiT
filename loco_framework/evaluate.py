# SiT/loco_framework/evaluate.py

import argparse
import importlib
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
# --- CORRECTED: Import InterpolationMode directly from torchvision.transforms ---
from torchvision.transforms import InterpolationMode

from datasets.MVTecLOCO import MVTecLOCO
from loco_framework.train import ReconstructionModel
from vision_transformer import RECHead, vit_small
from loco_framework import utils as loco_utils


def evaluate(model, data_loader, device, output_dir):
    """
    Main evaluation loop.
    
    Args:
        model (nn.Module): The trained model to evaluate.
        data_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to run evaluation on.
        output_dir (Path): Directory to save visualizations.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    model.eval()
    
    all_img_labels = []
    all_img_scores = []
    
    all_pixel_gt = []
    all_pixel_scores = []
    
    visualized = False

    with torch.no_grad():
        for i, (images, labels, masks) in enumerate(data_loader):
            images = images.to(device)
            
            reconstructed_images = model(images)
            
            anomaly_maps = F.l1_loss(reconstructed_images, images, reduction='none')
            anomaly_maps = anomaly_maps.mean(dim=1, keepdim=True)
            
            if not visualized:
                vis_path = output_dir / 'visualizations' / 'evaluation_reconstruction.jpg'
                loco_utils.visualize_evaluation(images, reconstructed_images.detach(), anomaly_maps.detach(), vis_path)
                logging.info(f"Saved evaluation visualization to {vis_path}")
                visualized = True
            
            img_scores = torch.amax(anomaly_maps, dim=(1, 2, 3))
            
            all_img_labels.extend(labels.cpu().numpy())
            all_img_scores.extend(img_scores.cpu().numpy())
            
            masks_np = masks.numpy().flatten()
            anomaly_maps_np = anomaly_maps.cpu().numpy().flatten()
            
            all_pixel_gt.extend(masks_np)
            all_pixel_scores.extend(anomaly_maps_np)

    image_auroc = roc_auc_score(all_img_labels, all_img_scores)
    
    pixel_gt_binary = np.array(all_pixel_gt) > 0
    pixel_auroc = roc_auc_score(pixel_gt_binary, all_pixel_scores)

    return {
        'image_auroc': image_auroc,
        'pixel_auroc': pixel_auroc,
    }


def main(config, checkpoint_path):
    output_dir = Path(config['logging']['output_dir'])
    loco_utils.setup_logging(output_dir / 'eval.log')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Starting evaluation for checkpoint: {checkpoint_path}")
    logging.info(f"Using device: {device}")

    target_size = config['dataset']['target_size']
    eval_transform = transforms.Compose([
        loco_utils.ResizeAndPad(target_size=target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # --- CORRECTED: Use the directly imported InterpolationMode ---
    target_transform = transforms.Compose([
        loco_utils.ResizeAndPad(target_size=target_size, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    dataset = MVTecLOCO(
        root=config['dataset']['path'],
        category=config['dataset']['category'],
        split='test',
        transform=eval_transform,
        target_transform=target_transform,
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config['eval']['batch_size'],
        num_workers=config['train']['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    logging.info(f"Test data loaded: {len(dataset)} images.")

    backbone = vit_small(
        patch_size=config['model']['patch_size'],
        img_size=[config['dataset']['target_size']]
    )
    rec_head = RECHead(in_dim=backbone.embed_dim, patch_size=config['model']['patch_size'])
    model = ReconstructionModel(backbone, rec_head).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded model state from {checkpoint_path}")

    metrics = evaluate(model, data_loader, device, output_dir)

    logging.info("Evaluation finished.")
    logging.info(f"Image-level AUROC: {metrics['image_auroc']:.4f}")
    logging.info(f"Pixel-level AUROC: {metrics['pixel_auroc']:.4f}")
    
    metrics_path = output_dir / 'evaluation_metrics.json'
    loco_utils.save_config(metrics, metrics_path)
    logging.info(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT Evaluation for Anomaly Detection')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration module, e.g., configs.loco_breakfast_box_recon')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint.')
    args = parser.parse_args()

    try:
        config_module = importlib.import_module(args.config)
    except ImportError as e:
        logging.error(f"Could not import config module '{args.config}'. Make sure it's a valid Python module path.")
        raise e

    config = config_module.get_config()
    
    main(config, args.checkpoint)