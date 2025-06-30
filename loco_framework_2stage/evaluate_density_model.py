# SiT/loco_framework_2stage/evaluate_density_model.py

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from vision_transformer import vit_small
from loco_framework import utils as loco_utils
from loco_framework.train import ReconstructionModel
from vision_transformer import RECHead
# Import the Autoencoder class to be able to load the model
from loco_framework_2stage.train_density_model import EmbeddingAutoencoder

def score_mahalanobis(embeddings, model_params, device):
    """Calculates Mahalanobis distance scores for a batch of embeddings."""
    mean = model_params['mean'].to(device)
    inv_cov = model_params['inv_cov'].to(device)
    
    scores = []
    for embed in embeddings:
        embed = embed.to(device, dtype=torch.float64)
        delta = embed - mean
        score_squared = torch.dot(delta, torch.matmul(inv_cov, delta))
        score = torch.sqrt(torch.clamp(score_squared, min=0.0)).item()
        scores.append(score)
    return scores

def score_autoencoder(embeddings, model, device):
    """Calculates reconstruction error scores for a batch of embeddings."""
    model.eval()
    model.to(device)
    
    criterion = nn.MSELoss(reduction='none')
    
    scores = []
    with torch.no_grad():
        for embed in embeddings:
            embed = embed.to(device)
            reconstructed_embed = model(embed.unsqueeze(0))
            loss_per_feature = criterion(reconstructed_embed.squeeze(0), embed)
            score = torch.mean(loss_per_feature).item()
            scores.append(score)
    return scores

def main(config, external_checkpoint_path):
    """Main function for Stage 4: Evaluating with a pre-trained density model."""
    output_dir = Path(config['logging']['output_dir'])
    loco_utils.setup_logging(output_dir / '4_evaluate.log')
    logging.info("--- Stage 4: Starting Density Model Evaluation ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_config = config['evaluate']
    model_type = eval_config.get('density_model', 'mahalanobis')
    
    embedding_path = output_dir / 'embeddings' / eval_config['embedding_filename']
    if not embedding_path.exists():
        logging.error(f"Embeddings not found. Please run Stage 2.")
        sys.exit(1)
    all_embeddings = torch.load(embedding_path)
    logging.info(f"Loaded embeddings for evaluation.")

    model_path = output_dir / 'density_model' / f"{model_type}_model.pt"
    if not model_path.exists():
        logging.error(f"Density model not found at {model_path}. Please run Stage 3 with `density_model: {model_type}`.")
        sys.exit(1)

    if model_type == 'mahalanobis':
        density_model_params = torch.load(model_path)
        logging.info(f"Loaded pre-trained Mahalanobis model.")
    elif model_type == 'autoencoder':
        ae_config = config['evaluate']['autoencoder']
        density_model = EmbeddingAutoencoder(
            embedding_dim=ae_config['embedding_dim'],
            hidden_dim=ae_config['hidden_dim'],
            bottleneck_dim=ae_config['bottleneck_dim']
        )
        density_model.load_state_dict(torch.load(model_path))
        logging.info(f"Loaded pre-trained Autoencoder model.")
    else:
        raise ValueError(f"Unknown density model type '{model_type}'")

    logging.info("Determining anomaly threshold using the validation set...")
    validation_embeds = [item['embedding'] for item in all_embeddings['validation']]
    
    if model_type == 'mahalanobis':
        validation_scores = score_mahalanobis(validation_embeds, density_model_params, device)
    else:
        validation_scores = score_autoencoder(validation_embeds, density_model, device)
    
    threshold = np.percentile(validation_scores, 95)
    logging.info(f"Anomaly threshold set to {threshold:.6f} (95th percentile of validation scores).")

    test_data = [item for item in all_embeddings['test'] if 'logical_anomalies' in item['image_path'] or 'good' in item['image_path']]
    test_embeds = [item['embedding'] for item in test_data]
    test_labels = [item['label'] for item in test_data]
    logging.info(f"Evaluating on {len(test_data)} test images (good and logical anomalies only).")
    
    if model_type == 'mahalanobis':
        test_scores = score_mahalanobis(test_embeds, density_model_params, device)
    else:
        test_scores = score_autoencoder(test_embeds, density_model, device)

    image_auroc = roc_auc_score(test_labels, test_scores)
    logging.info(f"--- Evaluation Finished ---")
    logging.info(f"Image-level AUROC on Logical Anomalies ({model_type}): {image_auroc:.4f}")

    logging.info("Generating example heatmaps...")
    if external_checkpoint_path:
        checkpoint_path = Path(external_checkpoint_path)
    else:
        checkpoint_path = output_dir / config['finetune']['output_checkpoint_name']

    if not checkpoint_path.exists():
        logging.error(f"Cannot generate heatmaps. Encoder checkpoint not found at: {checkpoint_path}")
    else:
        # --- CORRECTED: Instantiate the model with the correct target size ---
        backbone = vit_small(
            patch_size=config['model']['patch_size'],
            img_size=[config['dataset']['target_size']]
        )
        rec_head = RECHead(in_dim=backbone.embed_dim, patch_size=config['model']['patch_size'])
        recon_model = ReconstructionModel(backbone, rec_head).to(device)
        # We need to use weights_only=False here as well, since the checkpoint contains optimizer state
        recon_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model_state_dict'])
        recon_model.eval()

        anomalous_samples = [item for item in test_data if item['label'] == 1][:8]
        if anomalous_samples:
            transform = transforms.Compose([
                loco_utils.ResizeAndPad(target_size=config['dataset']['target_size']),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            original_images = torch.stack([transform(Image.open(item['image_path']).convert("RGB")) for item in anomalous_samples]).to(device)
            
            with torch.no_grad():
                reconstructed_images = recon_model(original_images)
                anomaly_maps = F.l1_loss(reconstructed_images, original_images, reduction='none').mean(dim=1, keepdim=True)

            vis_path = output_dir / 'visualizations' / f'heatmaps_{model_type}.jpg'
            loco_utils.visualize_evaluation(original_images, reconstructed_images, anomaly_maps, vis_path)
            logging.info(f"Saved heatmap visualization to {vis_path}")
    
    metrics_path = output_dir / f'final_metrics_{model_type}.json'
    loco_utils.save_config({'logical_anomaly_auroc': image_auroc, 'threshold': threshold}, metrics_path)
    logging.info(f"Final metrics saved to {metrics_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Stage 4: Density Model Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration module.')
    parser.add_argument('--external_checkpoint', type=str, default=None, help='Path to the encoder checkpoint file, required for generating heatmaps.')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    config = config_module.get_config()
    main(config, args.external_checkpoint)