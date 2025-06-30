# SiT/loco_framework_2stage/extract_embeddings.py

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.MVTecLOCO import MVTecLOCO
from vision_transformer import vit_small
from loco_framework import utils as loco_utils

class FeatureExtractor(nn.Module):
    """A wrapper to extract the CLS token from the ViT backbone."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        all_tokens = self.backbone(x)
        return all_tokens[:, 0]

def main(config, external_checkpoint_path):
    """Main function for Stage 2: Extracting CLS embeddings."""
    output_dir = Path(config['logging']['output_dir'])
    loco_utils.setup_logging(output_dir / '2_extract.log')
    logging.info("--- Stage 2: Starting Embedding Extraction ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load Model ---
    if external_checkpoint_path:
        checkpoint_path = Path(external_checkpoint_path)
        logging.info(f"Using provided external checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = output_dir / config['finetune']['output_checkpoint_name']
        logging.info(f"Using checkpoint from Stage 1: {checkpoint_path}")

    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found at {checkpoint_path}. Please provide a valid path or run Stage 1 first.")
        sys.exit(1)

    backbone = vit_small(patch_size=config['model']['patch_size'], img_size=[config['dataset']['target_size']])
    
    # --- CORRECTED: Use weights_only=False to load checkpoints with optimizer state ---
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    full_model_state = checkpoint['model_state_dict']
    
    backbone_state_dict = {k.replace('backbone.', ''): v for k, v in full_model_state.items() if k.startswith('backbone.')}
    backbone.load_state_dict(backbone_state_dict)
    
    feature_extractor = FeatureExtractor(backbone).to(device)
    feature_extractor.eval()
    logging.info(f"Successfully loaded fine-tuned backbone weights.")

    # --- Prepare Data ---
    transform = transforms.Compose([
        loco_utils.ResizeAndPad(target_size=config['dataset']['target_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    target_transform = transforms.ToTensor()

    all_embeddings = {'train': [], 'validation': [], 'test': []}
    
    for split in ['train', 'validation', 'test']:
        logging.info(f"Extracting embeddings for '{split}' split...")
        dataset = MVTecLOCO(
            root=config['dataset']['path'],
            category=config['dataset']['category'],
            split=split,
            transform=transform,
            target_transform=target_transform
        )
        
        data_loader = DataLoader(dataset, batch_size=config['evaluate']['batch_size'], shuffle=False, num_workers=config['train']['num_workers'])
        
        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(data_loader, desc=f"Extracting {split}")):
                if split == 'train':
                    images = data_batch
                    labels = [0] * images.size(0)
                else:
                    images, labels, _ = data_batch

                images = images.to(device)
                embeddings = feature_extractor(images)
                
                for j in range(embeddings.size(0)):
                    idx_in_split = i * config['evaluate']['batch_size'] + j
                    all_embeddings[split].append({
                        'embedding': embeddings[j].cpu(),
                        'label': labels[j] if isinstance(labels, list) else labels[j].item(),
                        'image_path': str(dataset.image_paths[idx_in_split])
                    })

    # --- Save Embeddings ---
    embedding_dir = output_dir / 'embeddings'
    embedding_dir.mkdir(exist_ok=True)
    save_path = embedding_dir / config['evaluate']['embedding_filename']
    torch.save(all_embeddings, save_path)
    logging.info(f"--- Stage 2 Finished ---")
    logging.info(f"All embeddings saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Stage 2: CLS Embedding Extraction')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration module.')
    parser.add_argument('--external_checkpoint', type=str, default=None, help='Path to an external, pre-trained model checkpoint to use for extraction.')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    config = config_module.get_config()
    main(config, args.external_checkpoint)