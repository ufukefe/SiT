# SiT/loco_framework_2stage/3_train_density_model.py

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from loco_framework import utils as loco_utils

# --- Model Definition for the Autoencoder ---
class EmbeddingAutoencoder(nn.Module):
    """A simple fully-connected autoencoder for embedding reconstruction."""
    def __init__(self, embedding_dim, hidden_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_mahalanobis(config, all_embeddings):
    """Fits and saves a Mahalanobis distance model."""
    logging.info("Fitting Mahalanobis distance model...")
    train_embeds = torch.stack([item['embedding'] for item in all_embeddings['train']])
    
    mean = torch.mean(train_embeds.to(torch.float64), dim=0)
    cov = torch.cov(train_embeds.t().to(torch.float64))
    epsilon = 1e-6
    identity_matrix = torch.eye(cov.size(0), dtype=torch.float64)
    inv_cov = torch.linalg.inv(cov + epsilon * identity_matrix)
    
    model_dir = Path(config['logging']['output_dir']) / 'density_model'
    model_dir.mkdir(exist_ok=True)
    model_save_path = model_dir / "mahalanobis_model.pt"
    
    torch.save({'mean': mean, 'inv_cov': inv_cov}, model_save_path)
    logging.info(f"Mahalanobis model saved to {model_save_path}")

def train_autoencoder(config, all_embeddings):
    """Trains and saves an embedding autoencoder."""
    logging.info("Training Embedding Autoencoder...")
    ae_config = config['evaluate']['autoencoder']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Prepare Data ---
    train_embeds = torch.stack([item['embedding'] for item in all_embeddings['train']])
    train_dataset = TensorDataset(train_embeds)
    train_loader = DataLoader(train_dataset, batch_size=ae_config['batch_size'], shuffle=True)
    
    # --- Initialize Model, Loss, Optimizer ---
    model = EmbeddingAutoencoder(
        embedding_dim=ae_config['embedding_dim'],
        hidden_dim=ae_config['hidden_dim'],
        bottleneck_dim=ae_config['bottleneck_dim']
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_config['learning_rate'])

    # --- Training Loop ---
    model.train()
    for epoch in range(ae_config['epochs']):
        epoch_loss = 0
        for batch in train_loader:
            embeddings = batch[0].to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, embeddings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{ae_config['epochs']}], Loss: {avg_loss:.6f}")

    # --- Save the trained model ---
    model_dir = Path(config['logging']['output_dir']) / 'density_model'
    model_dir.mkdir(exist_ok=True)
    model_save_path = model_dir / "autoencoder_model.pt"
    
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Autoencoder model saved to {model_save_path}")

def main(config):
    """Main function for Stage 3: Training the density model."""
    output_dir = Path(config['logging']['output_dir'])
    loco_utils.setup_logging(output_dir / '3_train_density.log')
    logging.info("--- Stage 3: Starting Density Model Training ---")

    eval_config = config['evaluate']
    
    # --- Load Embeddings ---
    embedding_path = output_dir / 'embeddings' / eval_config['embedding_filename']
    if not embedding_path.exists():
        logging.error(f"Embeddings not found at {embedding_path}. Please run Stage 2 first.")
        sys.exit(1)
    
    all_embeddings = torch.load(embedding_path)
    logging.info(f"Loaded {len(all_embeddings['train'])} train embeddings.")

    # --- Model Training Dispatcher ---
    model_type = eval_config.get('density_model', 'mahalanobis') # Default to mahalanobis
    logging.info(f"Selected density model: {model_type}")

    if model_type == 'mahalanobis':
        train_mahalanobis(config, all_embeddings)
    elif model_type == 'autoencoder':
        train_autoencoder(config, all_embeddings)
    else:
        raise ValueError(f"Unknown density model type '{model_type}' specified in config.")
    
    logging.info(f"--- Stage 3 Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Stage 3: Density Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration module.')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    config = config_module.get_config()
    main(config)