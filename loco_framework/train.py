# SiT/loco_framework/train.py

import argparse
import importlib
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.MVTecLOCO import MVTecLOCO
from datasets.datasets_utils import GMML_drop_rand_patches
from vision_transformer import RECHead, vit_small
from loco_framework import utils as loco_utils
import utils as sit_utils

class ReconstructionTransform:
    def __init__(self, config):
        self.patch_size = config['model']['patch_size']
        self.mask_ratio = config['dataset']['mask_ratio']
        target_size = config['dataset']['target_size']
        self.mask_align_patches = config['dataset']['mask_align_patches']
        
        self.base_transform = transforms.Compose([
            loco_utils.ResizeAndPad(target_size=target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image: Image.Image):
        original_image = self.base_transform(image)
        masked_image = original_image.clone()
        align = self.patch_size if self.mask_align_patches else 1
        masked_image, _ = GMML_drop_rand_patches(
            masked_image, 
            max_replace=self.mask_ratio, 
            align=align,
            drop_type='zeros'
        )
        return original_image, masked_image

class ReconstructionModel(nn.Module):
    def __init__(self, vit_backbone, rec_head):
        super().__init__()
        self.backbone = vit_backbone
        self.rec_head = rec_head

    def forward(self, x):
        all_tokens = self.backbone(x)
        patch_tokens = all_tokens[:, 1:]
        reconstructed_image = self.rec_head(patch_tokens)
        return reconstructed_image

def train_one_epoch(model, data_loader, optimizer, lr_schedule, epoch, config):
    model.train()
    metric_logger = sit_utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}/{config["train"]["epochs"]}]'
    
    start_iter = epoch * len(data_loader)
    for i, (original_images, masked_images) in enumerate(metric_logger.log_every(data_loader, config['logging']['log_freq'], header)):
        it = start_iter + i
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[it]

        original_images = original_images.cuda(non_blocking=True)
        masked_images = masked_images.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config['train']['use_fp16']):
            reconstructed_images = model(masked_images)
            loss = F.l1_loss(reconstructed_images, original_images)

        if not math.isfinite(loss.item()):
            logging.error(f"Loss is {loss.item()}, stopping training.")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        sit_utils.clip_gradients(model, config['train']['clip_grad'])
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats for epoch {epoch}: {metric_logger}")
    
    if sit_utils.is_main_process():
        vis_path = Path(config['logging']['output_dir']) / 'visualizations' / f'epoch_{epoch:04d}.jpg'
        loco_utils.visualize_reconstruction(original_images, masked_images, reconstructed_images.detach(), vis_path)

def main(config, resume_from_path=None):
    sit_utils.init_distributed_mode(argparse.Namespace(**config['system']))
    loco_utils.set_seed(config['system']['seed'])
    
    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sit_utils.is_main_process():
        loco_utils.setup_logging(output_dir / 'train.log')
        loco_utils.save_config(config, output_dir / 'config.json')

    logging.info(f"Starting training run in: {output_dir}")
    
    transform = ReconstructionTransform(config)
    dataset = MVTecLOCO(root=config['dataset']['path'], category=config['dataset']['category'], split='train', transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'], pin_memory=True, drop_last=True)
    logging.info(f"Data loaded: {len(dataset)} images.")

    backbone = vit_small(patch_size=config['model']['patch_size'], img_size=[config['dataset']['target_size']])
    rec_head = RECHead(in_dim=backbone.embed_dim, patch_size=config['model']['patch_size'])
    model = ReconstructionModel(backbone, rec_head).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
    start_epoch = 0

    if resume_from_path:
        checkpoint_path = Path(resume_from_path)
        if checkpoint_path.exists():
            logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resumed from epoch {start_epoch}.")
        else:
            logging.error(f"Resume checkpoint not found: {resume_from_path}"); sys.exit(1)
    else:
        imagenet_checkpoint_path = Path(config['model']['pretrained_weights'])
        if imagenet_checkpoint_path.exists():
            logging.info(f"Starting from ImageNet weights: {imagenet_checkpoint_path}")
            state_dict = torch.load(imagenet_checkpoint_path, map_location='cpu', weights_only=False)
            
            student_state_dict = state_dict.get('student', state_dict)
            backbone_state_dict = {k.replace('module.backbone.', ''): v for k, v in student_state_dict.items() if k.startswith('module.backbone.')}
            
            sit_utils.interpolate_pos_embed(model.backbone, backbone_state_dict)
            
            msg = model.backbone.load_state_dict(backbone_state_dict, strict=False)
            logging.info(f"Loaded backbone weights with message: {msg}")
        else:
            logging.warning("No ImageNet checkpoint found. Starting from scratch.")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[sit_utils.get_rank()])
    
    lr_schedule = sit_utils.cosine_scheduler(
        base_value=config['train']['learning_rate'],
        final_value=config['train']['min_lr'],
        epochs=config['train']['epochs'],
        niter_per_ep=len(data_loader),
        warmup_epochs=config['train']['warmup_epochs'],
    )
    
    logging.info(f"Starting training loop from epoch {start_epoch}...")
    for epoch in range(start_epoch, config['train']['epochs']):
        data_loader.sampler.set_epoch(epoch)
        # --- CORRECTED TYPO ---
        train_one_epoch(model, data_loader, optimizer, lr_schedule, epoch, config)
        
        if sit_utils.is_main_process() and (epoch + 1) % config['logging']['save_ckpt_freq'] == 0:
            save_path = output_dir / f'checkpoint_{epoch:04d}.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
            logging.info(f"Saved intermediate checkpoint to {save_path}")
            
    if sit_utils.is_main_process():
        final_checkpoint_name = config['train'].get('final_checkpoint_name', 'checkpoint_final.pth')
        final_save_path = output_dir / final_checkpoint_name
        torch.save({'epoch': config['train']['epochs'] - 1, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, final_save_path)
        logging.info(f"Final model saved to {final_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT One-Stage Fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration module.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint file to resume training from.')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    config = config_module.get_config()
    main(config, args.resume_from)