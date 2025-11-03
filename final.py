"""
FastFaceSR V4-Optimized for 30+ dB PSNR
Focus on pixel accuracy first, then perceptual quality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import time
from pathlib import Path

from fast_model_v4 import FastFaceSR_V4
from paired_dataset import PairedFaceDataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips


class OptimizedLoss(nn.Module):
    """Optimized loss focused on PSNR (pixel accuracy)"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        # Heavy focus on pixel accuracy for high PSNR
        loss_l1 = self.l1_loss(pred, target)
        loss_mse = self.mse_loss(pred, target)
        
        # 70% L1 + 30% MSE = strong pixel accuracy
        total_loss = 0.7 * loss_l1 + 0.3 * loss_mse
        
        return total_loss


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Simple training without GAN"""
    model.train()
    total_loss = 0
    
    for lr, hr in dataloader:
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            sr = model(lr)
            loss = criterion(sr, hr)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device, lpips_metric):
    """Validate model - reuse LPIPS to avoid reloading"""
    model.eval()
    
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    
    lpips_total = 0
    num_samples = 0
    
    for lr, hr in dataloader:
        lr, hr = lr.to(device), hr.to(device)
        
        with autocast('cuda'):
            sr = model(lr)
            sr = torch.clamp(sr, -1, 1)
        
        psnr_metric.update(sr, hr)
        ssim_metric.update(sr, hr)
        
        lpips_val = lpips_metric(sr, hr)
        lpips_total += lpips_val.sum().item()
        num_samples += lr.size(0)
    
    return psnr_metric.compute().item(), ssim_metric.compute().item(), lpips_total / num_samples
# ------------------------------------------------------------
# ✅ Load trained model before generating samples
# ------------------------------------------------------------
# ------------------------------------------------------------
# ✅ Recreate validation dataset & dataloader for inference
# ------------------------------------------------------------
from paired_dataset import PairedFaceDataset
from torch.utils.data import DataLoader
from pathlib import Path

lr_dir = Path('/workspace/innovision/test/lowres_test/LowRes_Generated_Valid')
hr_dir = Path('/workspace/innovision/test/highres_test/HighResolution_test')

# Use same patch size as last training phase
patch_size = 384  

val_dataset = PairedFaceDataset(lr_dir, hr_dir, patch_size=patch_size, augment=False)

# Small batch since we only sample 5 images
val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FastFaceSR_V4(scale=8, base_channels=48, num_groups=3, blocks_per_group=4).to(device)

checkpoint = torch.load("checkpoints_fast_v4_optimized/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"✅ Loaded model with PSNR: {checkpoint['psnr']:.2f} dB")
model.eval()

# ─────────────────────────────────────────────────────────────
# 📸 GENERATE SAMPLE SR OUTPUTS (5 images)
# ─────────────────────────────────────────────────────────────
from torchvision.utils import save_image
import os

sample_dir = "samples_output"
os.makedirs(sample_dir, exist_ok=True)

print("\n📸 Generating sample SR outputs for 5 images...")

# Take 5 samples from validation set
model.eval()
count = 0

with torch.no_grad(), autocast('cuda'):
    for lr, hr in val_loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        sr = torch.clamp(sr, -1, 1)

        for i in range(lr.size(0)):
            if count >= 5: break

            # Save images: input LR, model SR, and GT HR
            save_image((lr[i] * 0.5 + 0.5), f"{sample_dir}/sample_{count}_lr.png")
            save_image((sr[i] * 0.5 + 0.5), f"{sample_dir}/sample_{count}_sr.png")
            save_image((hr[i] * 0.5 + 0.5), f"{sample_dir}/sample_{count}_hr.png")

            print(f"✅ Saved sample {count} → LR / SR / HR images")
            count += 1

        if count >= 5:
            break

print(f"\n✅ Done! Samples stored in: {sample_dir}")

def main():
    print("\n" + "="*70)
    print("🚀 FastFaceSR V4 - Optimized for 30+ dB PSNR")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_dir = Path('/workspace/innovision/test/lowres_test/LowRes_Generated_Valid')
    hr_dir = Path('/workspace/innovision/test/highres_test/HighResolution_test')
    checkpoint_dir = Path('checkpoints_fast_v4_optimized')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Simplified 2-phase progressive
    phases = [
        {'patch_size': 256, 'epochs': 40, 'batch_size': 32, 'lr': 1e-4},
        {'patch_size': 384, 'epochs': 40, 'batch_size': 16, 'lr': 5e-5},
    ]
    
    print(f"\n📊 Optimized Configuration:")
    print(f"  ├─ Focus: Pixel accuracy (L1 + MSE)")
    print(f"  ├─ No GAN (adds complexity)")
    print(f"  ├─ Progressive: 2 phases (256→384)")
    print(f"  ├─ Total epochs: 80")
    print(f"  └─ Target: 30+ dB PSNR\n")
    
    # Smaller model for faster convergence (48 channels)
    model = FastFaceSR_V4(scale=8, base_channels=48, num_groups=3, blocks_per_group=4).to(device)
    
    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Simple loss focused on pixel accuracy
    criterion = OptimizedLoss().to(device)
    
    # Create LPIPS once (avoid reloading)
    print("\n✓ Loading LPIPS metric (once)...")
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    
    scaler = GradScaler()
    best_psnr = 0
    global_epoch = 0
    start_time = time.time()
    
    # Progressive training
    for phase_idx, phase_config in enumerate(phases, 1):
        patch_size = phase_config['patch_size']
        num_epochs = phase_config['epochs']
        batch_size = phase_config['batch_size']
        lr = phase_config['lr']
        
        print(f"\n{'='*70}")
        print(f"📍 PHASE {phase_idx}/2: {patch_size}×{patch_size} | {num_epochs} epochs | LR={lr}")
        print(f"{'='*70}")
        
        # Create datasets
        train_dataset = PairedFaceDataset(lr_dir, hr_dir, patch_size=patch_size, augment=True)
        val_dataset = PairedFaceDataset(lr_dir, hr_dir, patch_size=patch_size, augment=False)
        
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        
        # Cosine scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)
        
        # Phase training loop
        for epoch in range(num_epochs):
            global_epoch += 1
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            scheduler.step()
            
            # Validate every 5 epochs
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                psnr, ssim, lpips_val = validate(model, val_loader, device, lpips_metric)
                
                # Save best model
                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips_val,
                        'epoch': global_epoch,
                        'phase': phase_idx
                    }, checkpoint_dir / 'best_model.pth')
                    print(f"P{phase_idx} E{epoch+1:02d}: Loss={train_loss:.4f} | "
                          f"PSNR={psnr:.2f} ⭐ | SSIM={ssim:.4f} | LPIPS={lpips_val:.4f}")
                else:
                    print(f"P{phase_idx} E{epoch+1:02d}: Loss={train_loss:.4f} | "
                          f"PSNR={psnr:.2f} | SSIM={ssim:.4f} | LPIPS={lpips_val:.4f}")
            else:
                print(f"P{phase_idx} E{epoch+1:02d}: Loss={train_loss:.4f}")
        
        print(f"\n✓ Phase {phase_idx} complete | Best: {best_psnr:.2f} dB")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🏁 TRAINING COMPLETE")
    print("="*70)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print("="*70)
    print(f"\n✓ Best model: {checkpoint_dir}/best_model.pth\n")


if __name__ == '__main__':
    main()
