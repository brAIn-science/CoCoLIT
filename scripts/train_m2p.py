import os
import shutil
import argparse
from datetime import datetime

import wandb
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from monai import transforms
from monai.utils import set_determinism
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm

from cocolit import utils
from cocolit import networks
from cocolit.data import load_latents_data
from cocolit.sampling import CondDistributionSampler
from cocolit.wisl import WeightedImageSpaceLoss


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_volume_for_visualization(volume_path, confs):
    """
    Load a volume 
    """
    transforms_fn = transforms.Compose([
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=confs['dataloader']['gt_suvr_voxel_spacing']),
        transforms.DivisiblePad(k=confs['dataloader']['vae_divisible_pad_k']),
        transforms.ScaleIntensity(minv=0, maxv=1) # that's ok for visualization
    ])
    return transforms_fn(volume_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    #===================================================
    # Settings.
    #===================================================

    confs  = utils.load_config(args.config)
    run_name   = datetime.now().strftime(f"IST-%d-%m-%y-%H-%M-%S")
    print(f'Using device: {DEVICE}\nRunning with run name: {run_name}')

    #===================================================
    # Preparing datasets and dataloaders
    #===================================================

    train_loader, valid_loader, test_loader = load_latents_data(**confs['dataloader'])     
    print("DataLoaders initialized.")

    #===================================================
    # Initialising the models
    #===================================================

    if confs['vae_checkpoints'] is None:
        print('Error: pretrained PET VAE checkpoint is missing.')
        exit(1)
        
    if confs['unet_checkpoints'] is None:
        print('Error: pretrained diffusion U-Net checkpoint is missing.')
        exit(1)
    
    autoencoder = networks.create_maisi_vae(
        args=confs['vae_args'],  
        device=DEVICE,
        checkpoint=confs['vae_checkpoints']
    )
    
    # Freeze the encoder part of the VAE
    utils.freeze_module(autoencoder.encoder)
    
    diffusion = networks.create_diffusion(
        args=confs['unet_args'],
        device=DEVICE,
        checkpoint=confs['unet_checkpoints']
    )
        
    # Freeze the diffusion model.
    utils.freeze_module(diffusion)
        
    controlnet = networks.create_controlnet(
        args=confs['controlnet_args'],
        device=DEVICE,
        checkpoint=confs['controlnet_checkpoints']
    )    
    
    diffusion_scheduler = DDPMScheduler(**confs['ddpm_sched_args'])
    
    diffusion_sampler = CondDistributionSampler(
        source_vae=None,
        target_vae=autoencoder,
        udiffusion=diffusion,
        controlnet=controlnet,
        noise_scheduler=diffusion_scheduler
    )
    
    autoencoder.train().to(DEVICE)
    controlnet.train().to(DEVICE)
    print('Models initialised.')
        
    #===================================================
    # Print summary of the first batch
    #===================================================

    batch = next(iter(train_loader))
    print("Batch size:", batch["image"].shape[0])
    print("PET Latent shape:", batch["image"].shape)
    print("MRI Latent shape:", batch["cond"].shape)
    print("PET Volume shape:", batch["x_0"].shape)

    #===================================================
    # Creating the result directory
    #===================================================

    results_dir = os.path.join(confs['output_dir'], run_name)
    os.makedirs(results_dir, exist_ok=True)
        
    config_copy_path = os.path.join(results_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_copy_path)
    
    gens_dir = os.path.join(results_dir, 'images')
    os.makedirs(gens_dir, exist_ok=True)
    
    print(f"Results directory created: {results_dir}")
    print(f"Configuration file copied to: {config_copy_path}")

    #===================================================
    # Initialising wandb
    #===================================================

    wandb.init(
        entity=confs["wandb"]["entity"],
        project=confs["wandb"]["project"], 
        config=confs, 
        name=run_name
    )
    
    #=================================================================
    # Initialise the optimizers
    #=================================================================

    lr       = confs['training']['lr']
    n_epochs = confs['training']['n_epochs']
    opt_ckpt = confs['training']['optim_checkpoint']

    wisl = WeightedImageSpaceLoss(
        target_vae=autoencoder,
        noise_scheduler=diffusion_scheduler,
    )

    optimizer = torch.optim.AdamW([
        {'params': autoencoder.parameters(), 'lr': lr},
        {'params': controlnet.parameters(),  'lr': lr}
    ])

    if opt_ckpt is not None:
        print(f'Loading optimizer from {opt_ckpt}')
        optimizer.load_state_dict(torch.load(opt_ckpt))

    scaler    = GradScaler(device=DEVICE)
    
    #=================================================================
    # Training loop
    #=================================================================
    
    train_metrics = utils.MetricAverage()
    valid_metrics = utils.MetricAverage()
    
    loaders         = { 'train': train_loader,  'valid': valid_loader }
    metrics         = { 'train': train_metrics, 'valid': valid_metrics }
    global_counter  = { 'train': 0, 'valid': 0 }
    best_valid_loss = float('inf')


    for epoch in range(n_epochs):
        
        for mode in loaders.keys():
            
            loader = loaders[mode]
            metric = metrics[mode]
            
            autoencoder.train() if mode == 'train' else autoencoder.eval()
            controlnet.train()  if mode == 'train' else controlnet.eval()
            diffusion.train()   if mode == 'train' else diffusion.eval()
            
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in progress_bar:
                            
                with autocast(device_type=DEVICE, enabled=True):
                        
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    
                    controlnet_condition    = batch['cond'].to(DEVICE)
                    suvr_latents            = batch['image'].to(DEVICE)
                    suvr_true               = batch['x_0'].to(DEVICE)
                    
                    n = suvr_latents.shape[0]
                    
                    with torch.set_grad_enabled(mode == 'train'):
                        
                        noise = torch.randn_like(suvr_latents).to(DEVICE)
                        timesteps = torch.randint(0, diffusion_scheduler.num_train_timesteps, (n,), device=DEVICE).long()
                        images_noised = diffusion_scheduler.add_noise(suvr_latents, noise=noise, timesteps=timesteps)

                        down_h, mid_h = controlnet(
                            x=images_noised.float(), 
                            timesteps=timesteps, 
                            controlnet_cond=controlnet_condition.float()
                        )

                        noise_pred = diffusion(
                            x=images_noised.float(), 
                            timesteps=timesteps, 
                            down_block_additional_residuals=down_h,
                            mid_block_additional_residual=mid_h
                        )
                        
                        wisl_loss = wisl(suvr_true, images_noised, noise_pred, timesteps, DEVICE)
                        diff_loss = F.mse_loss(noise_pred.float(), noise.float())
                        loss = diff_loss + wisl_loss

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                wandb.log({ f'{mode}/batch-loss': loss.item(), 'iteration': global_counter[mode] })
                wandb.log({ f'{mode}/batch-diff': diff_loss.item(), 'iteration': global_counter[mode] })
                wandb.log({ f'{mode}/batch-wisl': wisl_loss.item(), 'iteration': global_counter[mode] })
                
                global_counter[mode] += 1
                
                metric.store('epoch_loss', loss.item())
                metric.store('diff_loss', diff_loss.item())
                metric.store('wisl_loss', wisl_loss.item())
        
            # end of epoch
            epoch_diff = metric.get_avg('diff_loss')
            epoch_wisl = metric.get_avg('wisl_loss')
            epoch_loss = metric.get_avg('epoch_loss')
            
            wandb.log({ f'{mode}/epoch-diff': epoch_diff, 'epoch': epoch })
            wandb.log({ f'{mode}/epoch-wisl': epoch_wisl, 'epoch': epoch })
            wandb.log({ f'{mode}/epoch-loss': epoch_loss, 'epoch': epoch })

            metric.clear()

            if mode == 'valid' and epoch_loss < best_valid_loss:
                best_valid_loss = epoch_loss    
                print(f"New best loss. Saving training state.")
                torch.save(controlnet.state_dict(),  os.path.join(results_dir, f"best_controlnet.pth"))
                torch.save(autoencoder.state_dict(), os.path.join(results_dir, f"best_autoencoder.pth"))
                torch.save(optimizer.state_dict(),   os.path.join(results_dir, f"best_optimizer.pth"))
                        
                # log one prediction
                cond_scan  = load_volume_for_visualization(batch['scan_path'][0], confs).squeeze(0).numpy()
                target_pet = load_volume_for_visualization(batch['suvr_path'][0], confs).squeeze(0).numpy()
                
                utils.log_prediction(
                    epoch=epoch, 
                    sampler=diffusion_sampler, 
                    scan_z=controlnet_condition[0],
                    cond_scan=cond_scan,
                    target_pet=target_pet,
                    device=DEVICE,
                    save_dir=gens_dir
                )