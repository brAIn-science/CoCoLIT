import os
import shutil
import argparse
from datetime import datetime

import wandb
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from monai.utils import set_determinism
from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm

from cocolit import utils
from cocolit import networks
from cocolit.sampling import UncondDistributionSampler
from cocolit.data import load_latents_data


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    #===================================================
    # Settings.
    #===================================================

    confs  = utils.load_config(args.config)
    run_name   = datetime.now().strftime(f"pet_ldm-%d-%m-%y-%H-%M-%S")
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
    
    autoencoder = networks.create_maisi_vae(
        args=confs['vae_args'],  
        device=DEVICE,
        checkpoint=confs['vae_checkpoints']
    )
    
    diffusion = networks.create_diffusion(
        args=confs['unet_args'],
        device=DEVICE,
        checkpoint=confs['unet_checkpoints']
    )
        
    diffusion_scheduler = DDPMScheduler(**confs['ddpm_sched_args'])
    
    inferer = DiffusionInferer(diffusion_scheduler)
    
    autoencoder.eval().to(DEVICE)
    diffusion.train().to(DEVICE)
    
    diffusion_sampler = UncondDistributionSampler(
        target_vae=autoencoder,
        udiffusion=diffusion,
        noise_scheduler=diffusion_scheduler
    )
    
    print('Models initialised.')
        
    #===================================================
    # Print summary of the first batch
    #===================================================

    batch = next(iter(train_loader))
    print("Batch size:", batch["image"].shape[0])
    print("Latent shape:", batch["image"].shape)

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

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=confs["training"]["lr"])
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
            diffusion.train() if mode == 'train' else diffusion.eval()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in progress_bar:
                            
                with autocast(device_type=DEVICE, enabled=True):
                        
                    if mode == 'train': optimizer.zero_grad(set_to_none=True)
                    latents = batch['image'].to(DEVICE)
                    n = latents.shape[0]
                    
                    with torch.set_grad_enabled(mode == 'train'):
                        
                        noise = torch.randn_like(latents).to(DEVICE)
                        timesteps = torch.randint(0, diffusion_scheduler.num_train_timesteps, (n,), device=DEVICE).long()

                        noise_pred = inferer(
                            inputs=latents, 
                            diffusion_model=diffusion, 
                            noise=noise, 
                            timesteps=timesteps
                        )

                        loss = F.mse_loss( noise.float(), noise_pred.float() )

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                wandb.log({ f'{mode}/batch-mse': loss.item(), 'iteration': global_counter[mode] })
                global_counter[mode] += 1
                metric.store('epoch_loss', loss.item())
        
            # end of epoch
            epoch_loss = metric.get_avg('epoch_loss')
            metric.clear()
            wandb.log({ f'{mode}/epoch-mse': epoch_loss, 'epoch': epoch })

            if mode == 'valid' and epoch_loss < best_valid_loss:
                best_valid_loss = epoch_loss
                print("New best validation loss. Saving training state.")
                torch.save(diffusion.state_dict(), os.path.join(results_dir, "best_diffusion.pth"))
                torch.save(optimizer.state_dict(), os.path.join(results_dir, "best_optimizer.pth"))
            
                # Perform generation and log to W&B
                utils.log_generation(epoch, diffusion_sampler, DEVICE, gens_dir)