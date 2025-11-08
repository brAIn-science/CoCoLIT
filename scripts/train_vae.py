import os
import shutil
import argparse
import warnings
from datetime import datetime

import wandb
import torch
from torch.nn import L1Loss
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.utils import set_determinism
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from tqdm import tqdm

from cocolit import utils
from cocolit import networks
from cocolit.data import load_volumetric_data
from mri2pet.gradacc import GradientAccumulation


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
    run_name   = datetime.now().strftime(confs['name'] + f"-%d-%m-%y-%H-%M-%S")
    print(f'Using device: {DEVICE}\nRunning with run name: {run_name}')

    #===================================================
    # Preparing datasets and dataloaders
    #===================================================

    train_loader, valid_loader, test_loader = load_volumetric_data(**confs['dataloader'])     
    print("DataLoaders initialized.")


    #===================================================
    # Initialising the models
    #===================================================
    
    autoencoder = networks.create_maisi_vae(
        args=confs['vae_args'],  
        device=DEVICE,
        checkpoint=confs['vae_checkpoints']
    )
    
    discriminator = networks.create_patch_discriminator(
        args=confs['patch_disc_args'],  
        device=DEVICE,
        checkpoint=confs['patch_disc_checkpoints']        
    )
    
    autoencoder.train().to(DEVICE)
    discriminator.train().to(DEVICE)
    print('Models initialised.')
        
    #===================================================
    # Print summary of the first batch
    #===================================================

    batch = next(iter(train_loader))
    print("Batch size: %s", batch["image"].shape[0])
    print("Image shape: %s", batch["image"].shape)

        
    #===================================================
    # Creating the result directory
    #===================================================

    results_dir = os.path.join(confs['output_dir'], run_name)
    os.makedirs(results_dir, exist_ok=True)
        
    config_copy_path = os.path.join(results_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_copy_path)
    
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

    #===================================================================
    # Initialise the optimizers, schedulers, and gradient accumulators
    #===================================================================

    lr                = confs['training']['lr']
    n_epochs          = confs["training"]["n_epochs"]
    grad_acc_true_bs  = confs['dataloader']['batch_size']
    grad_acc_expt_bs  = confs['training']['grad_acc_expt_bs']

    optimizer_g = torch.optim.Adam(autoencoder.parameters(),   lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=n_epochs, eta_min=lr * .1)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=n_epochs, eta_min=lr * .1)


    gradacc_g = GradientAccumulation(actual_batch_size=grad_acc_true_bs,
                                     expect_batch_size=grad_acc_expt_bs,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_g,
                                     grad_scaler=GradScaler(DEVICE))

    gradacc_d = GradientAccumulation(actual_batch_size=grad_acc_true_bs,
                                     expect_batch_size=grad_acc_expt_bs,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_d, 
                                     grad_scaler=GradScaler(DEVICE))

    #======================================================
    # Run the training loop
    #======================================================

    adv_weight          = float(confs['training']['adv_weight'])        
    perceptual_weight   = float(confs['training']['perceptual_weight']) 
    kl_weight           = float(confs['training']['kl_weight'])         

    l1_loss_fn  = L1Loss()
    kl_loss_fn  = utils.KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, 
                                      network_type="squeeze", 
                                      is_fake_3d=True, 
                                      fake_3d_ratio=0.2).to(DEVICE)

    train_metrics = utils.MetricAverage()
    valid_metrics = utils.MetricAverage()
    best_valid_loss = float('inf')
    
    for epoch in range(confs['training']['n_epochs']):
        
        #===================
        # Train epoch starts
        #===================
        autoencoder.train() 
        discriminator.train()
        
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            # run the "generator" step
            with autocast(device_type=DEVICE, enabled=True):
                images = batch["image"].to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                
                # Compute all the loss components
                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = rec_loss + kld_loss + per_loss + gen_loss
            
            gradacc_g.step(loss_g, step)
            
            # run the discriminator step.
            with autocast(device_type=DEVICE, enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss
            
            gradacc_d.step(loss_d, step)
            
            # Store the losses to log the average epoch loss
            train_metrics.store('Generator/reconstruction_loss',  rec_loss.item())
            train_metrics.store('Generator/perceptual_loss',      per_loss.item())
            train_metrics.store('Generator/adversarial_loss',     gen_loss.item())
            train_metrics.store('Generator/kl_regularization',    kld_loss.item())
            train_metrics.store('Discriminator/adversarial_loss', loss_d.item())

        scheduler_d.step()
        scheduler_g.step()

        # Let's log the average loss for each loss component
        for metric in train_metrics.keys():
            wandb.log({ metric: train_metrics.get_avg(metric), 'epoch': epoch })
        
        # Clear the metrics history before next epoch
        train_metrics.clear()

        #===================
        # Valid epoch starts
        #===================
        autoencoder.eval()
    
        with torch.no_grad():
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                with autocast(device_type=DEVICE, enabled=True):
                    images = batch["image"].to(DEVICE)
                    reconstruction = autoencoder.reconstruct(images)
                    rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                    valid_metrics.store('Valid/reconstruction_loss', rec_loss.item())

        epoch_valid_loss = valid_metrics.get_avg('Valid/reconstruction_loss')
        valid_metrics.clear()
        
        # Log the validation (reconstruction)loss        
        wandb.log({ 'Valid/reconstruction_loss': epoch_valid_loss, 'epoch': epoch })

        try:
            utils.log_reconstruction('Valid/reconstruction', 
                                     image=images[0].detach().cpu(),
                                     recon=reconstruction[0].detach().cpu(),
                                     epoch=epoch)        
        except Exception as e:
            print("There was an error: ", e)
            print("Resuming the training...")


        # if new best validation loss, save best models.
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            print("New best validation loss. Saving training state.")
            torch.save(autoencoder.state_dict(),         os.path.join(results_dir, "best_autoencoder.pth"))
            torch.save(discriminator.state_dict(),       os.path.join(results_dir, "best_discriminator.pth"))
            torch.save(gradacc_g.optimizer.state_dict(), os.path.join(results_dir, "best_optimizer_g.pth"))
            torch.save(gradacc_d.optimizer.state_dict(), os.path.join(results_dir, "best_optimizer_d.pth"))
            
            
    #===================
    # Training completed
    #===================

    wandb.finish()