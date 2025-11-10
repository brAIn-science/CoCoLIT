import torch
import torch.nn as nn
from torch.amp import autocast
from monai.networks.schedulers import Scheduler
from monai import transforms
from tqdm.rich import tqdm

from .utils import unpad


class CondDistributionSampler:
    
    def __init__(
        self, 
        source_vae: nn.Module, 
        target_vae: nn.Module,
        udiffusion: nn.Module,
        controlnet: nn.Module,
        noise_scheduler: Scheduler,
        scale_factor: float = 1.,
        verbose: bool = True,
        vae_space_shape = (4, 32, 38, 32),
        ldm_space_shape = (4, 32, 40, 32)
    ):
        """
        Initializes the DistributionSampler instance.

        Args:
            source_vae (nn.Module): The pretrained source VAE model.
            target_vae (nn.Module): The pretrained target VAE model.
            udiffusion (nn.Module): The pretrained U-Net diffusion model.
            controlnet (nn.Module): The pretrained ControlNet model.
            noise_scheduler (Scheduler): The noise scheduler for the diffusion process.
            scale_factor (int, optional): A scaling factor for the latent vector. Defaults to 1.
            verbose (bool, optional): If True, enables progress bar display. Defaults to True.
            vae_space_shape (tuple, optional): The latent space shape for the VAE.
            ldm_space_shape (tuple, optional): The latent space shape for the diffusion model.
        """
        self.source_vae = source_vae
        self.target_vae = target_vae
        self.udiffusion = udiffusion
        self.controlnet = controlnet
        self.noise_scheduler = noise_scheduler
        
        self.scale_factor = scale_factor
        self.verbose = verbose
        self.vae_space_shape = vae_space_shape
        self.ldm_space_shape = ldm_space_shape
        
        # warning: the transforms don't work on the channel. In this case
        # the channel is considered in the dimensions to pad, therefore we 
        # have to add another dimension to be able to pad the channel too.
        # this is ok because it means we can apply this with the batch dim.
        self.to_ldm_space = transforms.SpatialPad(ldm_space_shape)
    
        
    @torch.inference_mode()
    def sample(
        self,
        source_x: torch.Tensor,
        las_m: int,
        device: str
    ):
        """
        Generates a sample from the target distribution conditioned on a source input.

        Args:
            source_x (torch.Tensor): The input tensor from the source domain (C x H x W x D).
            las_m (int): The number of parallel samples for Latent Average Stabilization (LAS).
            device (str): The device to run the models on (e.g., 'cuda' or 'cpu').

        Returns:
            torch.Tensor: The generated output tensor in the target domain.
        """
        self.source_vae.eval()
        self.target_vae.eval()
        self.udiffusion.eval()
        self.controlnet.eval()
        
        if source_x.ndim == 4:
            # add batch size.
            source_x = source_x.unsqueeze(0)

        with autocast(device_type=device, dtype=torch.float16, enabled=True):
            source_z = self.source_vae(source_x.to(device))[1]

        source_z = self.to_ldm_space(source_z.cpu()).to(device)
        return self._sample(source_z, las_m, device)
        
        
    @torch.inference_mode()
    def _sample(
        self,
        source_z: torch.Tensor,
        las_m: int,
        device: str
    ):
        """
        Generates a sample from the target distribution conditioned on a source input.

        Args:
            source_z (torch.Tensor): The input tensor projected in the latent space.
            las_m (int): The number of parallel samples for Latent Average Stabilization (LAS).
            device (str): The device to run the models on (e.g., 'cuda' or 'cpu').

        Returns:
            torch.Tensor: The generated output tensor in the target domain.
        """
        source_z = source_z.to(device)

        # if performing LAS, we repeat the inputs for the diffusion process
        # m times (as specified in the paper) and perform the reverse diffusion
        # process in parallel to avoid overheads.
        if las_m > 1: source_z = source_z.repeat(las_m, 1, 1, 1, 1) 
    
        # this is z_T - the starting noise.
        z = torch.randn(*source_z.shape).to(device)
        
        # noise scheduling (beta_t)
        progress_bar = tqdm(self.noise_scheduler.timesteps) if self.verbose else self.noise_scheduler.timesteps

        for t in progress_bar:
            with autocast(device_type=device, enabled=True):

                # convert the timestep to a tensor.
                timestep = torch.tensor([t]).repeat(las_m).to(device)

                # obtain the ControlNet conditioning to inject to the U-Net using
                # the source latent z, the noise and the timestep.
                down_h, mid_h = self.controlnet(
                    x=z.float(),
                    timesteps=timestep,
                    controlnet_cond=source_z.float()
                )

                # the diffusion takes the intermediate features and predicts
                # the noise. This is why we conceptualize the two networks as
                # as a unified network.
                pred_noise = self.udiffusion(
                    x=z.float(), 
                    timesteps=timestep, 
                    down_block_additional_residuals=down_h,
                    mid_block_additional_residual=mid_h
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = self.noise_scheduler.step(pred_noise, t, z)

        
        # we follow the LAS procedure and average the m noise vectors
        z = (z / self.scale_factor).mean(axis=0)
        
        # before feeding the average latent to the decoder to project it
        # into the image space, we have to reshape the latent to the correct
        # shape (i.e., the VAE latent space dimensions, which might be slightly
        # different from the latent diffusion ones).
        z = unpad(z.cpu(), self.vae_space_shape)
        
        # now it's time to decode the average latent into the image space.
        # For some reasons the MAISI VAE implementation requires casting to float16.
        with autocast(device_type=device, dtype=torch.float16, enabled=True):
            x = self.target_vae.decode_stage_2_outputs(z.unsqueeze(0).to(device))
        
        x = x.squeeze(0).cpu()
        return x
    
    
class UncondDistributionSampler:
    
    def __init__(
        self, 
        target_vae: nn.Module,
        udiffusion: nn.Module,
        noise_scheduler: Scheduler,
        scale_factor: float = 1.,
        verbose: bool = True,
        vae_space_shape = (4, 32, 38, 32),
        ldm_space_shape = (4, 32, 40, 32)
    ):
        """
        Initializes the DistributionSampler instance.

        Args:
            target_vae (nn.Module): The pretrained target VAE model.
            udiffusion (nn.Module): The pretrained U-Net diffusion model.
            noise_scheduler (Scheduler): The noise scheduler for the diffusion process.
            scale_factor (int, optional): A scaling factor for the latent vector. Defaults to 1.
            verbose (bool, optional): If True, enables progress bar display. Defaults to True.
            vae_space_shape (tuple, optional): The latent space shape for the VAE.
            ldm_space_shape (tuple, optional): The latent space shape for the diffusion model.
        """
        self.target_vae = target_vae
        self.udiffusion = udiffusion
        self.noise_scheduler = noise_scheduler
        
        self.scale_factor = scale_factor
        self.verbose = verbose
        self.vae_space_shape = vae_space_shape
        self.ldm_space_shape = ldm_space_shape
    
        
    @torch.inference_mode()
    def sample(
        self,
        device: str
    ):
        """
        Generates a sample from the target distribution conditioned on a source input.

        Args:
            device (str): The device to run the models on (e.g., 'cuda' or 'cpu').

        Returns:
            torch.Tensor: The generated output tensor in the target domain.
        """
        self.target_vae.eval()
        self.udiffusion.eval()

        # this is z_T - the starting noise.
        z = torch.randn(1, *self.ldm_space_shape).to(device)
        
        # noise scheduling (beta_t)
        progress_bar = tqdm(self.noise_scheduler.timesteps) if self.verbose else self.noise_scheduler.timesteps

        for t in progress_bar:
            with autocast(device_type=device, enabled=True):

                # convert the timestep to a tensor.
                timestep = torch.tensor([t]).to(device)

                # predict the noise component.
                pred_noise = self.udiffusion(x=z.float(), timesteps=timestep)

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = self.noise_scheduler.step(pred_noise, t, z)

        
        # divide by the scale factor (see Rombach et al.)
        z = z / self.scale_factor
        
        # before feeding the average latent to the decoder to project it
        # into the image space, we have to reshape the latent to the correct
        # shape (i.e., the VAE latent space dimensions, which might be slightly
        # different from the latent diffusion ones).
        z = unpad(z.squeeze(0).cpu(), self.vae_space_shape)
        
        # now it's time to decode the average latent into the image space.
        # For some reasons the MAISI VAE implementation requires casting to float16.
        with autocast(device_type=device, dtype=torch.float16, enabled=True):
            x = self.target_vae.decode_stage_2_outputs(z.unsqueeze(0).to(device))
        
        x = x.squeeze(0).cpu()
        return x