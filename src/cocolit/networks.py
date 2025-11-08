import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet, ControlNet
from monai.networks.nets import PatchDiscriminator
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

from cocolit.defaults import (
    DEFAULT_VAE_ARGS,
    DEFAULT_DIFFUSION_ARGS,
    DEFAULT_CONTROLNET_ARGS,
)


def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize 

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        device = next(network.parameters()).device # Use the same device as the model
        network.load_state_dict(torch.load(checkpoints_path, map_location=device))

    return network


def create_maisi_vae(args: Dict = DEFAULT_VAE_ARGS, device: str = 'cuda', checkpoint: Optional[str] = None):
    """
    MAISI VAE from:
    Guo, Pengfei, et al. "MAISI: Medical AI for Synthetic Imaging." CoRR (2024).
    """
    net = AutoencoderKlMaisi(**args).to(device)
    return load_if(checkpoint, net)
    
    
def create_patch_discriminator(args: Dict, device: str = 'cuda', checkpoint: Optional[str] = None):
    """
    Patch Discriminator from:
    Pinaya, Walter HL, et al. "Brain imaging generation with latent diffusion models." 
    MICCAI Workshop on Deep Generative Models. Cham: Springer Nature Switzerland, 2022.
    """
    net = PatchDiscriminator(**args).to(device)
    return load_if(checkpoint, net)


def create_diffusion(args: Dict = DEFAULT_DIFFUSION_ARGS, device: str = 'cuda', checkpoint: Optional[str] = None):
    """
    Latent Diffusion Model from:
    Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." 
    Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.    
    """
    net = DiffusionModelUNet(**args).to(device)
    return load_if(checkpoint, net)


def create_controlnet(args: Dict = DEFAULT_CONTROLNET_ARGS, device: str = 'cuda', checkpoint: Optional[str] = None):
    """
    ControlNet from:
    Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding conditional control to text-to-image diffusion models." 
    Proceedings of the IEEE/CVF international conference on computer vision. 2023.    
    """
    net = ControlNet(**args).to(device)
    return load_if(checkpoint, net)