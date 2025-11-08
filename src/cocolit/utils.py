import os
import yaml
from collections import defaultdict

import wandb
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from nibabel.processing import resample_from_to, resample_to_output

from .defaults import DEFAULT_ZSCORES_PARAMS


class MetricAverage:
    """
    Collect metrics and return averages
    """
    
    def __init__(self):
        self._metrics = defaultdict(lambda: {'sum': 0, 'cnt': 0})
    
    def clear(self) -> None:
        for _dict in self._metrics.values():
            _dict['sum'] = _dict['cnt'] = 0
    
    def store(self, metric_name: str, metric_value: float) -> None:
        self._metrics[metric_name]['sum'] += metric_value
        self._metrics[metric_name]['cnt'] += 1
        
    def keys(self) -> list:
        return list(self._metrics.keys())
        
    def get_avg(self, metric_name: str) -> float:
        _sum = self._metrics[metric_name]['sum']
        _cnt = self._metrics[metric_name]['cnt'] 
        return (_sum / _cnt) if _cnt > 0 else 0.
    

class KLDivergenceLoss:
    """
    A class for computing the Kullback-Leibler divergence loss.
    """
    
    def __call__(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        """
        Computes the KL divergence loss for the given parameters.

        Args:
            z_mu (Tensor):  The mean of the distribution.
            z_sigma (Tensor): The standard deviation of the distribution.

        Returns:
            Tensor: The computed KL divergence loss, averaged over the batch size.
        """

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def log_reconstruction(tag, image, recon, epoch):
    """
    Display reconstruction in TensorBoard during AE training.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 5))
    for _ax in ax.flatten(): _ax.set_axis_off()

    if image.ndim == 4: image = image[0] 
    if recon.ndim == 4: recon = recon[0]

    ax[0, 0].set_title('original image', color='cyan')
    ax[0, 0].imshow(image[image.shape[0] // 2, :, :], cmap='magma')
    ax[0, 1].imshow(image[:, image.shape[1] // 2, :], cmap='magma')
    ax[0, 2].imshow(image[:, :, image.shape[2] // 2], cmap='magma')

    ax[1, 0].set_title('reconstructed image', color='magenta')
    ax[1, 0].imshow(recon[recon.shape[0] // 2, :, :], cmap='magma')
    ax[1, 1].imshow(recon[:, recon.shape[1] // 2, :], cmap='magma')
    ax[1, 2].imshow(recon[:, :, recon.shape[2] // 2], cmap='magma')

    fig.tight_layout()
    wandb.log({tag: wandb.Image(fig), 'epoch': epoch})
    plt.close()
    

def log_generation(epoch,
                   sampler,
                   device,
                   save_dir):
    """
    Visualize the generation on tensorboard, saving locally on failure.
    """    
    image = sampler.sample(device=device)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=3, figsize=(6, 3))
    for _ax in ax.flatten(): _ax.set_axis_off()

    ax[0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    ax[1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
    ax[2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')

    fig.tight_layout()
    fig.suptitle(f"Generate scan (epoch: {epoch})")

    try:
        wandb.log({"plot": wandb.Image(fig), 'epoch': epoch})
    except Exception as e:
        print(f"Wandb logging failed for epoch {epoch}: {e}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"generation_epoch_{epoch}.png")
        print(f"Saving image locally to: {save_path}")
        fig.savefig(save_path)

    plt.close(fig)
    

def log_prediction(epoch,
                   sampler,
                   scan_z,
                   cond_scan,
                   target_pet,
                   device,
                   save_dir,
                   las_m=4):
    """
    Visualize the generation on tensorboard, saving locally on failure.
    """
    image = sampler._sample(scan_z, las_m, device)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    for _ax in ax.flatten(): _ax.set_axis_off()

    ax[0, 1].set_title('conditioning T1w MRI')
    ax[0, 0].imshow(cond_scan[cond_scan.shape[0] // 2, :, :], cmap='gray')
    ax[0, 1].imshow(cond_scan[:, cond_scan.shape[1] // 2, :], cmap='gray')
    ax[0, 2].imshow(cond_scan[:, :, cond_scan.shape[2] // 2], cmap='gray')

    ax[1, 1].set_title('Target Amyloid SUVR')
    ax[1, 0].imshow(target_pet[target_pet.shape[0] // 2, :, :], cmap='jet')
    ax[1, 1].imshow(target_pet[:, target_pet.shape[1] // 2, :], cmap='jet')
    ax[1, 2].imshow(target_pet[:, :, target_pet.shape[2] // 2], cmap='jet')

    ax[2, 1].set_title('Predicted Amyloid SUVR')
    ax[2, 0].imshow(image[image.shape[0] // 2, :, :], cmap='jet')
    ax[2, 1].imshow(image[:, image.shape[1] // 2, :], cmap='jet')
    ax[2, 2].imshow(image[:, :, image.shape[2] // 2], cmap='jet')

    fig.tight_layout()
    fig.suptitle(f"Generate scan (epoch: {epoch})")

    try:
        wandb.log({"plot": wandb.Image(fig), 'epoch': epoch})
    except Exception as e:
        print(f"Wandb logging failed for epoch {epoch}: {e}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"prediction_epoch_{epoch}.png")
        print(f"Saving image locally to: {save_path}")
        fig.savefig(save_path)

    plt.close(fig)


def unpad(z: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Remove the padding from a tensor. If padding is odd, the extra dim is
    removed from the tail.
    """
    current_shape = np.array(z.shape)
    target_shape = np.array(target_shape)

    if current_shape.ndim != target_shape.ndim:
        # check that the target shape has the same dimension of the z vector.
        raise ValueError('The target shape does not match with the shape of the given vector.')
    
    if not np.all(current_shape >= target_shape):
        # check that the target shape is smaller (in each dimension) than the shape of z. 
        # this ensures that no negative values are obtaining when calculating the padding.
        raise ValueError('The target shape is bigger than the shape of the given vector.')

    diff = current_shape - target_shape
    pad_start = diff // 2
    pad_end = diff - pad_start

    slices = [
        slice(start, c_shape - end)
        for start, end, c_shape in zip(pad_start, pad_end, current_shape)
    ]

    return z[tuple(slices)]


def convert_to_suvr(x: torch.Tensor):
    """
    Converts x (with z-score intensities) back to the SUVR domain.
    """
    zscore_mean = DEFAULT_ZSCORES_PARAMS['suvr_mean']
    zscore_std  = DEFAULT_ZSCORES_PARAMS['suvr_std']
    return (x * zscore_std) + zscore_mean


def save_suvr(predicted_suvr, mri_input_path, suvr_output_path):
    """
    Utility function to save the SUVR in the same space as the input MRI

    Args:
        predicted_suvr (np.ndarray): The predicted SUVR map as torch tensor (C x H x W x D).
        mri_input_path (str): The file path to the input MRI scan
        suvr_output_path (str): The desired file path to save the output SUVR map.
    """    
    mri = nib.load(mri_input_path)
    mri_resampled = resample_to_output(mri, voxel_sizes=1.5)
    input_shape = mri_resampled.shape    
    predicted_suvr = unpad(predicted_suvr.squeeze(0), input_shape).cpu().numpy()
    predicted_suvr = nib.nifti1.Nifti1Image(predicted_suvr.astype(np.float32), mri_resampled.affine)
    predicted_suvr = resample_from_to(predicted_suvr, mri)
    predicted_suvr.to_filename(suvr_output_path)