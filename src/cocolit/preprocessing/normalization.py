"""
Code is borrowed from and adapted from:
https://github.com/jcreinhold/intensity-normalization

NOTE:   I love the intensity-normalization package, but it has a restrictive 
        dependency on Python 3.11 or later, which excludes a large part of the 
        community that does not regularly update their Python version. Therefore, 
        I decided to remove the dependency and keep only this normalization function.
"""
from typing import Optional

import numpy as np
import nibabel as nib
from scipy.signal import argrelmax
from scipy.stats import gaussian_kde


def whitestripe_normalize(
    img: nib.Nifti1Image,
    mask: Optional[nib.Nifti1Image] = None,
    modality: str = 'T1',
    width: float = 0.05,
    width_l: Optional[float] = None,
    width_u: Optional[float] = None
) -> nib.Nifti1Image:
    """
    Applies WhiteStripe (normal-appearing white matter) normalization.
    borrowed from: https://github.com/jcreinhold/intensity-normalization
    """
    width_l = width_l if width_l is not None else width
    width_u = width_u if width_u is not None else width
    modality = modality.upper()
    
    data = img.get_fdata()

    # extract foreground
    if mask is not None:
        mask_data = mask.get_fdata().astype(bool)
        foreground = data[mask_data]
    else:
        mask_data = data > 0
        foreground = data[mask_data]

    if len(foreground) == 0:
        raise ValueError("No foreground voxels found")

    def smooth_histogram(image_data):
        """
        Use kernel density estimate to get smooth histogram.
        """
        image_vec = image_data.flatten().astype(np.float64)
        kde = gaussian_kde(image_vec)
        grid = np.linspace(image_vec.min(), image_vec.max(), 80)
        pdf = kde(grid)
        
        return grid, pdf

    # find tissue mode based on modality using scipy argrelmax
    if modality == 'T1':
        threshold = float(np.percentile(foreground, 96.0))
        valid_data = foreground[foreground <= threshold]
        grid, pdf = smooth_histogram(valid_data)
        maxima = argrelmax(pdf)[0]
        wm_mode = float(grid[maxima[-1]]) if len(maxima) > 0 else float(grid[np.argmax(pdf)])

    elif modality in ('T2', 'FLAIR'):
        grid, pdf = smooth_histogram(foreground)
        wm_mode = float(grid[np.argmax(pdf)])

    elif modality == 'PD':
        threshold = float(np.percentile(foreground, 99.0))
        valid_data = foreground[foreground <= threshold]
        grid, pdf = smooth_histogram(valid_data)
        maxima = argrelmax(pdf)[0]
        wm_mode = float(grid[maxima[0]]) if len(maxima) > 0 else float(grid[np.argmax(pdf)])
        
    else:
        # defaulting to t1 logic for any unknown inputs
        threshold = float(np.percentile(foreground, 96.0))
        valid_data = foreground[foreground <= threshold]
        grid, pdf = smooth_histogram(valid_data)
        maxima = argrelmax(pdf)[0]
        wm_mode = float(grid[maxima[-1]]) if len(maxima) > 0 else float(grid[np.argmax(pdf)])

    # calculate bounds and thresholds
    wm_mode_quantile = float(np.mean(foreground < wm_mode))
    lower_bound = max(wm_mode_quantile - width_l, 0.0)
    upper_bound = min(wm_mode_quantile + width_u, 1.0)
    ws_l, ws_u = np.quantile(foreground, (lower_bound, upper_bound))

    # create whitestripe mask and compute stats
    if mask is not None:
        masked_data = data * mask_data
        ws_mask = (masked_data > ws_l) & (masked_data < ws_u)
    else:
        ws_mask = (data > ws_l) & (data < ws_u)

    whitestripe_values = data[ws_mask]

    if len(whitestripe_values) == 0:
        raise ValueError("No voxels found in white stripe region")

    ws_mean = float(np.mean(whitestripe_values))
    ws_std = float(np.std(whitestripe_values))

    if ws_std == 0:
        raise ValueError("Standard deviation of white stripe is zero")

    # transform and normalize
    normalized_data = (data - ws_mean) / ws_std

    # return new nibabel image with original affine/header
    return nib.Nifti1Image(normalized_data, img.affine, img.header)