from typing import Optional
from functools import partial

import torch
import pandas as pd
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset, CacheDataset
from monai.data.image_reader import NumpyReader



def load_volumetric_data(
    dataset_csv: str,
    image_path_column: str,
    voxel_spacing: float,
    divisible_pad_k: int,
    batch_size: int,
    do_augmentation: bool = True,
    do_caching: bool = False,
    do_zscoring: bool = False,
    zscore_mean: Optional[float] = None,
    zscore_std: Optional[float] = None
):
    """
    Load volumetric data (either MRIs or PETs)
    """
 
    df = pd.read_csv(dataset_csv)
    train_data = df[df.split == 'train'].to_dict(orient='records')
    valid_data = df[df.split == 'valid'].to_dict(orient='records')
    test_data  = df[df.split ==  'test'].to_dict(orient='records')

    normalization = transforms.ScaleIntensityd(minv=0, maxv=1, keys=['image']) if not do_zscoring else \
        transforms.NormalizeIntensityd(keys=["image"],subtrahend=zscore_mean, divisor=zscore_std)
    
    loading_transforms = [
        transforms.CopyItemsd(keys=[image_path_column], names=['image']),
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Spacingd(keys=['image'], pixdim=voxel_spacing),
        transforms.DivisiblePadd(keys=['image'], k=divisible_pad_k),
        normalization
    ]
    
    augmentation = [
        transforms.RandAffined(keys=["image"], 
                               prob=0.3,                           
                               rotate_range=(0.05, 0.05, 0.05),
                               scale_range=(0.05, 0.05, 0.05),
                               translate_range=(2, 2, 2))
    ] if do_augmentation else []
    
    train_transforms = transforms.Compose(loading_transforms + augmentation)
    valid_transforms = transforms.Compose(loading_transforms)

    trainset = CacheDataset(train_data, train_transforms, cache_rate=1, num_workers=8) \
        if do_caching else Dataset(train_data, train_transforms)

    validset = CacheDataset(valid_data, valid_transforms, cache_rate=1, num_workers=8) \
        if do_caching else Dataset(valid_data, valid_transforms)

    testset  = CacheDataset(test_data,  valid_transforms, cache_rate=1, num_workers=8) \
        if do_caching else Dataset(test_data, valid_transforms)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=False) \
        if len(trainset) > 0 else None
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False) \
        if len(validset) > 0 else None
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False) \
        if  len(testset) > 0 else None
    
    return train_loader, valid_loader, test_loader


def load_latents_data(
    dataset_csv: str,
    latent_path_column: str,
    divisible_pad_k: int,
    batch_size: int,
    do_caching: bool = False,
    cond_latent_path_column: Optional[str] = None,
    
    # Params for WISL (ControlNet training stage)
    load_gt_suvr: bool = False,
    gt_suvr_path_column: Optional[str] = None,
    gt_suvr_voxel_spacing: Optional[tuple] = None,
    gt_suvr_do_zscoring: Optional[bool] = True,
    gt_suvr_zscore_mean: Optional[float] = None,
    gt_suvr_zscore_std: Optional[float] = None,
    vae_divisible_pad_k: Optional[int] = None,
    
):
    """
    Load latent vectors (from either MRIs or PETs)
    """
    df = pd.read_csv(dataset_csv)
    train_data = df[df.split == 'train'].to_dict(orient='records')
    valid_data = df[df.split == 'valid'].to_dict(orient='records')
    test_data  = df[df.split ==  'test'].to_dict(orient='records')
    
    npz_reader = NumpyReader(npz_keys=['data'])
    
    # for the controlnet training we need both the MRI latent (`cond_latent_path_column`)
    # and the PET latent (`latent_path_column`).
    source_keys = [latent_path_column, cond_latent_path_column] if \
        cond_latent_path_column is not None else [latent_path_column]
    
    target_keys = ['image', 'cond'] \
        if cond_latent_path_column else ['image']
        
    transforms_list = [
        transforms.CopyItemsd(keys=source_keys, names=target_keys),
        transforms.LoadImaged(keys=target_keys, reader=npz_reader),
        transforms.EnsureChannelFirstd(keys=target_keys, channel_dim=0), 
        transforms.DivisiblePadd(keys=target_keys, k=divisible_pad_k, mode='constant'),        
    ]

    # This is useful if we want to train the ControlNet and the PET VAE decoder 
    # using WISL. In this case, we load the SUVR as we do during the VAE training. 
    # All these parameters are optional, so that the load function is compatible with 
    # previous LDM training. 
    if load_gt_suvr:
        
        normalization = transforms.ScaleIntensityd(minv=0, maxv=1, keys=['x_0']) \
            if not gt_suvr_do_zscoring else \
            transforms.NormalizeIntensityd(keys=["x_0"], subtrahend=gt_suvr_zscore_mean, divisor=gt_suvr_zscore_std)
        
        transforms_list += [
            transforms.CopyItemsd(keys=[gt_suvr_path_column], names=['x_0']),
            transforms.LoadImaged(keys=["x_0"]),
            transforms.EnsureChannelFirstd(keys=["x_0"]),
            transforms.Spacingd(keys=['x_0'], pixdim=gt_suvr_voxel_spacing),
            transforms.DivisiblePadd(keys=['x_0'], k=vae_divisible_pad_k),
            normalization
        ]

    # Join the transforms.
    transforms_fn = transforms.Compose(transforms_list)

    trainset = CacheDataset(train_data, transforms_fn, cache_rate=1, num_workers=8) \
        if do_caching else Dataset(train_data, transforms_fn)

    validset = CacheDataset(valid_data, transforms_fn, cache_rate=1, num_workers=8) \
        if do_caching else Dataset(valid_data, transforms_fn)

    testset  = CacheDataset(test_data,  transforms_fn, cache_rate=1, num_workers=8) \
        if do_caching else Dataset(test_data, transforms_fn)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=False) \
        if len(trainset) > 0 else None
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False) \
        if len(validset) > 0 else None
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False) \
        if  len(testset) > 0 else None
    
    return train_loader, valid_loader, test_loader