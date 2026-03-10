"""
Code is borrowed from and adapted from:
https://github.com/nipreps/synthstrip
"""
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import scipy
import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from nibabel.processing import resample_from_to
from huggingface_hub import hf_hub_download


class SkullStripping:
    """
    Skull Stripping procedure using the SynthStrip model.
    """

    def __init__(
        self, 
        device: str,
        border: int = 1,
        checkpoint: Optional[str] = None,
    ):
        self.border = border
        self.checkpoint = checkpoint
        self.device = device

        if  checkpoint is None:
            checkpoint = hf_hub_download(repo_id='lemuelpuglisi/generics', filename='synthstrip.1.pt')
        
        # Load the synthstrip model.
        self.model = StripModel().to(self.device)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.eval()
        

    @torch.inference_mode()
    def run(self, scan: Union[Nifti1Image,Nifti2Image]) -> Tuple:
        """
        Run skull stripping.
        borrowed from: https://github.com/nipreps/synthstrip
        """
        # avoid overriding the original scan
        scan = nib.nifti1.Nifti1Image(
            scan.get_fdata().copy(),
            scan.affine.copy(),
            scan.header
        )

        conformed = self._conform(scan)
        in_data = conformed.get_fdata(dtype='float32')
        in_data -= in_data.min()
        in_data = np.clip(in_data / np.percentile(in_data, 99), 0, 1)
        in_data = in_data[np.newaxis, np.newaxis]

        # predict the surface distance transform
        input_tensor = torch.from_numpy(in_data).to(self.device)
        sdt = self.model(input_tensor).cpu().numpy().squeeze()

        # move to the original space
        sdt_nii = nib.Nifti1Image(sdt, conformed.affine, None)
        sdt_nii = resample_from_to(sdt_nii, scan, cval=100)

        # extract the actual mask
        sdt_data = (sdt_nii.get_fdata() < self.border).astype(np.uint16)
        components = scipy.ndimage.label(sdt_data)[0]
        bincount = np.bincount(components.flatten())[1:]
        mask = components == (np.argmax(bincount) + 1)
        mask = scipy.ndimage.morphology.binary_fill_holes(mask)

        # save the skull-stripped scan
        img_data = scan.get_fdata()
        bg = np.min([0, img_data.min()])
        img_data[mask == 0] = bg
        output_scan = nib.Nifti1Image(img_data, scan.affine, scan.header)
            
        # save the brain mask
        hdr = scan.header.copy()
        hdr.set_data_dtype('uint8')
        output_mask = nib.Nifti1Image(mask, scan.affine, hdr)

        return output_scan, output_mask


    def _conform(self, input_nii):
        """
        Resample image as SynthStrip likes it.
        borrowed from: https://github.com/nipreps/synthstrip
        """
        shape = np.array(input_nii.shape[:3])
        affine = input_nii.affine

        # Get corner voxel centers in index coords
        corner_centers_ijk = (
            np.array(
                [
                    (i, j, k)
                    for k in (0, shape[2] - 1)
                    for j in (0, shape[1] - 1)
                    for i in (0, shape[0] - 1)
                ]
            )
            + 0.5
        )

        # Get corner voxel centers in mm
        corners_xyz = affine @ np.hstack((corner_centers_ijk, np.ones((len(corner_centers_ijk), 1)))).T

        # Target affine is 1mm voxels in LIA orientation
        target_affine = np.diag([-1.0, 1.0, -1.0, 1.0])[:, (0, 2, 1, 3)]

        # Target shape
        extent = corners_xyz.min(1)[:3], corners_xyz.max(1)[:3]
        target_shape = ((extent[1] - extent[0]) / 1.0 + 0.999).astype(int)

        # SynthStrip likes dimensions be multiple of 64 (192, 256, or 320)
        target_shape = np.clip(np.ceil(np.array(target_shape) / 64).astype(int) * 64, 192, 320)

        # Ensure shape ordering is LIA too
        target_shape[2], target_shape[1] = target_shape[1:3]

        # Coordinates of center voxel do not change
        input_c = affine @ np.hstack((0.5 * (shape - 1), 1.0))
        target_c = target_affine @ np.hstack((0.5 * (target_shape - 1), 1.0))

        # Rebase the origin of the new, plumb affine
        target_affine[:3, 3] -= target_c[:3] - input_c[:3]

        # Create the target image (reference space)
        reference = nib.Nifti1Image(np.zeros(target_shape), target_affine)
        
        # Resample input image into reference space
        nii = resample_from_to(input_nii, reference, order=1)  # order=1 = trilinear
        return nii



class StripModel(nn.Module):
    """
    borrowed from: https://github.com/nipreps/synthstrip
    same class used to train the original SynthStrip model.
    """
    def __init__(
        self,
        nb_features=16,
        nb_levels=7,
        feat_mult=2,
        max_features=64,
        nb_conv_per_level=2,
        max_pool=2,
        return_mask=False,
    ):
        super().__init__()

        # dimensionality
        ndims = 3

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for nf in final_convs:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activation: {activation}')

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out