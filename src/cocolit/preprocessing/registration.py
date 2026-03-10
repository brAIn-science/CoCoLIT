from typing import Union, List, Optional

import ants
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image



def align(
    moving: Union[Nifti1Image,Nifti2Image], 
    fixed:  Union[Nifti1Image,Nifti2Image],
    masks:  Optional[Union[Union[Nifti1Image,Nifti2Image], List[Union[Nifti1Image,Nifti2Image]]]] = None,
    random_seed=0
):
    """
    This function uses ANTsPy to align two MRIs.
    If masks are provided, the computed transform is also applied to them 
    using nearest neighbor interpolation.
    """
    mov_ants = nifti_to_ants(moving)
    fix_ants = nifti_to_ants(fixed)
    
    # Compute the registration and store the output dictionary
    registration_res = ants.registration(
        fixed=fix_ants, 
        moving=mov_ants, 
        type_of_transform='Affine', 
        random_seed=random_seed
    )
    
    warped_moving = to_nibabel(registration_res['warpedmovout'])
    
    # Return early if no masks are provided
    if masks is None:
        return warped_moving
        
    # Standardize masks into a list for uniform processing
    return_single = False
    if not isinstance(masks, list):
        masks = [masks]
        return_single = True
        
    warped_masks = []
    for mask in masks:
        mask_ants = nifti_to_ants(mask)
        # Apply the computed transform with nearest neighbor interpolation
        warped_mask_ants = ants.apply_transforms(
            fixed=fix_ants,
            moving=mask_ants,
            transformlist=registration_res['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        warped_masks.append(to_nibabel(warped_mask_ants))
        
    # Return registered image and registered mask(s)
    if return_single:
        return warped_moving, warped_masks[0]
    
    return warped_moving, warped_masks



def nifti_to_ants( nib_image ):
    """
    Fixed version of the nifti_to_ants function
    (ANTsPy uses the deprecated get_data call.)
    """
    ndim = nib_image.ndim
    if ndim < 3: raise Exception("Dimensionality is less than 3.")

    q_form      = nib_image.get_qform()
    spacing     = nib_image.header["pixdim"][1 : ndim + 1]
    origin      = np.zeros((ndim))
    origin[:3]  = q_form[:3, 3]
    direction   = np.diag(np.ones(ndim))
    direction[:3, :3] = q_form[:3, :3] / spacing[:3]

    return ants.from_numpy(
        data        = nib_image.get_fdata().astype(float),
        origin      = origin.tolist(),
        spacing     = spacing.tolist(),
        direction   = direction 
    )
    


def to_nibabel(img: "ants.core.ants_image.ANTsImage",header=None):
    """
    Source:
    https://github.com/ANTsX/ANTsPy/issues/693
    """
    affine = get_ras_affine(rotation=img.direction, spacing=img.spacing, origin=img.origin)
    return Nifti1Image(img.numpy(), affine, header)



def get_ras_affine(rotation, spacing, origin) -> np.ndarray:
    """
    Source:
    https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L357
    """
    rotation_zoom = rotation * spacing
    translation_ras = rotation.dot(origin)
    affine = np.eye(4)
    affine[:3, :3] = rotation_zoom
    affine[:3, 3] = translation_ras
    return affine