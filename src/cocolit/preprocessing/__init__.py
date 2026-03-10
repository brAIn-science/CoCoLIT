import torch
from monai.data import MetaTensor
from monai.transforms import Transform


from .registration import align
from .skull_stripping import SkullStripping
from .normalization import whitestripe_normalize


class NibToMetaTensor(Transform):
    def __call__(self, img):
        data = img.get_fdata(dtype="float32")
        return MetaTensor(
            torch.as_tensor(data),
            affine=torch.as_tensor(img.affine),
            meta={
                "nib_header": img.header,
                "filename_or_obj": img.get_filename(),
                "original_channel_dim": "no_channel",
            },
        )