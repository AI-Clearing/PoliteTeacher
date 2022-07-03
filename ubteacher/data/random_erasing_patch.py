


import numpy as np
import torch

import torchvision.transforms as transforms
from torchvision.transforms import functional as F



def concat_patches(patches, image_size):
    erased_mask = np.zeros(image_size).astype(np.bool8)
    for patch in patches:
        try:
            i, j, h, w = patch
            erased_mask[..., i : i + h, j : j + w] = True
        except TypeError:
            pass

    return erased_mask

class RandomErasingPatch(transforms.RandomErasing):
    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
            (x,y,h,w):    Which fragment was erased.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [
                    self.value,
                ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
       
            
            return F.erase(img, x, y, h, w, v, self.inplace), (x, y, h, w)
        else:
            return img, None

class ComposePatch(transforms.Compose):
    def __call__(self, img):
        patches = []
        for t in self.transforms:
            if isinstance(t, RandomErasingPatch):
                img, patch = t(img)
                patches.append(patch)
            else:
                img = t(img)
        return img, patches


