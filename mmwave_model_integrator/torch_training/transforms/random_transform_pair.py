import torch
from torchvision import transforms
from torchvision.transforms import functional as F

class RandomTransformPair:
    def __init__(self, transform_cls, *args, **kwargs):
        super().__init__()
        self.transform = transform_cls(*args, **kwargs)

    @staticmethod
    def _ensure_channel_dim(tensor):
        """
        Ensures tensor has a channel dimension (C,H,W).
        If tensor is 2D, adds a channel dimension and returns a flag to squeeze later.
        """
        squeeze_after = False
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 2:  # (H, W)
                tensor = tensor.unsqueeze(0)
                squeeze_after = True
            elif tensor.dim() == 3:
                squeeze_after = False
        # For PIL images, we don’t need to do anything
        return tensor, squeeze_after

    def __call__(self, img, mask):
        t = self.transform

        # Ensure correct dimensions
        mask, squeeze_mask = self._ensure_channel_dim(mask)
        img, squeeze_img = self._ensure_channel_dim(img)

        if isinstance(t, transforms.RandomRotation):
            angle = t.get_params(t.degrees)
            img = F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)

        elif isinstance(t, transforms.RandomCrop):
            i, j, h, w = t.get_params(img, t.size)
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

        elif isinstance(t, transforms.RandomResizedCrop):
            i, j, h, w = t.get_params(img, t.scale, t.ratio)
            img = F.resized_crop(img, i, j, h, w, t.size, interpolation=F.InterpolationMode.BILINEAR)
            mask = F.resized_crop(mask, i, j, h, w, t.size, interpolation=F.InterpolationMode.NEAREST)

        elif isinstance(t, transforms.RandomHorizontalFlip):
            if torch.rand(1) < t.p:
                img = F.hflip(img)
                mask = F.hflip(mask)

        elif isinstance(t, transforms.RandomVerticalFlip):
            if torch.rand(1) < t.p:
                img = F.vflip(img)
                mask = F.vflip(mask)

        else:
            img = t(img)
            mask = t(mask)

        # Squeeze back any dimensions added
        if squeeze_mask:
            mask = mask.squeeze(0)
        if squeeze_img:
            img = img.squeeze(0)

        return img, mask

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self.transform})"


# Concrete subclasses
class RandomRotationPair(RandomTransformPair):
    def __init__(self, degrees):
        super().__init__(transforms.RandomRotation, degrees)

class RandomCropPair(RandomTransformPair):
    def __init__(self, size):
        super().__init__(transforms.RandomCrop, size)

class RandomResizedCropPair(RandomTransformPair):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        super().__init__(transforms.RandomResizedCrop, size, scale=scale, ratio=ratio)

class RandomHorizontalFlipPair(RandomTransformPair):
    def __init__(self, p=0.5):
        super().__init__(transforms.RandomHorizontalFlip, p=p)

class RandomVerticalFlipPair(RandomTransformPair):
    def __init__(self, p=0.5):
        super().__init__(transforms.RandomVerticalFlip, p=p)

class ComposePair:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
