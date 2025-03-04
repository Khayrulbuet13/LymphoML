import numpy as np
import torch
import torchvision.transforms as T
from imgaug import augmenters as iaa
from PIL import Image

class ConvertToRGB:
    """Convert 1-channel tensors to 3-channel tensors by duplicating the channel 3 times."""
    def __call__(self, tensor):
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

class ImgAugTransform:
    """
    Wrapper to allow imgaug to work with Pytorch transformation pipeline, modified for 1-channel images.
    """
    def __init__(self, rgb=True):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.8, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True) if rgb else iaa.Identity()
        ])
        self.rgb = rgb

    def __call__(self, img):
        # Convert 1-channel images to 3-channel images for augmentation if needed
        if img.mode == 'L' and self.rgb:
            img = img.convert("RGB")
        img = np.array(img)
        img = self.aug.augment_image(img)
        # Convert back to 1-channel image if originally it was
        if not self.rgb:
            img = Image.fromarray(img).convert('L')
        else:
            img = Image.fromarray(img)
        return img

def get_teacher_transforms(im_size=120):
    """Get transforms for teacher model."""
    # Use ImgAugTransform for data augmentation
    aug_transform = ImgAugTransform(rgb=True)
    
    train_transform = T.Compose([
        T.Resize((im_size, im_size)),
        aug_transform,
        T.ToTensor(),
        T.Normalize([0.4819], [0.1484]),
        ConvertToRGB()
    ])
    
    val_transform = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize([0.4819], [0.1484]),
        ConvertToRGB()
    ])
    
    return train_transform, val_transform

def get_student_transforms(im_size=64):
    """
    Get transforms for student models.
    
    Args:
        im_size: Image size for the student model (default: 64)
        
    Returns:
        (train_transform, val_transform)
    """
    train_transform = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize([0.4819], [0.1484]),
    ])
    
    val_transform = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor(),
        T.Normalize([0.4819], [0.1484]),
    ])
    
    return train_transform, val_transform
