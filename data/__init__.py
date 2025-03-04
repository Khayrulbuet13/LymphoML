from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .transformation import train_transform, val_transform
from .dataset import FilteredLymphoMNIST, get_dataloaders, balanced_weights


def get_image_folder_dataloaders(train_dir, val_dir, val_transform=val_transform, train_transform=train_transform, batch_size=64, *args, **kwargs):
    """
    Create dataloaders for image folder datasets.
    """
    train_ds = ImageFolder(train_dir, transform=train_transform)
    val_ds = ImageFolder(val_dir, transform=val_transform)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    
    return train_dl, val_dl, test_dl
