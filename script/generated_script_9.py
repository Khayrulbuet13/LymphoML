# %%

# Lets import the packages
import torch
from torchvision import transforms
from LymphoMNIST.LymphoMNIST import LymphoMNIST

import torch.nn as nn
from torchvision import models
from torchsummary import summary


import time
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import Project
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback
from logger import logging
import datetime, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# check LymphoMNIST virsion
import LymphoMNIST as info
print(f"LymphoMNIST v{info.__version__} @ {info.HOMEPAGE}")

# %%
from utils import device

import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


def get_dataloaders(
        train_ds,
        val_ds,
        split=(0.5, 0.5),
        batch_size=64,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """

    # now we want to split the val_ds in validation and test
    lengths = np.array(split) * len(val_ds)
    lengths = lengths.astype(int)
    left = len(val_ds) - lengths.sum()
    # we need to add the different due to float approx to int
    lengths[-1] += left

    val_ds, test_ds = random_split(val_ds, lengths.tolist())
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl


class FilteredLymphoMNIST(Dataset):
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        # Filter indices based on labels_to_keep
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        # Map the current index to the index of the original dataset
        original_index = self.indices[index]
        return self.original_dataset[original_index]

    def __len__(self):
        return len(self.indices)


class ConvertToRGB:
    """
    Convert 1-channel tensors to 3-channel tensors by duplicating the channel 3 times.
    """
    def __call__(self, tensor):
        # Check if the tensor is 1-channel (C, H, W) where C == 1
        if tensor.shape[0] == 1:
            # Duplicate the channel 3 times
            tensor = tensor.repeat(3, 1, 1)
        return tensor



class ImgAugTransform:
    """
    Wrapper to allow imgaug to work with Pytorch transformation pipeline, modified for 1-channel images.
    """
    def __init__(self):
        
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3))),
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])),
            iaa.Sometimes(0.5, iaa.Sequential([
                iaa.Crop(percent=(0.1, 0.2))
            ])),
            iaa.LinearContrast((.50, 1.75)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.8,
                          iaa.Affine(
                              scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                              translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                              rotate=(-25, 25),
                              shear=(-8, 8)
                          )),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        # Convert 1-channel images to 3-channel images for augmentation
        if img.mode == 'L':
            img = img.convert("RGB")
        img = np.array(img)
        img = self.aug.augment_image(img)
        # Convert back to 1-channel image if originally it was
        img = Image.fromarray(img).convert('L')
        return img

im_size = 64

val_transform = T.Compose([
    T.Resize((im_size, im_size)),
    T.ToTensor(),
    T.Normalize([0.4819], [0.1484]),  # Adjusted for 1-channel
    ConvertToRGB()
])

train_transform = T.Compose([
    T.Resize((im_size, im_size)),
    ImgAugTransform(),
    T.ToTensor(),
    T.Normalize([0.4819], [0.1484]),  # Adjusted for 1-channel
    ConvertToRGB()
])


#%%
# our hyperparameters
params = {
    'lr': 5e-5,
    'batch_size': 16,
    'epochs': 1000,
    'model': "LCE-1.75"
}


# Initialize dataset
original_train_ds = LymphoMNIST(root='./dataset', train=True, download=True, transform=train_transform, num_classes=3)
original_test_ds = LymphoMNIST(root='./dataset', train=False, download=True, transform=val_transform, num_classes=3)


# Specify labels to keep
labels_to_keep = [0, 1] # 0: B, 1: T4, 2: T8

# Initialize filtered dataset with labels to keep
train_ds = FilteredLymphoMNIST(original_train_ds, labels_to_keep)
test_ds= FilteredLymphoMNIST(original_test_ds, labels_to_keep)


train_dl, val_dl, test_dl = get_dataloaders(train_ds,
                                            test_ds,
                                            split=(0.5, 0.5),
                                            batch_size=params['batch_size'],
                                            num_workers=4
                                           )

# print one image shape
for x, y in train_dl:
    print("train",x.shape)
    break
for x, y in val_dl:
    print("val",x.shape)
    break

# %%


# Load a pre-trained ResNet50 model
resnet50 = models.resnet50(weights='IMAGENET1K_V1')

# Adjust the final fully connected layer for your number of classes
num_ftrs = resnet50.fc.in_features
num_classes = 2  # Change to your number of classes
resnet50.fc = nn.Linear(num_ftrs, num_classes)

# Move the modified model to CUDA
cnn = resnet50.to(device)

# Adjust the input size if necessary
summary(resnet50, (3, 64, 64))


# %%

project = Project()

logging.info(f'Using device={device} 🚀')

model_name = params['model']

# define our comet experiment
experiment = Experiment(
    api_key="2iwTpjYhUb3dGr4yIiVtt1oRA",
    project_name="TvsB-ablation",
    # project_name="",
    workspace="khayrulbuet13")

experiment.log_parameters(params)
experiment.set_name(model_name)




# define custom optimizer and instantiace the trainer `Model`
optimizer = optim.Adam(cnn.parameters(), lr=params['lr'])
model = Model(cnn, optimizer, "cross_entropy",
                batch_metrics=["accuracy"]).to(device)
# usually you want to reduce the lr on plateau and store the best model
callbacks = [
    ReduceLROnPlateau(monitor="val_acc", patience=100, verbose=True),
    ModelCheckpoint(str(project.checkpoint_dir /
                        f"""{datetime.datetime.now().strftime('%d %B %H:%M')}-model{model_name}.pt"""), save_best_only="True", verbose=True),
    EarlyStopping(monitor="val_acc", patience=100, mode='max'),
    CometCallback(experiment)
]

model.fit_generator(
    train_dl,
    val_dl,
    epochs=params['epochs'],
    callbacks=callbacks,
)
# get the results on the test set
loss, test_acc = model.evaluate_generator(test_dl)
logging.info(f'test_acc=({test_acc})')
experiment.log_metric('test_acc', test_acc)

