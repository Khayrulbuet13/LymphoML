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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# check LymphoMNIST virsion
import LymphoMNIST as info
print(f"LymphoMNIST v{info.__version__} @ {info.HOMEPAGE}")



# %%

from torch.utils.data import WeightedRandomSampler

def make_weights_for_balanced_classes(dataset, nclasses):
    # Count each class's occurrences in the dataset
    count = [0] * nclasses
    for _, label in dataset:
        count[label] += 1
    # Calculate the weight for each class based on their occurrences
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    # Assign weight to each sample based on its class
    weights = [weight_per_class[label] for _, label in dataset]
    return weights




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
        sampler=None, 
        num_workers=4,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    lengths = [int(len(val_ds) * frac) for frac in split]
    lengths[1] += len(val_ds) - sum(lengths)  # Correct split length sum
    val_ds, test_ds = random_split(val_ds, lengths)

    # # now we want to split the val_ds in validation and test
    # lengths = np.array(split) * len(val_ds)
    # lengths = lengths.astype(int)
    # left = len(val_ds) - lengths.sum()
    # # we need to add the different due to float approx to int
    # lengths[-1] += left

    # val_ds, test_ds = random_split(val_ds, lengths.tolist())
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    shuffle = False if sampler else True

    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, *args, **kwargs)
    # val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    # test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, sampler=sampler,  *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl



class FilteredLymphoMNIST(Dataset):
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        self.labels_to_keep = labels_to_keep
        self.label_map = {label: i for i, label in enumerate(labels_to_keep)}  # Map original labels to new labels
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        if index >= len(self.indices):
            raise IndexError("Index out of range")
        original_index = self.indices[index]
        image, label = self.original_dataset[original_index]
        # Remap the label
        return image, self.label_map[label.item()]

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
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])),
            iaa.Sometimes(0.5, iaa.Sequential([
                iaa.Crop(percent=(0.1, 0.2))
            ])),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.8, iaa.Affine(
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
    'lr': 1e-6,
    'batch_size': 16,
    'epochs': 10000,
    'model': "teacher_final-balanced-transfer-frozen"
}


# Initialize dataset
original_train_ds = LymphoMNIST(root='./dataset', train=True, download=True, transform=train_transform, num_classes=3)
original_test_ds = LymphoMNIST(root='./dataset', train=False, download=True, transform=val_transform, num_classes=3)


# Specify labels to keep
labels_to_keep = [0, 2] # 0: B, 1: T4, 2: T8

# Initialize filtered dataset with labels to keep
train_ds = FilteredLymphoMNIST(original_train_ds, labels_to_keep)
test_ds= FilteredLymphoMNIST(original_test_ds, labels_to_keep)





# Using the function with your dataset
weights = make_weights_for_balanced_classes(train_ds, 2)
sampler = WeightedRandomSampler(weights, len(weights))



# train_dl = DataLoader(train_ds, batch_size=params['batch_size'], sampler=sampler, *args, **kwargs)


train_dl, val_dl, test_dl = get_dataloaders(train_ds,
                                            test_ds,
                                            split=(0.5, 0.5),
                                            batch_size=params['batch_size'],
                                            sampler=sampler,
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

# Freeze all layers except the fully connected layer
for param in resnet50.parameters():
    param.requires_grad = False

# Only the parameters of the final layer are set to require gradients
for param in resnet50.fc.parameters():
    param.requires_grad = True

# Move the modified model to CUDA
cnn = resnet50.to(device)

#load saved weight
cnn.load_state_dict(torch.load('checkpoint/final_version/teacher_97_percent.pt'))

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


# Save the final model weights after training with Poutyne
model_weights_path = os.path.join(project.checkpoint_dir, f"{datetime.datetime.now().strftime('%d %B %H:%M')}_{model_name}_final_weights.pt")
model.save_weights(model_weights_path)
logging.info(f"Model weights saved to {model_weights_path}")


from sklearn.metrics import confusion_matrix
import numpy as np



def calculate_confusion_matrix(model, dataloader, device):
    # Ensure the model is in evaluation mode
    # model.model.eval()
    
    all_preds = []
    all_targets = []
    
    # Iterate over the DataLoader for predictions
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        outputs = model.predict_on_batch(inputs)
        
        _, preds = torch.max(torch.tensor(outputs), dim=1)
        all_preds.extend(preds.numpy())
        all_targets.extend(targets.numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    return cm

# Calculate the confusion matrices for the train and test sets
train_cm = calculate_confusion_matrix(model, train_dl, device)
test_cm = calculate_confusion_matrix(model, test_dl, device)

# Log confusion matrices to Comet ML
experiment.log_confusion_matrix(matrix=train_cm, file_name='train_confusion_matrix.json')
experiment.log_confusion_matrix(matrix=test_cm, file_name='test_confusion_matrix.json')


# get the results on the test set
loss, test_acc = model.evaluate_generator(test_dl)
logging.info(f'test_acc=({test_acc})')
experiment.log_metric('test_acc', test_acc)





# Don't forget to end the experiment
experiment.end()



# %%
