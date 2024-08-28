import datetime
# from arrow import get
from comet_ml import Experiment
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from glob import glob
from PIL import Image
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler
import torch
from tqdm import tqdm
import torchvision.transforms as T

from LymphoMNIST.LymphoMNIST import LymphoMNIST
from torch.utils.data import DataLoader, Dataset, random_split
import os
from imgaug import augmenters as iaa

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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


# CONSTANTS
EPOCHS = 100000
TEMPERATURE = 2
INIT_LR = 0.001
WEIGHT_DECAY = .0001
CLIP_THRESHOLD = 1.0
ALPHA = 1
BATCH_SIZE = 64
RESIZE = 64
BIGGER = 64
MODEL = 'KD-timm-good-teacher'
# both teacher and student is devided by temperature



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
experiment = Experiment(
        api_key="2iwTpjYhUb3dGr4yIiVtt1oRA",
        project_name="tvsb-ablation",
        # project_name="",
        workspace="khayrulbuet13")

experiment.set_name(MODEL)




transform_train = T.Compose([T.Resize((BIGGER, BIGGER)),
                                      ImgAugTransform(),
                                      T.ToTensor(),
                                      T.Normalize([0.4819], [0.1484]),
                                      ConvertToRGB()])


transform_val = T.Compose([T.Resize((BIGGER, BIGGER)),
                                      T.ToTensor(),
                                      T.Normalize([0.4819], [0.1484]),
                                      ConvertToRGB()])


class FilteredLymphoMNIST(Dataset):
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        return self.original_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

def get_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE, **kwargs):
    # Split val_ds to validation and test sets evenly
    val_size = len(val_ds) // 2
    test_size = len(val_ds) - val_size
    val_ds, test_ds = random_split(val_ds, [val_size, test_size])
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs),
    )

# Load datasets
labels_to_keep = [0, 1]  # Specify labels to keep
train_ds = FilteredLymphoMNIST(LymphoMNIST(root='./dataset', train=True, download=True, transform=transform_train, num_classes=3), labels_to_keep)
val_test_ds = FilteredLymphoMNIST(LymphoMNIST(root='./dataset', train=False, download=True, transform=transform_val, num_classes=3), labels_to_keep)

# Initialize dataloaders
train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_test_ds, num_workers=4)



def mixup(image):
    alpha = torch.rand(1)
    mixedup_images = (alpha * image +
                      (1 - alpha) * torch.flip(image, dims=[0]))
    return mixedup_images




def evaluate(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Compute the mixup data. Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.uniform(0, 1)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


# def mixup_criterion(criterion, pred, y_a, y_b, lam, num_classes):
#     y_a_one_hot = F.one_hot(y_a, num_classes=num_classes).float()
#     y_b_one_hot = F.one_hot(y_b, num_classes=num_classes).float()
#     return lam * criterion(pred, y_a_one_hot) + (1 - lam) * criterion(pred, y_b_one_hot)



def mixup_criterion(criterion, pred, teacher_pred, y_a, y_b, lam, temperature):
    # Softening the probabilities with temperature for both predictions
    pred_soft = F.log_softmax(pred / temperature, dim=1)
    teacher_pred_soft = F.softmax(teacher_pred / temperature, dim=1)

    # # Creating one-hot encoded labels
    # y_a_one_hot = F.one_hot(y_a, num_classes=pred.size(1)).float().to(pred.device)
    # y_b_one_hot = F.one_hot(y_b, num_classes=pred.size(1)).float().to(pred.device)

    # # Mixing the one-hot labels according to lambda
    # mixed_labels = lam * y_a_one_hot + (1 - lam) * y_b_one_hot

    # Calculating the distillation loss as KL divergence
    distillation_loss = criterion(pred_soft, teacher_pred_soft)


    # Combining the losses
    return distillation_loss


class WarmUpCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, warmup_learning_rate, learning_rate_base, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.learning_rate_base = learning_rate_base
        super(WarmUpCosine, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps * self.last_epoch + self.warmup_learning_rate
        else:
            # Cosine annealing
            lr = self.learning_rate_base * 0.5 * (1 + np.cos(np.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        
        # After total_steps, reduce to zero
        if self.last_epoch > self.total_steps:
            lr = 0.0
        
        return [lr for _ in self.base_lrs]



# Load a pre-trained ResNet50 model
resnet50 = models.resnet50(weights='IMAGENET1K_V1')

# Adjust the final fully connected layer for your number of classes
num_ftrs = resnet50.fc.in_features
num_classes = 2  # Change to your number of classes
resnet50.fc = nn.Linear(num_ftrs, num_classes)

# Move the modified model to CUDA
teacher_model = resnet50.to(device)



# Set all parameters to be trainable
for param in teacher_model.parameters():
    param.requires_grad = True

# teacher_model.fc = nn.Sequential(nn.Linear(teacher_model.fc.in_features, 2))
teacher_model.load_state_dict(torch.load(
    'checkpoint/final_version/teacher_97_percent.pt', map_location=device))

teacher_model = teacher_model.to(device)
teacher_model.eval()
student_model = models.resnet18(num_classes=2).to(device)


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def forward(self, x):
        teacher_output = self.teacher(x)
        student_output = self.student(x)

        return student_output, teacher_output



# Create an instance of the Distiller class
distiller = Distiller(student=student_model, teacher=teacher_model)


# Define loss function and optimizer
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.AdamW(student_model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)



# optimizer = optim.SGD(student_model.parameters(), lr=INIT_LR,
#                       momentum=0.9, weight_decay=WEIGHT_DECAY)
checkpoint_path = str("/checkpoint" +
                      f"{datetime.datetime.now().strftime('%d %B %H:%M')}-student_model.pt")

best_val_accuracy = 0.0
early_stopping_patience = 1000
epochs_no_improve = 0
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10, verbose=True)

# scheduler = WarmUpCosine(
#     optimizer,
#     warmup_steps=5000,
#     total_steps=EPOCHS,
#     warmup_learning_rate=0.0001,
#     learning_rate_base=INIT_LR
# )

from timm.scheduler.cosine_lr import CosineLRScheduler
scheduler = CosineLRScheduler(optimizer, t_initial=20, lr_min=2e-8,
                  cycle_mul=2.0, cycle_decay=.5, cycle_limit=5,
                  warmup_t=10, warmup_lr_init=1e-6, warmup_prefix=False, t_in_epochs=True,
                  noise_range_t=None, noise_pct=0.67, noise_std=1.0,
                  noise_seed=42, k_decay=1.0, initialize=True)

# Training loop
for epoch in range(EPOCHS):
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    student_model.train()  # Set the model to training mode
    train_loader_progress = tqdm(train_loader, total=len(train_loader),
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    for inputs, labels in train_loader_progress:
        inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
        optimizer.zero_grad()

        # Apply mixup
        mixed_inputs, targets_a, targets_b, lam = mixup_data(
            inputs, labels, alpha=ALPHA, device=device)

        # Forward pass through the distiller with mixed inputs
        student_output, teacher_output = distiller(mixed_inputs)
        teacher_output = teacher_output.detach()  # Detach teacher's output

        # # Convert the student's output to log probabilities
        # student_output_log_prob = F.log_softmax(
        #     student_output / TEMPERATURE, dim=1)
        
        # # and probabilities respectively using temperature scaling
        # student_output_log_prob = F.log_softmax(student_output / TEMPERATURE, dim=1)
        # teacher_output_prob = F.softmax(teacher_output / TEMPERATURE, dim=1)

        # loss = criterion(student_output_log_prob, teacher_output_prob)



        # Adjust the loss function for mixup
        num_classes = 2
        loss = mixup_criterion(criterion, student_output, teacher_output, targets_a, targets_b, lam, TEMPERATURE)


        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(student_output.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Backward and optimize
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            student_model.parameters(), CLIP_THRESHOLD)
        optimizer.step()

        # Update progress bar
        train_loader_progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS}")
        train_loader_progress.set_postfix(
            loss=loss.item(), acc=100. * train_correct / train_total)

    train_accuracy = 100 * train_correct / train_total
    train_loss /= train_total

    # Evaluate on validation data
    val_accuracy = evaluate(student_model, val_loader)

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_accuracy)


    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        print(f'Validation accuracy increased ({best_val_accuracy:.2f}%), LR: {current_lr}')
        torch.save(student_model.state_dict(), f"./checkpoint/KD_{MODEL}_{datetime.datetime.now().strftime('%d %B %H:%M')}.pt")
        
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs. No improvement in validation accuracy.')
            break
        

    # Log metrics to Comet ML
    experiment.log_metric("train_loss", train_loss, step=epoch)
    experiment.log_metric("train_accuracy", train_accuracy, step=epoch)
    experiment.log_metric("val_accuracy", val_accuracy, step=epoch)
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}'
          f', Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

# Evaluate on test data
test_accuracy = evaluate(student_model, test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Log test accuracy to Comet.ml
experiment.log_metric("test_accuracy", test_accuracy, step=epoch)


experiment.end()


