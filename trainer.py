import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchsummary import summary
from tqdm import tqdm
from comet_ml import Experiment
from timm.scheduler.cosine_lr import CosineLRScheduler

from config import TeacherConfig, Student2Config, Student1Config
from models import TeacherModel, Student2, Student1, Distiller
from models.distiller import mixup_data, mixup_criterion, distillation_loss
from data import FilteredLymphoMNIST, get_dataloaders, balanced_weights
from data.transformation import (
    get_teacher_transforms, 
    get_student_transforms
)
from LymphoMNIST.LymphoMNIST import LymphoMNIST
from logger import logging

class BaseTrainer:
    """Base trainer class for all models."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility exactly as in teacher.py
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        import numpy as np
        import random
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Initialize Comet experiment
        self.experiment = Experiment(
            api_key=config.comet_api_key,
            project_name=config.comet_project_name,
            workspace=config.comet_workspace
        )
        self.experiment.log_parameters(config.__dict__)
        # Use model_name from config if available, otherwise use model_type
        model_name = getattr(config, 'model_name', None)
        self.experiment.set_name(model_name if model_name else f"{config.model_type}")
        
        # Initialize dataloaders and model
        self.setup_data()
        self.setup_model()
        
        # Initialize optimizer and criterion
        self.setup_training()
        
    def setup_data(self):
        """Setup data loaders. To be implemented by subclasses."""
        raise NotImplementedError
        
    def setup_model(self):
        """Setup model. To be implemented by subclasses."""
        raise NotImplementedError
        
    def setup_training(self):
        """Setup optimizer and criterion. To be implemented by subclasses."""
        raise NotImplementedError
        
    def train(self):
        """Training loop. To be implemented by subclasses."""
        raise NotImplementedError
        
    def evaluate(self, data_loader):
        """Evaluate model on the given data loader."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy
    
    def save_checkpoint(self, path=None):
        """Save model checkpoint."""
        if path is None:
            timestamp = datetime.datetime.now().strftime('%d_%B_%H_%M')
            path = f"{self.config.checkpoint_dir}/{self.config.model_type}_{timestamp}.pt"
        
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")
        return path

class TeacherTrainer(BaseTrainer):
    """Trainer for the teacher model."""
    def setup_data(self):
        # Get transforms for teacher
        train_transform, val_transform = get_teacher_transforms(
            im_size=self.config.image_size
        )
        
        # Load datasets
        original_train_ds = LymphoMNIST(
            root='./dataset', 
            train=True, 
            download=True, 
            transform=train_transform, 
            num_classes=3
        )
        original_val_ds = LymphoMNIST(
            root='./dataset', 
            train=False, 
            download=True, 
            transform=val_transform, 
            num_classes=3
        )
        
        # Filter datasets
        train_ds = FilteredLymphoMNIST(original_train_ds, self.config.labels_to_keep)
        val_ds = FilteredLymphoMNIST(original_val_ds, self.config.labels_to_keep)
        
        # Create sampler if needed
        sampler = None
        if self.config.use_weighted_sampler:
            weights = balanced_weights(train_ds, len(self.config.labels_to_keep))
            sampler = WeightedRandomSampler(weights, len(weights))
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            train_ds, 
            val_ds, 
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=4
        )
    
    def setup_model(self):
        # Initialize teacher model
        self.model = TeacherModel(num_classes=self.config.num_classes).to(self.device)
        summary(self.model, (3, self.config.image_size, self.config.image_size))
    
    def setup_training(self):
        # Initialize optimizer and criterion exactly as in teacher.py
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=self.config.scheduler_patience, 
            verbose=True
        )
    
    def train(self):
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
            
            train_loss = running_loss / total
            train_acc = running_corrects.double() / total
            
            # Evaluate on validation set
            val_acc = self.evaluate(self.val_loader)
            
            # Log metrics
            self.experiment.log_metrics({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc.item(),
                'val_acc': val_acc
            }, step=epoch)
            
            print(f'Epoch {epoch+1}/{self.config.epochs} - '
                  f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - '
                  f'Val acc: {val_acc:.4f}')
            
            # Step the scheduler
            self.scheduler.step(val_acc)
            
            # Check for early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                self.save_checkpoint()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stopping_patience:
                    print('Early stopping!')
                    break
        
        # Evaluate on test set
        test_acc = self.evaluate(self.test_loader)
        self.experiment.log_metric('test_acc', test_acc)
        print(f'Test accuracy: {test_acc:.4f}')
        
        # End experiment
        self.experiment.end()

# class Student2Trainer(BaseTrainer):
#     """Trainer for the quantized student model."""
#     def setup_data(self):
#         # Get transforms
#         train_transform, val_transform = get_student_transforms(self.config.student_image_size)
#         self.student_im_size = self.config.student_image_size
        
#         # Load datasets
#         train_ds = FilteredLymphoMNIST(
#             LymphoMNIST(root='./dataset', train=True, download=True, transform=train_transform, num_classes=3), 
#             self.config.labels_to_keep
#         )
#         val_test_ds = FilteredLymphoMNIST(
#             LymphoMNIST(root='./dataset', train=False, download=True, transform=val_transform, num_classes=3), 
#             self.config.labels_to_keep
#         )
        
#         # Create dataloaders
#         self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
#             train_ds, 
#             val_test_ds, 
#             batch_size=self.config.batch_size,
#             num_workers=4
#         )
    
#     def setup_model(self):
#         # Initialize teacher model
#         teacher_model = TeacherModel(num_classes=self.config.num_classes).to(self.device)
#         teacher_model.load_state_dict(torch.load(self.config.teacher_checkpoint, map_location=self.device))
#         teacher_model.eval()
        
#         # Initialize student model
#         self.student_model = Student2(
#             num_classes=self.config.num_classes, 
#             input_size=(1, self.student_im_size, self.student_im_size)
#         ).to(self.device)
        
#         # Create distiller
#         self.model = Distiller(self.student_model, teacher_model)
        
#         # Print model summary
#         summary(self.student_model, (1, self.student_im_size, self.student_im_size))
    
#     def setup_training(self):
#         # Initialize optimizer and criterion
#         self.optimizer = optim.AdamW(
#             self.student_model.parameters(), 
#             lr=self.config.learning_rate,
#             weight_decay=self.config.weight_decay
#         )
#         self.criterion = nn.KLDivLoss(reduction='batchmean')
    
#     def train(self):
#         best_val_accuracy = 0.0
#         epochs_no_improve = 0
#         checkpoint_path = f"{self.config.checkpoint_dir}/KD_{datetime.datetime.now().strftime('%d_%B_%H_%M')}_{self.config.model_type}.pt"
        
#         for epoch in range(self.config.epochs):
#             train_loss = 0.0
#             train_total = 0
#             train_correct = 0
#             self.student_model.train()
            
#             for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"):
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 self.optimizer.zero_grad()
                
#                 # Extract first channel and resize for student model
#                 images_student = images[:, 0:1, :, :]
#                 images_student = F.interpolate(
#                     images_student, 
#                     size=(self.student_im_size, self.student_im_size), 
#                     mode='bilinear', 
#                     align_corners=False
#                 )
                
#                 # Apply mixup if enabled
#                 if self.config.use_mixup:
#                     mixed_inputs_student, targets_a, targets_b, lam = mixup_data(
#                         images_student, 
#                         labels, 
#                         self.config.alpha, 
#                         self.device
#                     )
#                     student_outputs, teacher_outputs = self.model(images, mixed_inputs_student)
                    
#                     # Compute loss
#                     student_output_log_prob = F.log_softmax(student_outputs / self.config.temperature, dim=1)
#                     teacher_output_soft = F.softmax(teacher_outputs / self.config.temperature, dim=1).detach()
#                     loss = mixup_criterion(
#                         self.criterion, 
#                         student_output_log_prob, 
#                         targets_a, 
#                         targets_b, 
#                         lam, 
#                         self.config.num_classes
#                     )
#                 else:
#                     student_outputs, teacher_outputs = self.model(images, images_student)
                    
#                     # Compute loss
#                     student_output_log_prob = F.log_softmax(student_outputs / self.config.temperature, dim=1)
#                     teacher_output_soft = F.softmax(teacher_outputs / self.config.temperature, dim=1).detach()
#                     loss = self.criterion(student_output_log_prob, teacher_output_soft)
                
#                 loss.backward()
                
#                 # Clip gradients
#                 torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.clip_threshold)
#                 self.optimizer.step()
                
#                 # Update statistics
#                 train_loss += loss.item() * images_student.size(0)
#                 _, predicted = torch.max(student_outputs.data, 1)
#                 train_total += labels.size(0)
#                 train_correct += (predicted == labels).sum().item()
            
#             train_accuracy = 100 * train_correct / train_total
#             val_accuracy = self.evaluate(self.val_loader)
            
#             # Log metrics
#             self.experiment.log_metric("train_loss", train_loss / train_total, step=epoch)
#             self.experiment.log_metric("train_accuracy", train_accuracy, step=epoch)
#             self.experiment.log_metric("val_accuracy", val_accuracy, step=epoch)
            
#             print(f'Epoch {epoch+1}, Train Loss: {train_loss/train_total:.4f}, '
#                   f'Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
            
#             # Check for early stopping
#             if val_accuracy > best_val_accuracy:
#                 best_val_accuracy = val_accuracy
#                 epochs_no_improve = 0
#                 torch.save(self.student_model.state_dict(), checkpoint_path)
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= self.config.early_stopping_patience:
#                     print("Early stopping due to no improvement in validation accuracy.")
#                     break
        
#         # Evaluate on test set
#         test_accuracy = self.evaluate(self.test_loader)
#         self.experiment.log_metric("test_accuracy", test_accuracy)
#         print(f'Test accuracy: {test_accuracy:.4f}')
        
#         # End experiment
#         self.experiment.end()
    
#     def evaluate(self, data_loader):
#         """Evaluate student model on the given data loader."""
#         self.student_model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in data_loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
                
#                 # Extract first channel and resize for student model
#                 images_student = images[:, 0:1, :, :]
#                 images_student = F.interpolate(
#                     images_student, 
#                     size=(self.student_im_size, self.student_im_size), 
#                     mode='bilinear', 
#                     align_corners=False
#                 )
                
#                 outputs = self.student_model(images_student)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         accuracy = 100 * correct / total
#         return accuracy

# class Student1Trainer(BaseTrainer):
#     """Trainer for the ResNet18 student model."""
#     def setup_data(self):
#         # Get transforms for teacher with the crop augmentation parameter from config
#         teacher_train_transform, teacher_val_transform = get_teacher_transforms(
#             use_crop_augmentation=getattr(self.config, 'use_crop_augmentation', True)
#         )
        
#         # Get transforms for student
#         student_train_transform, student_val_transform = get_student_transforms(self.config.image_size)
        
#         # Load datasets for teacher
#         train_ds_teacher = FilteredLymphoMNIST(
#             LymphoMNIST(root='./dataset', train=True, download=True, transform=teacher_train_transform, num_classes=3), 
#             self.config.labels_to_keep
#         )
#         val_test_ds_teacher = FilteredLymphoMNIST(
#             LymphoMNIST(root='./dataset', train=False, download=True, transform=teacher_val_transform, num_classes=3), 
#             self.config.labels_to_keep
#         )
        
#         # Load datasets for student
#         train_ds_student = FilteredLymphoMNIST(
#             LymphoMNIST(root='./dataset', train=True, download=True, transform=student_train_transform, num_classes=3), 
#             self.config.labels_to_keep
#         )
#         val_test_ds_student = FilteredLymphoMNIST(
#             LymphoMNIST(root='./dataset', train=False, download=True, transform=student_val_transform, num_classes=3), 
#             self.config.labels_to_keep
#         )
        
#         # Create dataloaders for teacher
#         self.train_loader_teacher, self.val_loader_teacher, self.test_loader_teacher = get_dataloaders(
#             train_ds_teacher, 
#             val_test_ds_teacher, 
#             batch_size=self.config.batch_size,
#             num_workers=4
#         )
        
#         # Create dataloaders for student
#         self.train_loader_student, self.val_loader_student, self.test_loader_student = get_dataloaders(
#             train_ds_student, 
#             val_test_ds_student, 
#             batch_size=self.config.batch_size,
#             num_workers=4
#         )
    
#     def setup_model(self):
#         # Initialize teacher model
#         teacher_model = TeacherModel(num_classes=self.config.num_classes).to(self.device)
#         teacher_model.load_state_dict(torch.load(self.config.teacher_checkpoint, map_location=self.device))
#         teacher_model.eval()
        
#         # Initialize student model
#         self.student_model = Student1(
#             num_classes=self.config.num_classes, 
#             in_channels=1
#         ).to(self.device)
        
#         # Create distiller
#         self.model = Distiller(self.student_model, teacher_model)
        
#         # Print model summary
#         summary(self.student_model, (1, self.config.image_size, self.config.image_size))
    
#     def setup_training(self):
#         # Initialize optimizer and criterion
#         self.optimizer = optim.AdamW(
#             self.student_model.parameters(), 
#             lr=self.config.learning_rate,
#             weight_decay=self.config.weight_decay
#         )
#         self.criterion = nn.KLDivLoss(reduction='batchmean')
        
#         # Initialize scheduler if needed
#         if self.config.use_timm_scheduler:
#             self.scheduler = CosineLRScheduler(
#                 self.optimizer,
#                 t_initial=20,
#                 lr_min=2e-8,
#                 cycle_mul=2.0,
#                 cycle_decay=0.5,
#                 cycle_limit=5,
#                 warmup_t=10,
#                 warmup_lr_init=1e-6,
#                 warmup_prefix=False,
#                 t_in_epochs=True,
#                 noise_range_t=None,
#                 noise_pct=0.67,
#                 noise_std=1.0,
#                 noise_seed=42,
#                 k_decay=1.0,
#                 initialize=True
#             )
    
#     def train(self):
#         best_val_accuracy = 0.0
#         epochs_no_improve = 0
#         best_val_accuracy = 0.0
#         epochs_no_improve = 0
#         checkpoint_path = f"{self.config.checkpoint_dir}/KD_{datetime.datetime.now().strftime('%d_%B_%H_%M')}_{self.config.model_type}.pt"
        
#         for epoch in range(self.config.epochs):
#             train_loss = 0.0
#             train_total = 0
#             train_correct = 0
#             self.student_model.train()
            
#             train_loader_progress = tqdm(
#                 zip(self.train_loader_student, self.train_loader_teacher),
#                 total=min(len(self.train_loader_student), len(self.train_loader_teacher)),
#                 desc=f"Epoch {epoch+1}/{self.config.epochs}"
#             )
            
#             for (inputs_student, labels_student), (inputs_teacher, labels_teacher) in train_loader_progress:
#                 inputs_student, labels_student = inputs_student.to(self.device), labels_student.to(self.device)
#                 inputs_teacher, labels_teacher = inputs_teacher.to(self.device), labels_teacher.to(self.device)
                
#                 self.optimizer.zero_grad()
                
#                 # Apply mixup if enabled
#                 if self.config.use_mixup:
#                     mixed_inputs_student, targets_a, targets_b, lam = mixup_data(
#                         inputs_student, 
#                         labels_student, 
#                         self.config.alpha, 
#                         self.device
#                     )
#                     student_output = self.student_model(mixed_inputs_student)
#                     teacher_output = self.model.get_teacher_output(inputs_teacher)
                    
#                     # Compute loss
#                     student_output_log_prob = F.log_softmax(student_output / self.config.temperature, dim=1)
#                     loss = mixup_criterion(
#                         self.criterion, 
#                         student_output_log_prob, 
#                         targets_a, 
#                         targets_b, 
#                         lam, 
#                         self.config.num_classes
#                     )
#                 else:
#                     student_output = self.student_model(inputs_student)
#                     teacher_output = self.model.get_teacher_output(inputs_teacher)
                    
#                     # Compute loss
#                     student_output_log_prob = F.log_softmax(student_output / self.config.temperature, dim=1)
#                     teacher_output_soft = F.softmax(teacher_output / self.config.temperature, dim=1).detach()
#                     loss = self.criterion(student_output_log_prob, teacher_output_soft)
                
#                 loss.backward()
                
#                 # Clip gradients
#                 torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.clip_threshold)
#                 self.optimizer.step()
                
#                 # Update statistics
#                 train_loss += loss.item() * inputs_student.size(0)
#                 _, predicted = torch.max(student_output.data, 1)
#                 train_total += labels_student.size(0)
#                 train_correct += (predicted == labels_student).sum().item()
            
#             train_accuracy = 100 * train_correct / train_total
#             val_accuracy = self.evaluate(self.val_loader_student)
            
#             # Log metrics
#             self.experiment.log_metric("train_loss", train_loss / train_total, step=epoch)
#             self.experiment.log_metric("train_accuracy", train_accuracy, step=epoch)
#             self.experiment.log_metric("val_accuracy", val_accuracy, step=epoch)
            
#             print(f'Epoch {epoch+1}, Train Loss: {train_loss/train_total:.4f}, '
#                   f'Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
            
#             # Step the scheduler if using timm scheduler
#             if self.config.use_timm_scheduler:
#                 self.scheduler.step(epoch)
            
#             # Check for early stopping
#             if val_accuracy > best_val_accuracy:
#                 best_val_accuracy = val_accuracy
#                 epochs_no_improve = 0
#                 torch.save(self.student_model.state_dict(), checkpoint_path)
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= self.config.early_stopping_patience:
#                     print("Early stopping due to no improvement in validation accuracy.")
#                     break
        
#         # Evaluate on test set
#         test_accuracy = self.evaluate(self.test_loader_student)
#         self.experiment.log_metric("test_accuracy", test_accuracy)
#         print(f'Test accuracy: {test_accuracy:.4f}')
        
#         # End experiment
#         self.experiment.end()
    
#     def evaluate(self, data_loader):
#         """Evaluate student model on the given data loader."""
#         self.student_model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in data_loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 outputs = self.student_model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         accuracy = 100 * correct / total
#         return accuracy

class StudentTrainer(BaseTrainer):
    """Unified trainer for both student models (Student1 and Student2)."""
    def setup_data(self):
        # Get image size from config
        if self.config.model_type == "Student1":
            self.student_im_size = self.config.student_image_size
            self.teacher_im_size = self.config.teacher_image_size
        elif self.config.model_type == "Student2":
            self.student_im_size = self.config.student_image_size
            self.teacher_im_size = self.config.teacher_image_size
        
        # Get transforms for teacher
        teacher_train_transform, teacher_val_transform = get_teacher_transforms(
            im_size=self.teacher_im_size
        )
        
        # Get transforms for student
        student_train_transform, student_val_transform = get_student_transforms(self.student_im_size)
        
        # Load datasets for teacher
        train_ds_teacher = FilteredLymphoMNIST(
            LymphoMNIST(root='./dataset', train=True, download=True, transform=teacher_train_transform, num_classes=3), 
            self.config.labels_to_keep
        )
        val_test_ds_teacher = FilteredLymphoMNIST(
            LymphoMNIST(root='./dataset', train=False, download=True, transform=teacher_val_transform, num_classes=3), 
            self.config.labels_to_keep
        )
        
        # Load datasets for student
        train_ds_student = FilteredLymphoMNIST(
            LymphoMNIST(root='./dataset', train=True, download=True, transform=student_train_transform, num_classes=3), 
            self.config.labels_to_keep
        )
        val_test_ds_student = FilteredLymphoMNIST(
            LymphoMNIST(root='./dataset', train=False, download=True, transform=student_val_transform, num_classes=3), 
            self.config.labels_to_keep
        )
        
        # Create dataloaders for teacher
        self.train_loader_teacher, self.val_loader_teacher, self.test_loader_teacher = get_dataloaders(
            train_ds_teacher, 
            val_test_ds_teacher, 
            batch_size=self.config.batch_size,
            num_workers=4
        )
        
        # Create dataloaders for student
        self.train_loader_student, self.val_loader_student, self.test_loader_student = get_dataloaders(
            train_ds_student, 
            val_test_ds_student, 
            batch_size=self.config.batch_size,
            num_workers=4
        )
        
        # Set the main loaders for evaluation
        self.train_loader = self.train_loader_student
        self.val_loader = self.val_loader_student
        self.test_loader = self.test_loader_student
    
    def setup_model(self):
        # Initialize teacher model
        teacher_model = TeacherModel(num_classes=self.config.num_classes).to(self.device)
        teacher_model.load_state_dict(torch.load(self.config.teacher_checkpoint, map_location=self.device))
        teacher_model.eval()
        
        # Initialize student model based on model_type
        if self.config.model_type == "Student1":
            self.student_model = Student1(
                num_classes=self.config.num_classes, 
                in_channels=1
            ).to(self.device)
        elif self.config.model_type == "Student2":
            self.student_model = Student2(
                num_classes=self.config.num_classes, 
                input_size=(1, self.student_im_size, self.student_im_size)
            ).to(self.device)
        
        # Create distiller
        self.model = Distiller(self.student_model, teacher_model)
        
        # Print model summary
        summary(self.student_model, (1, self.student_im_size, self.student_im_size))
    
    def setup_training(self):
        # Initialize optimizer and criterion
        self.optimizer = optim.AdamW(
            self.student_model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        
        # Initialize scheduler if needed for Student1
        if self.config.model_type == "Student1" and getattr(self.config, 'use_timm_scheduler', False):
            self.scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=20,
                lr_min=2e-8,
                cycle_mul=2.0,
                cycle_decay=0.5,
                cycle_limit=5,
                warmup_t=10,
                warmup_lr_init=1e-6,
                warmup_prefix=False,
                t_in_epochs=True,
                noise_range_t=None,
                noise_pct=0.67,
                noise_std=1.0,
                noise_seed=42,
                k_decay=1.0,
                initialize=True
            )
    
    def train(self):
        best_val_accuracy = 0.0
        epochs_no_improve = 0
        checkpoint_path = f"{self.config.checkpoint_dir}/KD_{datetime.datetime.now().strftime('%d_%B_%H_%M')}_{self.config.model_type}.pt"
        
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            train_total = 0
            train_correct = 0
            self.student_model.train()
            
            # Use the same training loop for both Student1 and Student2
            train_loader_progress = tqdm(
                zip(self.train_loader_student, self.train_loader_teacher),
                total=min(len(self.train_loader_student), len(self.train_loader_teacher)),
                desc=f"Epoch {epoch+1}/{self.config.epochs}"
            )
            
            for (inputs_student, labels_student), (inputs_teacher, labels_teacher) in train_loader_progress:
                inputs_student, labels_student = inputs_student.to(self.device), labels_student.to(self.device)
                inputs_teacher, labels_teacher = inputs_teacher.to(self.device), labels_teacher.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Apply mixup if enabled
                if self.config.use_mixup:
                    mixed_inputs_student, targets_a, targets_b, lam = mixup_data(
                        inputs_student, 
                        labels_student, 
                        self.config.alpha, 
                        self.device
                    )
                    student_output = self.student_model(mixed_inputs_student)
                    teacher_output = self.model.get_teacher_output(inputs_teacher)
                    
                    # Compute loss
                    student_output_log_prob = F.log_softmax(student_output / self.config.temperature, dim=1)
                    loss = mixup_criterion(
                        self.criterion, 
                        student_output_log_prob, 
                        targets_a, 
                        targets_b, 
                        lam, 
                        self.config.num_classes
                    )
                else:
                    student_output = self.student_model(inputs_student)
                    teacher_output = self.model.get_teacher_output(inputs_teacher)
                    
                    # Compute loss
                    student_output_log_prob = F.log_softmax(student_output / self.config.temperature, dim=1)
                    teacher_output_soft = F.softmax(teacher_output / self.config.temperature, dim=1).detach()
                    loss = self.criterion(student_output_log_prob, teacher_output_soft)
                
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.clip_threshold)
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs_student.size(0)
                _, predicted = torch.max(student_output.data, 1)
                train_total += labels_student.size(0)
                train_correct += (predicted == labels_student).sum().item()
            
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = self.evaluate(self.val_loader)
            
            # Log metrics
            self.experiment.log_metric("train_loss", train_loss / train_total, step=epoch)
            self.experiment.log_metric("train_accuracy", train_accuracy, step=epoch)
            self.experiment.log_metric("val_accuracy", val_accuracy, step=epoch)
            
            print(f'Epoch {epoch+1}, Train Loss: {train_loss/train_total:.4f}, '
                  f'Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
            
            # Step the scheduler if using timm scheduler for Student1
            if self.config.model_type == "Student1" and getattr(self.config, 'use_timm_scheduler', False):
                self.scheduler.step(epoch)
            
            # Check for early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
                torch.save(self.student_model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stopping_patience:
                    print("Early stopping due to no improvement in validation accuracy.")
                    break
        
        # Evaluate on test set
        test_accuracy = self.evaluate(self.test_loader)
        self.experiment.log_metric("test_accuracy", test_accuracy)
        print(f'Test accuracy: {test_accuracy:.4f}')
        
        # End experiment
        self.experiment.end()
    
    def evaluate(self, data_loader):
        """Evaluate student model on the given data loader."""
        self.student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.student_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

def get_trainer(config):
    """Factory function to get the appropriate trainer based on the config."""
    if config.model_type == 'teacher':
        return TeacherTrainer(config)
    elif config.model_type == 'Student1' or config.model_type == 'Student2':
        return StudentTrainer(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
