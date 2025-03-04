import torch
import torch.nn as nn
import torch.nn.functional as F

class Distiller(nn.Module):
    """
    Knowledge Distillation model that combines a teacher and student model.
    """
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        
        # Freeze the teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def forward(self, x, student_input=None):
        """
        Forward pass through both teacher and student models.
        
        Args:
            x: Input tensor for the teacher model
            student_input: Optional separate input for the student model.
                           If None, x will be used for both models.
        
        Returns:
            student_output: Output from the student model
            teacher_output: Output from the teacher model
        """
        if student_input is None:
            student_input = x
            
        teacher_output = self.teacher(x)
        student_output = self.student(student_input)
        
        return student_output, teacher_output
    
    def get_student_output(self, x):
        """Get only the student model output."""
        return self.student(x)
    
    def get_teacher_output(self, x):
        """Get only the teacher model output."""
        with torch.no_grad():
            return self.teacher(x)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Applies Mixup augmentation to the inputs and targets."""
    if alpha > 0:
        lam = torch.tensor(torch.distributions.Beta(alpha, alpha).sample())
    else:
        lam = torch.tensor(1.0)
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, num_classes):
    """Calculates the Mixup loss."""
    y_a_one_hot = F.one_hot(y_a, num_classes=num_classes).float()
    y_b_one_hot = F.one_hot(y_b, num_classes=num_classes).float()
    return lam * criterion(pred, y_a_one_hot) + (1 - lam) * criterion(pred, y_b_one_hot)

def distillation_loss(student_logits, teacher_logits, labels, num_classes, 
                       alpha=0.5, temperature=1.0, use_mixup=False, mixup_alpha=1.0):
    """
    Compute the knowledge distillation loss.
    
    Args:
        student_logits: Logits from the student model
        teacher_logits: Logits from the teacher model
        labels: Ground truth labels
        num_classes: Number of classes
        alpha: Weight for balancing the soft and hard targets
        temperature: Temperature for softening the teacher outputs
        use_mixup: Whether to use mixup augmentation
        mixup_alpha: Alpha parameter for mixup
        
    Returns:
        total_loss: Combined distillation loss
    """
    if use_mixup:
        device = student_logits.device
        mixed_student_logits, targets_a, targets_b, lam = mixup_data(
            student_logits, labels, mixup_alpha, device)
        
        # KL divergence loss for soft targets
        student_log_softmax = F.log_softmax(mixed_student_logits / temperature, dim=1)
        teacher_softmax = F.softmax(teacher_logits / temperature, dim=1).detach()
        soft_loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean')
        
        # Cross entropy loss for hard targets with mixup
        hard_loss = mixup_criterion(
            lambda x, y: F.cross_entropy(x, y), 
            mixed_student_logits, 
            targets_a, 
            targets_b, 
            lam, 
            num_classes
        )
    else:
        # KL divergence loss for soft targets
        student_log_softmax = F.log_softmax(student_logits / temperature, dim=1)
        teacher_softmax = F.softmax(teacher_logits / temperature, dim=1).detach()
        soft_loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean')
        
        # Cross entropy loss for hard targets
        hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combine the losses
    total_loss = (1 - alpha) * hard_loss + alpha * soft_loss * (temperature ** 2)
    
    return total_loss
