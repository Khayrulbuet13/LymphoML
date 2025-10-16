import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import pathlib

@dataclass
class TrainingConfig:
    """Base configuration for all training scenarios."""
    model_type: str  # 'teacher', 'Student2', 'Student1'
    batch_size: int = 64
    epochs: int = 10000
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 100
    clip_threshold: float = 1.0
    num_classes: int = 2
    labels_to_keep: List[int] = field(default_factory=lambda: [0, 1])
    device: str = "cuda"
    gpu_id: Optional[str] = None
    seed: int = 42
    comet_api_key: str = ""
    comet_workspace: str = ""
    comet_project_name: str = ""
    checkpoint_dir: str = "./checkpoint"
    teacher_checkpoint: str = "checkpoint/teacher_model.pt"
    
    def __post_init__(self):
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

@dataclass
class TeacherConfig(TrainingConfig):
    """Configuration for teacher model training."""
    model_type: str = "teacher"
    image_size: int = 120
    use_weighted_sampler: bool = True
    scheduler_patience: int = 5
    use_crop_augmentation: bool = True
    model_name: str = None
    
@dataclass
class Student2Config(TrainingConfig):
    """Configuration for quantized student model training."""
    model_type: str = "Student2"
    teacher_image_size: int = 120
    student_image_size: int = 48
    temperature: float = 1.0
    alpha: float = 1.0  # Mixup alpha
    use_mixup: bool = True
    
@dataclass
class Student1Config(TrainingConfig):
    """Configuration for ResNet18 student model training."""
    model_type: str = "Student1"
    teacher_image_size: int = 120
    student_image_size: int = 64
    temperature: float = 1.0
    alpha: float = 1.0  # Mixup alpha
    use_mixup: bool = True
    use_timm_scheduler: bool = True
    
def load_config_from_json(config_path: str) -> Union[TeacherConfig, Student2Config, Student1Config]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Load Comet credentials from secrets.json
    secrets_path = os.path.join(os.path.dirname(config_path), 'secrets.json')
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            secrets = json.load(f)
            # Add Comet credentials to config_dict
            config_dict['comet_api_key'] = secrets.get('comet_api_key')
            config_dict['comet_workspace'] = secrets.get('comet_workspace')
            config_dict['comet_project_name'] = secrets.get('comet_project_name')
    
    model_type = config_dict.get('model_type', '')
    
    if model_type == 'teacher':
        return TeacherConfig(**config_dict)
    elif model_type == 'Student2':
        return Student2Config(**config_dict)
    elif model_type == 'Student1':
        return Student1Config(**config_dict)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def save_config_to_json(config: Union[TeacherConfig, Student2Config, Student1Config], 
                        config_path: str) -> None:
    """Save configuration to a JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4)
