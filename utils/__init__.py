"""
Utility functions for Snooker RL
"""

import os
import json
import numpy as np
import torch
from datetime import datetime


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_info():
    """Get information about available compute devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        
    return info


def save_config(config, filepath):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(filepath):
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_experiment_dir(base_dir='experiments'):
    """Create timestamped experiment directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class LinearSchedule:
    """Linear learning rate schedule"""
    def __init__(self, start_value, end_value, start_step, end_step):
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step
        
    def get_value(self, step):
        if step <= self.start_step:
            return self.start_value
        if step >= self.end_step:
            return self.end_value
            
        fraction = (step - self.start_step) / (self.end_step - self.start_step)
        return self.start_value + fraction * (self.end_value - self.start_value)


def compute_running_stats(values, window=100):
    """Compute running statistics"""
    if len(values) < window:
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        
    return {
        'mean': np.mean(values[-window:]),
        'std': np.std(values[-window:]),
        'min': np.min(values[-window:]),
        'max': np.max(values[-window:])
    }


def format_time(seconds):
    """Format seconds to human readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
