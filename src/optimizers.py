"""
Optimizer utilities for training deep learning models.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.optim import Adam, SGD, AdamW, Optimizer
from torch.optim import lr_scheduler


def get_optimizer(
    model_parameters: Any,
    optimizer_name: str = "ADAM",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs
) -> Optimizer:
    """
    Create an optimizer for model training.
    
    Args:
        model_parameters: Parameters to optimize (typically model.parameters())
        optimizer_name: Name of the optimizer ("ADAM", "ADAMW", "SGD")
        lr: Learning rate
        weight_decay: L2 regularization factor
        **kwargs: Additional optimizer-specific parameters
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If the specified optimizer is not supported
    """
    optimizer_name = optimizer_name.upper()
    
    if optimizer_name == "ADAM":
        return Adam(model_parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == "ADAMW":
        return AdamW(model_parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == "SGD":
        # Default SGD momentum if not specified
        momentum = kwargs.pop("momentum", 0.9)
        return SGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Available options: ADAM, ADAMW, SGD")


def get_lr_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "PLATEAU",
    monitor: str = "val_loss",
    **kwargs
) -> Dict[str, Any]:
    """
    Create a learning rate scheduler configuration.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_name: Name of the scheduler ("PLATEAU", "STEP", "COSINE", etc.)
        monitor: Metric to monitor for ReduceLROnPlateau
        **kwargs: Additional scheduler-specific parameters
        
    Returns:
        Dictionary with scheduler configuration for PyTorch Lightning
        
    Raises:
        ValueError: If the specified scheduler is not supported
    """
    scheduler_name = scheduler_name.upper()
    
    scheduler_config = {
        "monitor": monitor,
        "frequency": 1,
        "strict": True
    }
    
    if scheduler_name == "PLATEAU":
        # Default parameters for ReduceLROnPlateau
        factor = kwargs.pop("factor", 0.1)
        patience = kwargs.pop("patience", 10)
        
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            **kwargs
        )
        
    elif scheduler_name == "STEP":
        # Default parameters for StepLR
        step_size = kwargs.pop("step_size", 30)
        gamma = kwargs.pop("gamma", 0.1)
        
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs
        )
        
    elif scheduler_name == "COSINE":
        # Default parameters for CosineAnnealingLR
        T_max = kwargs.pop("T_max", 100)  # Usually number of epochs
        eta_min = kwargs.pop("eta_min", 0)
        
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            **kwargs
        )
        
    else:
        raise ValueError(f"Unsupported scheduler '{scheduler_name}'. Available options: PLATEAU, STEP, COSINE")
    
    scheduler_config["scheduler"] = scheduler
    
    return scheduler_config