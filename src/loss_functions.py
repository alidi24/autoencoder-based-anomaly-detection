"""
Loss functions for time series anomaly detection.

This module provides various loss functions for comparing original and 
reconstructed time series in both time and frequency domains.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_signal(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove channel dimension from signals for loss calculation.
    
    Args:
        x: Original signal [batch_size, channels, signal_length]
        y: Reconstructed signal [batch_size, channels, signal_length]
        
    Returns:
        Tuple of prepared signals [batch_size, signal_length]
    """
    return x.squeeze(1), y.squeeze(1)


def to_frequency_domain(x: torch.Tensor) -> torch.Tensor:
    """
    Convert signal to frequency domain using FFT.
    
    Args:
        x: Time-domain signal
        
    Returns:
        Frequency-domain representation
    """
    return torch.abs(torch.fft.rfft(x)) / x.shape[-1]


def apply_in_time_domain(x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
    """
    Apply a loss function in the time domain.
    
    Args:
        x: Original signal
        y: Reconstructed signal
        loss_fn: Loss function to apply
        
    Returns:
        Loss value
    """
    x, y = prepare_signal(x, y)
    return loss_fn(x, y)


def apply_in_frequency_domain(x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
    """
    Apply a loss function in the frequency domain.
    
    Args:
        x: Original signal
        y: Reconstructed signal
        loss_fn: Loss function to apply
        
    Returns:
        Loss value
    """
    x, y = prepare_signal(x, y)
    x_freq = to_frequency_domain(x)
    y_freq = to_frequency_domain(y)
    return loss_fn(x_freq, y_freq)


def apply_in_both_domains(
    x: torch.Tensor, 
    y: torch.Tensor, 
    loss_fn: Callable,
    time_weight: float = 0.5,
    freq_weight: float = 0.5
) -> torch.Tensor:
    """
    Apply a loss function in both time and frequency domains.
    
    Args:
        x: Original signal
        y: Reconstructed signal
        loss_fn: Loss function to apply
        time_weight: Weight for time domain loss (default 0.5)
        freq_weight: Weight for frequency domain loss (default 0.5)
        
    Returns:
        Combined loss value
    """
    time_loss = apply_in_time_domain(x, y, loss_fn)
    freq_loss = apply_in_frequency_domain(x, y, loss_fn)
    return time_weight * time_loss + freq_weight * freq_loss


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Calculate KL divergence between two distributions.
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        KL divergence
    """
    # Normalize to probability distributions
    p_dist = p / (torch.sum(p, dim=-1, keepdim=True) + 1e-6)
    q_dist = q / (torch.sum(q, dim=-1, keepdim=True) + 1e-6)

    # Calculate KL divergence
    kl = torch.sum(p_dist * torch.log(p_dist / (q_dist + 1e-6) + 1e-6), dim=-1)
    return kl.mean()



def shape_factor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate shape factor loss (ratio of RMS to mean absolute value).
    Good for detecting general pattern changes.
    
    Args:
        x: First signal
        y: Second signal
        
    Returns:
        Shape factor loss
    """
    def get_shape_factor(signal):
        rms = torch.sqrt(torch.mean(signal**2, dim=-1))
        mean_abs = torch.mean(torch.abs(signal), dim=-1)
        return rms / (mean_abs + 1e-8)
        
    return torch.abs(get_shape_factor(x) - get_shape_factor(y)).mean()


def combined_loss(x: torch.Tensor, y: torch.Tensor, weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """
    Combine multiple loss functions with configurable weights.
    
    Args:
        x: Original signal
        y: Reconstructed signal
        weights: Dictionary of loss weights. Default weights if None provided.
        
    Returns:
        Weighted combination of multiple losses
    """
    if weights is None:
        weights = {
            "mse": 1.0,
            "kl_div": 0.5,
            "shape": 0.2
        }
    
    # Calculate individual losses
    mse_loss = F.mse_loss(x, y)
    kl_loss = kl_divergence(x, y)
    shape_loss = shape_factor(x, y)
    
    # Combine losses with weights
    total_loss = (
        weights.get("mse", 1.0) * mse_loss +
        weights.get("kl_div", 0.5) * kl_loss +
        weights.get("shape", 0.2) * shape_loss
    )
    
    return total_loss


def get_loss_function(loss_name: str, domain: str = "TIME", config: Optional[Dict] = None) -> Callable:
    """
    Get the specified loss function in the specified domain.
    
    Args:
        loss_name: Name of the loss function to use (MSE, MAE, HUBER, etc.)
        domain: Domain to apply the loss function in ("TIME", "FREQUENCY", or "BOTH")
        config: Configuration dictionary for loss parameters
        
    Returns:
        The loss function to use for training
        
    Raises:
        ValueError: If the specified loss function or domain is not recognized
    """
    if config is None:
        config = {}
        
    # Define all available loss functions
    loss_fns = {
        "MSE": nn.MSELoss(),
        "MAE": nn.L1Loss(),
        "HUBER": nn.HuberLoss(),
        "COSINE": lambda x, y: 1 - nn.CosineSimilarity(dim=1)(x, y).mean(),
        "KL_DIVERGENCE": kl_divergence,
        "COMBINED": lambda x, y: combined_loss(x, y, config.get("loss_weights")),
        "SHAPE_FACTOR": shape_factor,
    }
    
    if loss_name not in loss_fns:
        raise ValueError(f"Loss function '{loss_name}' not recognized. Available options: {list(loss_fns.keys())}")
    
    if domain not in ["TIME", "FREQUENCY", "BOTH"]:
        raise ValueError(f"Domain '{domain}' not recognized. Use 'TIME', 'FREQUENCY', or 'BOTH'")
    
    # Get the base loss function
    base_loss_fn = loss_fns[loss_name]
    
    # Wrap it in the appropriate domain
    if domain == "TIME":
        return lambda x, y: apply_in_time_domain(x, y, base_loss_fn)
    elif domain == "FREQUENCY":
        return lambda x, y: apply_in_frequency_domain(x, y, base_loss_fn)
    else:  # "BOTH"
        time_weight = config.get("time_domain_weight", 0.5)
        freq_weight = config.get("freq_domain_weight", 0.5)
        return lambda x, y: apply_in_both_domains(x, y, base_loss_fn, time_weight, freq_weight)