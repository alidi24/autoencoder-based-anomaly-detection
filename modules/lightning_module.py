import os
from typing import Any, Dict, Tuple, Optional, Callable, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW, lr_scheduler

from modules.loss_functions import get_loss_function
from modules.optimizers import get_optimizer, get_lr_scheduler
from modules.anomaly_visualization import plot_anomaly_scores


class LightningModel(pl.LightningModule):
    """
    Lightning Module for unsupervised anomaly detection using autoencoder architectures.
    
    This module wraps various autoencoder models and provides training, validation,
    and testing functionality with different loss functions for time series reconstruction.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]) -> None:
        """
        Initialize the Lightning Module.
        
        Args:
            model: The autoencoder model to be trained
            config: Configuration dictionary containing training parameters and loss function settings
        """
        super().__init__()
        self.model = model
        self.config = config
        
        # Set up loss function from config
        loss_name = config["loss_fn"].upper()
        domain = config.get("loss_domain", "TIME").upper()
        self.loss_fn = get_loss_function(loss_name, domain, config)

        # Initialize lists to store test anomaly scores
        self.test_anomaly_scores = []

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, signal_length]
            
        Returns:
            Reconstructed signal of the same shape
        """
        return self.model(x)
    


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform one training step.
        
        Args:
            batch: Batch of input signals
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing the loss
        """
        batch_recon = self(batch)
        loss = self.loss_fn(batch, batch_recon)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}   

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform one validation step.
        
        Args:
            batch: Batch of input signals
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing the validation loss
        """
        batch_recon = self(batch)
        loss = self.loss_fn(batch, batch_recon)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform one test step.
        
        Args:
            batch: Batch of input signals
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing the test loss
        """
        batch_recon = self(batch)
        loss = self.loss_fn(batch, batch_recon)

        # Calculate anomaly scores for each sample in the batch
        # Using the same loss function as training for consistency
        with torch.no_grad():
            # Calculate loss per sample
            anomaly_scores = []
            for i in range(batch.size(0)):
                # Extract individual samples
                sample = batch[i:i+1]  # Keep batch dimension
                sample_recon = batch_recon[i:i+1]  # Keep batch dimension
                
                sample_loss = self.loss_fn(sample, sample_recon)
                
                # Convert to Python float for storage
                if isinstance(sample_loss, torch.Tensor):
                    sample_score = sample_loss.item()
                else:
                    sample_score = float(sample_loss)
                
                anomaly_scores.append(sample_score)
            
            self.test_anomaly_scores.extend(anomaly_scores)
        
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss": loss}

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch to process collected anomaly scores.
        """
        # Check if anomaly score plotting is enabled
        if self.config.get("plot_anomaly_scores", True):
            plot_anomaly_scores(self.test_anomaly_scores, self.config)
        else:
            # Just save the raw scores without plotting
            output_dir = os.path.join(self.config.get("output_dir", "outputs"), "anomaly_data")
            os.makedirs(output_dir, exist_ok=True)
            model_name = self.config["which_model"]
            scores_save_path = os.path.join(output_dir, f"{model_name}_anomaly_scores.npy")
            np.save(scores_save_path, np.array(self.test_anomaly_scores))
            print(f"Anomaly scores saved to {scores_save_path}")
        
        # Reset scores list
        self.test_anomaly_scores = []


    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary with optimizer and lr_scheduler configuration
        """
        optimizer = get_optimizer(
            self.parameters(),
            optimizer_name=self.config.get("optimizer_name", "adam"),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )
        
        scheduler_config = get_lr_scheduler(
            optimizer,
            scheduler_name=self.config.get("scheduler_name", "plateau"),
            monitor=self.config["lr_monitor"],
            factor=self.config["lr_decay_factor"],
            patience=self.config["lr_patience"]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }