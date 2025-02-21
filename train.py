"""Train script for unsupervised anomaly detection using three autoencoder architectures. Handles model training, validation and testing 
with early stopping and model checkpointing."""

import os
from typing import List
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.callback import Callback
from config import config
from src.data_module import DataModule
from src.lightning_module import LightningModel

def setup_callbacks() -> List[Callback]:
    """Initialize training callbacks for model checkpointing and early stopping."""
    # Create outputs directory structure if it doesn't exist
    output_dir = config.get("output_dir", "outputs")
    checkpoint_dir = os.path.join(output_dir, "model_checkpoints")
    
    # Ensure directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Optionally clean the directory if you want to start fresh each run
    if config.get("clean_checkpoints", False):
        import shutil
        shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,  # Updated path
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor=config["early_stop_monitor"],
        min_delta=config["early_stop_mindelta"],
        patience=config["early_stop_patience"],
        verbose=True,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [checkpoint_callback, lr_monitor, early_stop_callback]

def main() -> None:
    """Main training loop. Sets up model, data, and trainer for anomaly detection.

    This function:
        1. Initializes training callbacks
        2. Creates the appropriate data module
        3. Loads the specified autoencoder model architecture (CAE, WavenetAE, or AttentionAE)
        4. Wraps the base model in a Lightning module
        5. Performs model training with early stopping
        6. Evaluates the best checkpoint on the test set
    """
    torch.set_float32_matmul_precision("high")
    callbacks = setup_callbacks()

    # Initialize modules
    data_module = DataModule(config)
    if config['which_model'] == 'CAE':
        from models.cae import CAEModel
        base_model = CAEModel(config)
    elif config['which_model'] == 'WavenetAE':
        from models.wavenet_ae import WaveNetAEModel
        base_model = WaveNetAEModel(config)
    elif config['which_model'] == 'AttentionAE':
        from models.attention_ae import AttentionAEModel
        base_model = AttentionAEModel(config)

    model = LightningModel(base_model, config)


    # Create output directories
    output_dir = config.get("output_dir", "outputs")
    logs_dir = os.path.join(output_dir, "lightning_logs")
    os.makedirs(logs_dir, exist_ok=True)

    trainer = Trainer(
        max_epochs=config["max_epoch"],
        callbacks=callbacks,
        accelerator="cuda",  # "cpu" or "cuda"
        default_root_dir=output_dir,
    )

    try:
        trainer.fit(model, datamodule=data_module)

        # Test best model
        checkpoint_callback = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
        best_model_path = checkpoint_callback.best_model_path

        if best_model_path:
            print(f"Loading best model from {best_model_path}")
            model = LightningModel.load_from_checkpoint(
                best_model_path, model=base_model, config=config
            )
            trainer.test(model, datamodule=data_module)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

if __name__ == "__main__":
    main()