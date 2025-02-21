
"""
Data module for time series anomaly detection using autoencoder architectures.
Handles loading, preprocessing, and data splitting for normal and faulty data.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader
from src.transforms import SplitToFrame, NormalizeSample


class DataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for time series anomaly detection.
    
    Manages dataset loading, preprocessing, train/val splitting, and dataloader creation
    for normal (healthy) and anomalous (faulty) time series data.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DataModule with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for data processing
                   including frame_length, hop_length, batch_size, and train_val_ratio.
        """
        super().__init__()
        self.config = config
        self.normal_dataset = None
        self.faulty_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Load and preprocess the time series dataset from Hugging Face.
        
        The implementation loads data divided into:
        - Normal/healthy signals (train split)
        - Anomalous/faulty signals (test split)
        
        Each signal is split into frames and normalized.
        
        
        This implementation uses the CWRU Bearing dataset as an example,
        which contains vibration measurements from normal and faulty bearings.
        For more information on the original CWRU dataset:
        https://engineering.case.edu/bearingdatacenter
        
        The dataset can be replaced with any other time series dataset
        structured with normal and anomalous samples.
        
        Returns:
            Tuple of (normal_dataset, faulty_dataset)
        """
        # Load datasets
        self.normal_dataset, self.faulty_dataset = self._load_datasets()
        
        # Preprocess datasets
        self.normal_dataset = self._preprocess_dataset(self.normal_dataset)
        self.faulty_dataset = self._preprocess_dataset(self.faulty_dataset)
        
        return self.normal_dataset, self.faulty_dataset

    def _load_datasets(self):
        """
        Load datasets from the configured source.
        
        By default, uses the CWRU bearing dataset from Hugging Face.
        Override this method to use different data sources.
        
        Returns:
            Tuple of (normal_dataset, faulty_dataset)
        """
        ds_normal = load_dataset("alidi/cwru-dataset", split="train")
        ds_faulty = load_dataset("alidi/cwru-dataset", split="test")
        return ds_normal, ds_faulty

    def _preprocess_dataset(self, dataset):
        """
        Apply preprocessing transformations to a dataset.
        
        Args:
            dataset: The dataset to preprocess
            
        Returns:
            Preprocessed dataset
        """
        # Split signal into frames
        dataset = dataset.map(SplitToFrame(
            frame_length=self.config["frame_length"],
            hop_length=self.config["hop_length"],
            signal_column="signal",
            key_column="key"),
            batched=True)
        
        # Normalize samples
        dataset = dataset.map(NormalizeSample(), batched=True)
        
        return dataset

    def setup(self, stage: Optional[str] = None) -> Optional[Dataset]:
        """
        Set up datasets for different stages (fit, test).
        
        For training ('fit'): 
        - Splits normal data into training and validation sets
        
        For testing ('test'):
        - Combines normal and anomalous data for evaluation
        
        Args:
            stage: 'fit', 'test', or None
        
        Returns:
            Dataset splits relevant to the specified stage
        """
        if stage == "fit" or stage is None:
            seed_everything(self.config["random_seed"], workers=True)
            train_size = int(self.config["train_val_ratio"] * len(self.normal_dataset))

            self.normal_dataset = self.normal_dataset.shuffle(seed=self.config["random_seed"])
            self.train_dataset = self.normal_dataset.select(range(train_size))
            self.val_dataset = self.normal_dataset.select(range(train_size, len(self.normal_dataset)))
            return self.train_dataset, self.val_dataset
        
        elif stage == "test":
            self.test_dataset = concatenate_datasets([self.normal_dataset, self.faulty_dataset])
            return self.test_dataset

    def _get_dataloader_kwargs(self, is_train=False):
        """Get common DataLoader parameters."""
        return {
            "batch_size": self.config["batch_size"],
            "num_workers": self.config.get("num_workers", 7),
            "drop_last": True if is_train else False,
        }

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader with data augmentation."""
        kwargs = self._get_dataloader_kwargs(is_train=True)
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.train_collate,
            **kwargs
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader."""
        kwargs = self._get_dataloader_kwargs()
        return DataLoader(
            self.val_dataset,
            collate_fn=self._collate_fn,
            **kwargs
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test data loader with both normal and anomalous samples."""
        kwargs = self._get_dataloader_kwargs()
        return DataLoader(
            self.test_dataset,
            collate_fn=self._collate_fn,
            **kwargs
        )


    def train_collate(self, batch):
        return self._collate_fn(batch, is_train=True)

    def _collate_fn(
        self, batch: List[Dict[str, Any]], is_train: bool = False
    ) -> Tuple[torch.Tensor, List[Dict[str, float]], List[str]]:
        """
        Collate function for the dataloaders.
        
        Converts signal data to tensors and optionally applies augmentation during training.
        
        Args:
            batch: List of data samples
            is_train: Whether this batch is for training (to apply augmentation)
            
        Returns:
            Tensor of stacked signal data
        """
        if is_train and self.config["is_aug"]:
            batch = self.__augmentation(batch)

        default_float_dtype = torch.get_default_dtype()
        tensor_signals = [
            torch.tensor(sample["signal"], dtype=default_float_dtype).unsqueeze(0) for sample in batch
        ]

        return torch.stack(tensor_signals)

    def __augmentation(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply data augmentation to the batch:
        - Randomly invert the signal (70% chance)
        - Randomly roll the signal (70% chance)
        - Add Gaussian noise (80% chance)
        
        Args:
            batch: List of data samples
            
        Returns:
            Transformed batch with augmented signals
        """
        transformed_batch = []
        for example in batch:
            signal = np.array(example["signal"])
            if np.random.rand() > 0.3:
                signal = -signal

            if np.random.rand() > 0.3:
                roll_amount = np.random.randint(1, len(signal))
                direction = np.random.choice(["start", "end"])
                signal = np.roll(signal, roll_amount if direction == "start" else -roll_amount)


            if np.random.rand() > 0.2:
                signal_std = np.std(signal)
                noise_level = np.random.uniform(0.05, 0.10)  # 5-10% of signal std
                noise = np.random.normal(0, signal_std * noise_level, size=signal.shape)
                signal = signal + noise

            transformed_example = example.copy()
            transformed_example["signal"] = signal
            transformed_batch.append(transformed_example)
        return transformed_batch