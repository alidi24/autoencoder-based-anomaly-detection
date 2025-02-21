"""
Convolutional Autoencoder for Time Series Data


- 1D convolutional layers for feature extraction in the encoder
- 1D transposed convolutional layers for signal reconstruction in the decoder
- Batch normalization for training stability
- Configurable activation functions
- Dropout layers for regularization
- Symmetric structure with matching encoder and decoder layers

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict



class CAEModel(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(CAEModel, self).__init__()
        self.config = config
        self.activation = self._get_activation(config["act_fn"])
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {"relu": nn.ReLU(), "leakyrelu": nn.LeakyReLU(), "elu": nn.ELU(), "tanh": nn.Tanh()}
        return activations[name.lower()]

    def _get_encoder(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(
                in_channels=self.config["ch_nums"][0],
                out_channels=self.config["ch_nums"][1],
                kernel_size=self.config["conv_kernel_sizes"][0],
                stride=self.config["stride"][0],
                padding=self.config["padding"][0],
            ),
            nn.BatchNorm1d(self.config["ch_nums"][1]),
            self.activation,
            nn.Dropout(self.config["dropout_input"]),

            nn.Conv1d(
                in_channels=self.config["ch_nums"][1],
                out_channels=self.config["ch_nums"][2],
                kernel_size=self.config["conv_kernel_sizes"][1],
                stride=self.config["stride"][1],
                padding=self.config["padding"][1],
            ),
            nn.BatchNorm1d(self.config["ch_nums"][2]),
            self.activation,
            nn.Dropout(self.config["dropout_hidden"]),

            nn.Conv1d(
                in_channels=self.config["ch_nums"][2],
                out_channels=self.config["ch_nums"][3],
                kernel_size=self.config["conv_kernel_sizes"][2],
                stride=self.config["stride"][2],
                padding=self.config["padding"][2],
            ),
            nn.BatchNorm1d(self.config["ch_nums"][3]),
            self.activation,
            nn.Dropout(self.config["dropout_hidden"]),
        )

    def _get_decoder(self) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.config["ch_nums"][3],
                out_channels=self.config["ch_nums"][2],
                kernel_size=self.config["conv_kernel_sizes"][2],
                stride=self.config["stride"][2],
                padding=self.config["padding"][2],
            ),
            nn.BatchNorm1d(self.config["ch_nums"][2]),
            self.activation,
            nn.Dropout(self.config["dropout_hidden"]),

            nn.ConvTranspose1d(
                in_channels=self.config["ch_nums"][2],
                out_channels=self.config["ch_nums"][1],
                kernel_size=self.config["conv_kernel_sizes"][1],
                stride=self.config["stride"][1],
                padding=self.config["padding"][1],
            ),
            nn.BatchNorm1d(self.config["ch_nums"][1]),
            self.activation,
            nn.Dropout(self.config["dropout_hidden"]),

            nn.ConvTranspose1d(
                in_channels=self.config["ch_nums"][1],
                out_channels=self.config["ch_nums"][0],
                kernel_size=self.config["conv_kernel_sizes"][0],
                stride=self.config["stride"][0],
                padding=self.config["padding"][0],
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 0), "constant", 0)
        x_encode = self.encoder(x)
        x_reconstructed = self.decoder(x_encode)
        x_reconstructed = x_reconstructed[..., : self.config["frame_length"]]
        return x_reconstructed