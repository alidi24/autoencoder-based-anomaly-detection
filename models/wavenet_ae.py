"""
WaveNet Autoencoder for Time Series Data


- WaveNet blocks with dilated convolutions for capturing multi-scale temporal patterns
- Gated activation functions combining tanh and sigmoid activations
- Residual connections within each WaveNet block for gradient flow
- Two-stage downsampling/upsampling for efficient information compression
- Balanced encoder/decoder structure with mirrored dilation patterns

"""

import torch
import torch.nn as nn
from typing import Any, Dict


class WaveNetBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2) -> None:
        super(WaveNetBlock, self).__init__()
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size=5, padding="same", dilation=dilation)
        self.bn = nn.BatchNorm1d(2 * channels)
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.bn(self.conv(x))
        filter_, gate = torch.chunk(out, 2, dim=1)
        out = torch.tanh(filter_) * torch.sigmoid(gate)
        return self.res_conv(out) + x


class WaveNetAEModel(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(WaveNetAEModel, self).__init__()
        
        # Get parameters from config
        input_channels = config.get("wavenet_input_channels", 1)
        hidden_channels = config.get("wavenet_hidden_channels", 32)
        bottleneck_channels = config.get("wavenet_bottleneck_channels", 8)
        num_blocks = config.get("wavenet_blocks", 4)
        dilation_base = config.get("wavenet_dilation_base", 2)
        dropout = config.get("wavenet_dropout", 0.2)
        
        # Initialize dilations
        dilations = [dilation_base ** i for i in range(num_blocks)]
        
        # Build encoder
        encoder_layers = []
        
        # Add encoder blocks with increasing dilation
        for i, dilation in enumerate(dilations):
            
            if i == 0:
                # Create a special first block that can handle input channels
                first_block = nn.Sequential(
                    nn.Conv1d(input_channels, hidden_channels, kernel_size=1),
                    WaveNetBlock(hidden_channels, dilation, dropout)
                )
                encoder_layers.append(first_block)
            else:
                encoder_layers.append(WaveNetBlock(hidden_channels, dilation, dropout))
        
        # Add downsampling to bottleneck
        encoder_layers.extend([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=4, stride=4),
            nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=4, stride=4)
        ])
            
        # Create encoder
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = [
            # Upsample from bottleneck - two stages to expand from 256 to 4096
            nn.ConvTranspose1d(bottleneck_channels, hidden_channels, kernel_size=4, stride=4),
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=4)
        ]
        
        # Add decoder blocks with reversed dilation pattern
        for i, dilation in enumerate(reversed(dilations)):
            
            if i == num_blocks - 1:
                # Create a special last block that can output the right channels
                last_block = nn.Sequential(
                    WaveNetBlock(hidden_channels, dilation, dropout),
                    nn.Conv1d(hidden_channels, input_channels, kernel_size=1)
                )
                decoder_layers.append(last_block)
            else:
                decoder_layers.append(WaveNetBlock(hidden_channels, dilation, dropout))
        
        # Create decoder
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store frame length for possible output adjustment
        self.frame_length = config.get("frame_length", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        
        # Ensure output length matches input length if frame_length is specified
        if self.frame_length is not None and reconstructed.shape[-1] != self.frame_length:
            reconstructed = reconstructed[..., :self.frame_length]
            
        return reconstructed