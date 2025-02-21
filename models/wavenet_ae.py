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
        bottleneck_channels = config.get("wavenet_bottleneck_channels", 16)
        encoder_blocks = config.get("wavenet_encoder_blocks", 8)
        decoder_blocks = config.get("wavenet_decoder_blocks", 8)
        dilation_base = config.get("wavenet_dilation_base", 2)
        dropout = config.get("wavenet_dropout", 0.2)
        
        # Build encoder
        encoder_layers = [
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels)
        ]
        
        # Add encoder blocks with increasing dilation
        for i in range(encoder_blocks):
            dilation = dilation_base ** i
            encoder_layers.append(WaveNetBlock(hidden_channels, dilation, dropout))
        
        # Add downsampling
        encoder_layers.append(nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=4, stride=4))
            
        # Create encoder
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = [
            # Upsample
            nn.ConvTranspose1d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=4)
        ]
        
        # Add decoder blocks
        for i in range(decoder_blocks):
            dilation = dilation_base ** (i % 4)
            decoder_layers.append(WaveNetBlock(bottleneck_channels, dilation, dropout))
            
        # Add final layers
        decoder_layers.extend([
            nn.ConvTranspose1d(bottleneck_channels, hidden_channels, kernel_size=4, stride=4),
            nn.BatchNorm1d(hidden_channels)
        ])
        
        for i in range(decoder_blocks):
            dilation = dilation_base ** i
            decoder_layers.append(WaveNetBlock(hidden_channels, dilation, dropout))
            
        decoder_layers.append(nn.Conv1d(hidden_channels, input_channels, kernel_size=7, padding=3))
        
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