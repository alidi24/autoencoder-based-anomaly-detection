import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List


class SelfAttention(nn.Module):
    """Simple self-attention module for 1D signals."""
    def __init__(self, channels: int, head_dim: int, num_heads: int):
        super(SelfAttention, self).__init__()
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.qkv_proj = nn.Conv1d(channels, 3 * channels, kernel_size=1)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        qkv = self.qkv_proj(x)  # [b, 3*c, n]
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, n)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, b, num_heads, n, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [b, num_heads, n, head_dim]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, num_heads, n, n]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v)  # [b, num_heads, n, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(b, c, n)
        out = self.out_proj(out)
        return out


class ResidualAttentionBlock(nn.Module):
    """Residual block with convolution, batch norm, activation, and self-attention."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, padding: int, head_dim: int, num_heads: int, dropout: float):
        super(ResidualAttentionBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = SelfAttention(out_channels, head_dim, num_heads)
        
        # Residual connection adapter if dimensions don't match
        self.residual_adapter = None
        if stride != 1 or in_channels != out_channels:
            self.residual_adapter = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self.conv_block(x)
        
        # Apply attention
        out = out + self.attention(out)
        
        # Apply residual connection if possible
        if self.residual_adapter is not None:
            residual = self.residual_adapter(x)
        else:
            residual = x
            
        # Add residual connection
        if residual.shape == out.shape:
            out = out + residual
            
        return out


class AttentionAEModel(nn.Module):
    """Autoencoder with attention mechanisms and skip connections"""
    def __init__(self, config: Dict[str, Any]) -> None:
        super(AttentionAEModel, self).__init__()
        
        # Get parameters from config
        input_channels = config.get("attention_input_channels", 1)
        hidden_channels = config.get("attention_hidden_channels", [32, 64, 128])
        kernel_sizes = config.get("attention_kernel_sizes", [9, 5, 3])
        strides = config.get("attention_strides", [4, 2, 2])
        paddings = config.get("attention_paddings", [4, 2, 1])
        dropout = config.get("attention_dropout", 0.2)
        head_dim = config.get("attention_head_dim", 64)
        num_heads = config.get("attention_num_heads", 4)
        use_skip = config.get("use_skip_connections", True)
        
        self.use_skip_connections = use_skip
        
        # Build encoder blocks
        self.encoder_blocks = nn.ModuleList()
        channels = [input_channels] + hidden_channels
        
        for i in range(len(hidden_channels)):
            self.encoder_blocks.append(
                ResidualAttentionBlock(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    head_dim=head_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
        
        # Build decoder blocks (in reverse order)
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_channels)-1, -1, -1):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=channels[i+1], 
                        out_channels=channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i],
                        output_padding=0
                    ),
                    nn.BatchNorm1d(channels[i]) if i > 0 else nn.Identity(),
                    nn.ReLU() if i > 0 else nn.Identity()
                )
            )
        
        # Store frame length for output size adjustment
        self.frame_length = config.get("frame_length", 4096)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store encoder outputs for skip connections
        skip_outputs = []
        
        # Encoder forward pass
        for block in self.encoder_blocks:
            x = block(x)
            skip_outputs.append(x)
        
        # Decoder forward pass with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            
            # Apply skip connection if enabled and dimensions match
            if self.use_skip_connections and i < len(self.encoder_blocks) - 1:
                skip_idx = len(self.encoder_blocks) - i - 2
                skip = skip_outputs[skip_idx]
                
                # Adjust dimensions if needed
                if x.shape[-1] != skip.shape[-1]:
                    # Pad or trim to match
                    if x.shape[-1] < skip.shape[-1]:
                        x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
                    else:
                        x = x[..., :skip.shape[-1]]
                
                if x.shape[1] == skip.shape[1]:  # Check channel dimension
                    x = x + skip
        
        # Ensure output has the correct length
        if x.shape[-1] != self.frame_length:
            if x.shape[-1] > self.frame_length:
                x = x[..., :self.frame_length]
            else:
                x = F.pad(x, (0, self.frame_length - x.shape[-1]))
                
        return x