import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalAttention(nn.Module):
    """
    Temporal Attention Module for Video Processing
    
    Implements multi-head self-attention across the temporal dimension
    while preserving spatial structure.
    """
    def __init__(self, feature_dim, num_frames, num_heads=8, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Positional encoding for temporal relationships
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(num_frames, feature_dim) * 0.02
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)
            
        Returns:
            attended_x: Output tensor of same shape with temporal attention applied
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape to (batch * height * width, frames, channels)
        x_reshaped = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, C)
        x_reshaped = x_reshaped.view(-1, num_frames, channels)  # (B*H*W, T, C)
        
        # Add positional encoding
        x_pos = x_reshaped + self.temporal_pos_encoding.unsqueeze(0)
        
        # Compute Q, K, V
        Q = self.query_proj(x_pos)  # (B*H*W, T, C)
        K = self.key_proj(x_pos)
        V = self.value_proj(x_pos)
        
        # Reshape for multi-head attention: (B*H*W, T, heads, head_dim)
        Q = Q.view(-1, num_frames, self.num_heads, self.head_dim)
        K = K.view(-1, num_frames, self.num_heads, self.head_dim)
        V = V.view(-1, num_frames, self.num_heads, self.head_dim)
        
        # Transpose to (B*H*W, heads, T, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (B*H*W, heads, T, head_dim)
        
        # Concatenate heads: (B*H*W, T, C)
        attended = attended.transpose(1, 2).contiguous().view(-1, num_frames, channels)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x_reshaped)
        
        # Reshape back to original format
        output = output.view(batch_size, height, width, num_frames, channels)
        output = output.permute(0, 3, 4, 1, 2).contiguous()  # (B, T, C, H, W)
        
        return output


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM for temporal modeling in spatial feature maps
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # for i, f, g, o gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
    def forward(self, input_tensor, hidden_state=None):
        """
        Args:
            input_tensor: (batch, frames, channels, height, width)
            hidden_state: tuple of (h, c) each of shape (batch, hidden_dim, height, width)
        
        Returns:
            output_sequence: (batch, frames, hidden_dim, height, width)
        """
        batch_size, seq_len, _, height, width = input_tensor.shape
        
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_dim, height, width, 
                          device=input_tensor.device, dtype=input_tensor.dtype)
            c = torch.zeros(batch_size, self.hidden_dim, height, width,
                          device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            h, c = hidden_state
        
        output_sequence = []
        
        for t in range(seq_len):
            x_t = input_tensor[:, t, :, :, :]  # (batch, channels, height, width)
            
            # Concatenate input and hidden state
            combined = torch.cat([x_t, h], dim=1)  # (batch, input_dim + hidden_dim, H, W)
            
            # Compute gates
            combined_conv = self.conv(combined)
            cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
            
            i = torch.sigmoid(cc_i)  # input gate
            f = torch.sigmoid(cc_f)  # forget gate
            g = torch.tanh(cc_g)     # candidate values
            o = torch.sigmoid(cc_o)  # output gate
            
            # Update cell state and hidden state
            c = f * c + i * g
            h = o * torch.tanh(c)
            
            output_sequence.append(h.unsqueeze(1))
        
        output_sequence = torch.cat(output_sequence, dim=1)  # (batch, frames, hidden_dim, H, W)
        
        return output_sequence


class Temporal3DConv(nn.Module):
    """
    3D Convolution for temporal modeling with spatial preservation
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), 
                 stride=(1, 1, 1), padding=(1, 1, 1)):
        super(Temporal3DConv, self).__init__()
        
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: (batch, frames, channels, height, width)
        Returns:
            output: (batch, frames, out_channels, height, width)
        """
        # Permute to (batch, channels, frames, height, width) for 3D conv
        x = x.permute(0, 2, 1, 3, 4)
        
        # Apply 3D convolution
        x = self.conv3d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Permute back to (batch, frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        return x


class TemporalFeatureFusion(nn.Module):
    """
    Multi-scale temporal feature fusion combining different temporal modeling approaches
    """
    def __init__(self, feature_dim, num_frames):
        super(TemporalFeatureFusion, self).__init__()
        
        # Different temporal modeling components
        # Ensure feature_dim is divisible by num_heads (8 by default)
        num_heads = 8
        if feature_dim % num_heads != 0:
            # Adjust num_heads to be a divisor of feature_dim
            for heads in [4, 2, 1]:
                if feature_dim % heads == 0:
                    num_heads = heads
                    break
        
        self.temporal_attention = TemporalAttention(feature_dim, num_frames, num_heads=num_heads)
        self.conv_lstm = ConvLSTM(feature_dim, feature_dim // 2)
        self.temporal_3d = Temporal3DConv(feature_dim, feature_dim // 2)
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(
            feature_dim + feature_dim // 2 + feature_dim // 2,  # attention + lstm + 3d
            feature_dim,
            kernel_size=1
        )
        self.fusion_norm = nn.BatchNorm2d(feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, frames, channels, height, width)
        """
        batch, frames, channels, height, width = x.shape
        
        # Apply different temporal modeling approaches
        attn_features = self.temporal_attention(x)  # (B, T, C, H, W)
        lstm_features = self.conv_lstm(x)           # (B, T, C//2, H, W)
        conv3d_features = self.temporal_3d(x)       # (B, T, C//2, H, W)
        
        # Aggregate across temporal dimension (average pooling)
        attn_agg = torch.mean(attn_features, dim=1)      # (B, C, H, W)
        lstm_agg = torch.mean(lstm_features, dim=1)      # (B, C//2, H, W)
        conv3d_agg = torch.mean(conv3d_features, dim=1)  # (B, C//2, H, W)
        
        # Concatenate and fuse
        fused = torch.cat([attn_agg, lstm_agg, conv3d_agg], dim=1)  # (B, C + C//2 + C//2, H, W)
        output = self.fusion_conv(fused)  # (B, C, H, W)
        output = self.fusion_norm(output)
        
        return output
