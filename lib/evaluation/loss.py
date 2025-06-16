"""
Losses and Attention Modules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##############################################################################
# Temporal Attention Modules for Video Processing
##############################################################################

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

##############################################################################
# Basic Loss Functions
##############################################################################

def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)

##############################################################################
# Temporal Losses for Video Anomaly Detection
##############################################################################

class TemporalConsistencyLoss(nn.Module):
    """
    Temporal Consistency Loss for video anomaly detection
    Enforces smooth temporal transitions between consecutive frames
    """
    
    def __init__(self, alpha=0.1, beta=0.05):
        super(TemporalConsistencyLoss, self).__init__()
        self.alpha = alpha  # Weight for frame-to-frame consistency
        self.beta = beta    # Weight for feature consistency
        
    def forward(self, predictions, features=None):
        """
        Args:
            predictions: Generated frames (B, T, C, H, W)
            features: Intermediate features (B, T, F, H, W) - optional
        """
        batch_size, num_frames = predictions.shape[:2]
        
        # Frame-to-frame temporal consistency
        frame_diff_loss = 0.0
        for t in range(num_frames - 1):
            current_frame = predictions[:, t]
            next_frame = predictions[:, t + 1]
            
            # L1 difference between consecutive frames
            frame_diff = torch.abs(current_frame - next_frame)
            frame_diff_loss += torch.mean(frame_diff)
        
        frame_diff_loss /= (num_frames - 1)
        
        # Feature consistency loss (if features provided)
        feature_consistency_loss = 0.0
        if features is not None:
            for t in range(num_frames - 1):
                current_feat = features[:, t]
                next_feat = features[:, t + 1]
                
                # Cosine similarity for feature consistency
                cosine_sim = F.cosine_similarity(
                    current_feat.view(batch_size, -1), 
                    next_feat.view(batch_size, -1), 
                    dim=1
                )
                feature_consistency_loss += torch.mean(1 - cosine_sim)
            
            feature_consistency_loss /= (num_frames - 1)
        
        total_loss = self.alpha * frame_diff_loss + self.beta * feature_consistency_loss
        
        return total_loss


class TemporalMotionLoss(nn.Module):
    """
    Motion-aware temporal loss that considers optical flow patterns
    """
    
    def __init__(self, motion_weight=0.1):
        super(TemporalMotionLoss, self).__init__()
        self.motion_weight = motion_weight
        
        # Simple gradient-based motion estimation
        self.sobel_x = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.sobel_y = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        
        # Sobel filters for motion detection
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Set up kernels for each channel
        for i in range(3):
            self.sobel_x.weight.data[i, 0] = sobel_x_kernel
            self.sobel_y.weight.data[i, 0] = sobel_y_kernel
            
        # Freeze gradient computation for filters
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
    
    def compute_motion_map(self, frame1, frame2):
        """Compute motion map between two consecutive frames"""
        # Frame difference
        frame_diff = torch.abs(frame1 - frame2)
        
        # Gradient-based motion
        grad_x = torch.abs(self.sobel_x(frame_diff))
        grad_y = torch.abs(self.sobel_y(frame_diff))
        motion_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        return motion_magnitude
    
    def forward(self, real_frames, fake_frames):
        """
        Args:
            real_frames: Ground truth frames (B, T, C, H, W)
            fake_frames: Generated frames (B, T, C, H, W)
        """
        batch_size, num_frames = real_frames.shape[:2]
        motion_loss = 0.0
        
        for t in range(num_frames - 1):
            # Real motion
            real_motion = self.compute_motion_map(
                real_frames[:, t], 
                real_frames[:, t + 1]
            )
            
            # Fake motion  
            fake_motion = self.compute_motion_map(
                fake_frames[:, t], 
                fake_frames[:, t + 1]
            )
            
            # Motion consistency loss
            motion_diff = torch.abs(real_motion - fake_motion)
            motion_loss += torch.mean(motion_diff)
        
        motion_loss /= (num_frames - 1)
        
        return self.motion_weight * motion_loss


class TemporalAttentionRegularization(nn.Module):
    """
    Regularization for temporal attention weights to prevent overfitting
    """
    
    def __init__(self, entropy_weight=0.01, sparsity_weight=0.005):
        super(TemporalAttentionRegularization, self).__init__()
        self.entropy_weight = entropy_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(self, attention_weights):
        """
        Args:
            attention_weights: Attention weights from temporal attention (B*H*W, heads, T, T)
        """
        # Entropy regularization (encourage diversity)
        # Higher entropy = more diverse attention patterns
        log_attention = torch.log(attention_weights + 1e-8)
        entropy = -torch.sum(attention_weights * log_attention, dim=-1)
        entropy_loss = -torch.mean(entropy)  # Negative because we want to maximize entropy
        
        # Sparsity regularization (encourage focused attention)
        # L1 penalty to encourage sparse attention patterns
        sparsity_loss = torch.mean(torch.abs(attention_weights))
        
        total_reg = self.entropy_weight * entropy_loss + self.sparsity_weight * sparsity_loss
        
        return total_reg


class CombinedTemporalLoss(nn.Module):
    """
    Combined temporal loss that integrates all temporal objectives
    """
    
    def __init__(self, consistency_weight=0.1, motion_weight=0.05, reg_weight=0.01):
        super(CombinedTemporalLoss, self).__init__()
        
        self.consistency_loss = TemporalConsistencyLoss()
        self.motion_loss = TemporalMotionLoss(motion_weight=motion_weight)
        self.attention_reg = TemporalAttentionRegularization()
        
        self.consistency_weight = consistency_weight
        self.motion_weight = motion_weight
        self.reg_weight = reg_weight
    
    def forward(self, real_frames, fake_frames, features=None, attention_weights=None):
        """
        Comprehensive temporal loss computation
        
        Args:
            real_frames: Ground truth frames (B, T, C, H, W)
            fake_frames: Generated frames (B, T, C, H, W)  
            features: Intermediate features (B, T, F, H, W) - optional
            attention_weights: Temporal attention weights - optional
        """
        losses = {}
        total_loss = 0.0
        
        # Temporal consistency
        consistency_result = self.consistency_loss(fake_frames, features)
        losses['consistency'] = consistency_result
        total_loss += self.consistency_weight * consistency_result
        
        # Motion consistency
        motion_result = self.motion_loss(real_frames, fake_frames)
        losses['motion'] = motion_result
        total_loss += self.motion_weight * motion_result
        
        # Attention regularization
        if attention_weights is not None:
            reg_result = self.attention_reg(attention_weights)
            losses['attention_reg'] = reg_result
            total_loss += self.reg_weight * reg_result
        
        losses['total_temporal'] = total_loss
        
        return losses
