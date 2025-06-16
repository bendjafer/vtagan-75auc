import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .FlowNetSD import FlowNetSD

##############################################################################
# Base Temporal Attention
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


class TemporalFeatureFusion(nn.Module):
    """
    Temporal Feature Fusion Module
    
    Combines features across temporal dimension using learned weights
    """
    def __init__(self, feature_dim, num_frames):
        super(TemporalFeatureFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_frames) / num_frames)
        
        # Optional: Add a small MLP for more complex fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * num_frames, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)
            
        Returns:
            fused: Fused tensor of shape (batch, channels, height, width)
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Simple weighted average
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted_features = x * weights.view(1, -1, 1, 1, 1)
        simple_fusion = torch.sum(weighted_features, dim=1)
        
        # Complex fusion using MLP
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous()
        x_flat = x_flat.view(batch_size, channels, height, width, -1)
        x_concat = x_flat.view(batch_size, channels * num_frames, height, width)
        
        # Apply MLP per spatial location
        b, c_expanded, h, w = x_concat.shape
        x_mlp_input = x_concat.permute(0, 2, 3, 1).contiguous()
        x_mlp_input = x_mlp_input.view(-1, c_expanded)
        
        fusion_mask = self.fusion_mlp(x_mlp_input)
        fusion_mask = fusion_mask.view(b, h, w, channels).permute(0, 3, 1, 2)
        
        # Combine simple and complex fusion
        fused = simple_fusion * fusion_mask
        
        return fused


##############################################################################
# Other Temporal Attention Modules
##############################################################################


class MultiScaleTemporalAttention(nn.Module):
    """
    Multi-scale temporal attention that operates at different temporal resolutions
    Captures both short-term and long-term temporal dependencies
    """
    
    def __init__(self, feature_dim, num_frames, num_heads=8):
        super(MultiScaleTemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # Different temporal scales
        self.scales = [1, 2, 4]  # 1x, 2x, 4x temporal downsampling
        
        # Temporal attention modules for each scale
        self.temporal_attentions = nn.ModuleList()
        for scale in self.scales:
            effective_frames = max(num_frames // scale, 2)  # Ensure at least 2 frames
            self.temporal_attentions.append(
                TemporalAttention(
                    feature_dim=feature_dim,
                    num_frames=effective_frames,
                    num_heads=num_heads
                )
            )
        
        # Scale fusion layer
        self.scale_fusion = nn.Conv2d(
            feature_dim * len(self.scales),
            feature_dim,
            kernel_size=1
        )
        self.scale_norm = nn.BatchNorm2d(feature_dim)
        
    def temporal_downsample(self, x, scale):
        """
        Temporal downsampling by averaging consecutive frames
        Args:
            x: (batch, frames, channels, height, width)
            scale: downsampling factor
        """
        if scale == 1:
            return x
            
        batch, frames, channels, height, width = x.shape
        
        # Pad frames if necessary
        pad_frames = (scale - frames % scale) % scale
        if pad_frames > 0:
            # Repeat last frame for padding
            last_frame = x[:, -1:].repeat(1, pad_frames, 1, 1, 1)
            x = torch.cat([x, last_frame], dim=1)
            frames += pad_frames
        
        # Reshape and average
        x = x.view(batch, frames // scale, scale, channels, height, width)
        x = x.mean(dim=2)  # Average over the scale dimension
        
        return x
    
    def temporal_upsample(self, x, target_frames):
        """
        Temporal upsampling using interpolation
        Args:
            x: (batch, frames, channels, height, width)
            target_frames: target number of frames
        """
        batch, frames, channels, height, width = x.shape
        
        if frames == target_frames:
            return x
            
        # Temporal interpolation
        x = x.permute(0, 2, 1, 3, 4)  # (batch, channels, frames, height, width)
        x = F.interpolate(x, size=(target_frames, height, width), mode='trilinear', align_corners=False)
        x = x.permute(0, 2, 1, 3, 4)  # Back to (batch, frames, channels, height, width)
        
        return x
    
    def forward(self, x):
        """
        Args:
            x: (batch, frames, channels, height, width)
        """
        batch, frames, channels, height, width = x.shape
        scale_features = []
        
        # Process each temporal scale
        for i, scale in enumerate(self.scales):
            # Downsample temporally
            x_downsampled = self.temporal_downsample(x, scale)
            
            # Apply temporal attention
            x_attended = self.temporal_attentions[i](x_downsampled)
            
            # Upsample back to original temporal resolution
            x_upsampled = self.temporal_upsample(x_attended, frames)
            
            # Average across temporal dimension for fusion
            x_aggregated = torch.mean(x_upsampled, dim=1)  # (batch, channels, height, width)
            scale_features.append(x_aggregated)
        
        # Fuse multi-scale features
        fused_features = torch.cat(scale_features, dim=1)  # (batch, channels*scales, height, width)
        output = self.scale_fusion(fused_features)
        output = self.scale_norm(output)
        
        return output


class HierarchicalTemporalAttention(nn.Module):
    """
    Hierarchical temporal attention that processes video at multiple temporal hierarchies
    - Frame-level: Individual frame processing
    - Sequence-level: Short sequence patterns (4-8 frames)
    - Snippet-level: Full snippet patterns (16 frames)
    """
    
    def __init__(self, feature_dim, num_frames=16, num_heads=8):
        super(HierarchicalTemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # Define hierarchical levels
        self.frame_level_size = 1
        self.sequence_level_size = 4
        self.snippet_level_size = num_frames
        
        # Frame-level processing (individual frames)
        self.frame_processor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Sequence-level temporal attention (4-frame groups)
        self.sequence_attention = TemporalAttention(
            feature_dim=feature_dim,
            num_frames=self.sequence_level_size,
            num_heads=num_heads // 2
        )
        
        # Snippet-level temporal attention (full 16 frames)
        self.snippet_attention = TemporalAttention(
            feature_dim=feature_dim,
            num_frames=self.snippet_level_size,
            num_heads=num_heads
        )
        
        # Hierarchical fusion
        self.hierarchy_fusion = nn.ModuleList([
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),  # Frame level
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),  # Sequence level
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),  # Snippet level
        ])
        
        # Final fusion
        self.final_fusion = nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1)
        self.final_norm = nn.BatchNorm2d(feature_dim)
        
    def process_frame_level(self, x):
        """Process individual frames"""
        batch, frames, channels, height, width = x.shape
        
        # Process each frame independently
        x_flat = x.view(-1, channels, height, width)  # (batch*frames, channels, h, w)
        processed = self.frame_processor(x_flat)
        processed = processed.view(batch, frames, channels, height, width)
        
        # Average across temporal dimension
        frame_features = torch.mean(processed, dim=1)  # (batch, channels, height, width)
        return frame_features
    
    def process_sequence_level(self, x):
        """Process 4-frame sequences"""
        batch, frames, channels, height, width = x.shape
        
        # Group frames into sequences of 4
        num_sequences = frames // self.sequence_level_size
        if frames % self.sequence_level_size != 0:
            # Pad to make divisible by sequence_level_size
            pad_frames = self.sequence_level_size - (frames % self.sequence_level_size)
            last_frame = x[:, -1:].repeat(1, pad_frames, 1, 1, 1)
            x = torch.cat([x, last_frame], dim=1)
            frames += pad_frames
            num_sequences = frames // self.sequence_level_size
        
        sequence_features = []
        for i in range(num_sequences):
            start_idx = i * self.sequence_level_size
            end_idx = start_idx + self.sequence_level_size
            sequence = x[:, start_idx:end_idx]  # (batch, 4, channels, h, w)
            
            # Apply temporal attention to sequence
            attended_sequence = self.sequence_attention(sequence)
            # Average across temporal dimension
            sequence_feat = torch.mean(attended_sequence, dim=1)  # (batch, channels, h, w)
            sequence_features.append(sequence_feat)
        
        # Average across all sequences
        sequence_features = torch.stack(sequence_features, dim=0)  # (num_sequences, batch, channels, h, w)
        sequence_features = torch.mean(sequence_features, dim=0)  # (batch, channels, h, w)
        
        return sequence_features
    
    def process_snippet_level(self, x):
        """Process full snippet (16 frames)"""
        # Apply temporal attention to full snippet
        attended_snippet = self.snippet_attention(x)
        # Average across temporal dimension
        snippet_features = torch.mean(attended_snippet, dim=1)  # (batch, channels, height, width)
        return snippet_features
    
    def forward(self, x):
        """
        Args:
            x: (batch, frames, channels, height, width)
        """
        # Process at different hierarchical levels
        frame_features = self.process_frame_level(x)
        sequence_features = self.process_sequence_level(x)
        snippet_features = self.process_snippet_level(x)
        
        # Apply level-specific fusion
        frame_fused = self.hierarchy_fusion[0](frame_features)
        sequence_fused = self.hierarchy_fusion[1](sequence_features)
        snippet_fused = self.hierarchy_fusion[2](snippet_features)
        
        # Combine all hierarchical features
        combined_features = torch.cat([frame_fused, sequence_fused, snippet_fused], dim=1)
        
        # Final fusion
        output = self.final_fusion(combined_features)
        output = self.final_norm(output)
        
        return output


class AdaptiveTemporalPooling(nn.Module):
    """
    Adaptive temporal pooling that adjusts temporal receptive field based on content
    """
    
    def __init__(self, feature_dim, num_frames, pool_sizes=[2, 4, 8]):
        super(AdaptiveTemporalPooling, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.pool_sizes = pool_sizes
        
        # Attention-based pool size selection
        self.pool_selector = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global spatiotemporal pooling
            nn.Flatten(),
            nn.Linear(feature_dim, len(pool_sizes)),
            nn.Softmax(dim=1)
        )
        
        # Different temporal pooling operations
        self.temporal_pools = nn.ModuleList()
        for pool_size in pool_sizes:
            if pool_size <= num_frames:
                self.temporal_pools.append(
                    nn.AdaptiveAvgPool3d((num_frames // pool_size, None, None))
                )
            else:
                # If pool_size > num_frames, use identity
                self.temporal_pools.append(nn.Identity())
        
        # Feature fusion after pooling
        self.pool_fusion = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: (batch, frames, channels, height, width)
        """
        batch, frames, channels, height, width = x.shape
        
        # Prepare for 3D operations: (batch, channels, frames, height, width)
        x_3d = x.permute(0, 2, 1, 3, 4)
        
        # Determine adaptive pool weights
        pool_weights = self.pool_selector(x_3d)  # (batch, num_pool_sizes)
        
        # Apply different temporal pooling strategies
        pooled_features = []
        for i, pool_op in enumerate(self.temporal_pools):
            if isinstance(pool_op, nn.Identity):
                pooled = x_3d
            else:
                pooled = pool_op(x_3d)
                # Interpolate back to original temporal resolution if needed
                if pooled.shape[2] != frames:
                    pooled = F.interpolate(
                        pooled, 
                        size=(frames, height, width), 
                        mode='trilinear', 
                        align_corners=False
                    )
            pooled_features.append(pooled)
        
        # Weighted combination of pooled features
        adaptive_features = torch.zeros_like(x_3d)
        for i, pooled in enumerate(pooled_features):
            weight = pool_weights[:, i:i+1, None, None, None]  # (batch, 1, 1, 1, 1)
            adaptive_features += weight * pooled
        
        # Convert back to original format and aggregate temporally
        adaptive_features = adaptive_features.permute(0, 2, 1, 3, 4)  # (batch, frames, channels, h, w)
        output = torch.mean(adaptive_features, dim=1)  # (batch, channels, height, width)
        
        # Final processing
        output = self.pool_fusion(output)
        
        return output


class EnhancedTemporalFusion(nn.Module):
    """
    Enhanced temporal fusion combining all temporal attention mechanisms
    """
    
    def __init__(self, feature_dim, num_frames=16, num_heads=8):
        super(EnhancedTemporalFusion, self).__init__()
        
        # Multi-scale temporal attention
        self.multiscale_attention = MultiScaleTemporalAttention(
            feature_dim=feature_dim,
            num_frames=num_frames,
            num_heads=num_heads
        )
        
        # Hierarchical temporal attention  
        self.hierarchical_attention = HierarchicalTemporalAttention(
            feature_dim=feature_dim,
            num_frames=num_frames,
            num_heads=num_heads
        )
        
        # Adaptive temporal pooling
        self.adaptive_pooling = AdaptiveTemporalPooling(
            feature_dim=feature_dim,
            num_frames=num_frames
        )
        
        # Final fusion of all temporal mechanisms
        self.final_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, frames, channels, height, width)
        """
        # Apply different temporal attention mechanisms
        multiscale_features = self.multiscale_attention(x)
        hierarchical_features = self.hierarchical_attention(x)
        adaptive_features = self.adaptive_pooling(x)
        
        # Combine all temporal features
        combined_features = torch.cat([
            multiscale_features,
            hierarchical_features,
            adaptive_features
        ], dim=1)
        
        # Final fusion
        output = self.final_fusion(combined_features)
        
        return output


class OpticalFlowFeatureFusion(nn.Module):
    """
    Optical Flow Feature Fusion Module
    
    Analyzes motion patterns using optical flow and creates motion-aware 
    feature representations for enhanced anomaly detection.
    
    Similar to TemporalFeatureFusion but focuses on motion dynamics:
    1. Computes optical flow between consecutive frames
    2. Analyzes motion patterns across temporal sequence  
    3. Extracts motion features that capture anomalous movements
    4. Enhances input streams with motion-aware context
    """
    
    def __init__(self, feature_dim=3, num_frames=8):
        super(OpticalFlowFeatureFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # FlowNet for optical flow computation
        self.flow_net = FlowNetSD(args=None, batchNorm=True)
          # Motion pattern analysis layers
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, 32),  # Use GroupNorm for better stability with small batches
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, 64),  # Use GroupNorm for better stability with small batches
            nn.ReLU(inplace=True)
        )
        
        # Temporal motion aggregation
        self.temporal_motion_pool = nn.AdaptiveAvgPool3d((1, None, None))
          # Motion feature compression to match input channels
        self.motion_compress = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),  # Use GroupNorm instead of BatchNorm for better stability
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure values are between 0-1 for fusion
        )
        
        # Motion attention mechanism
        self.motion_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Motion magnitude analysis for anomaly sensitivity
        self.magnitude_analyzer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
          # Learnable fusion parameters
        self.motion_weight = nn.Parameter(torch.tensor(0.2))  # Start with 20% motion influence
        self.magnitude_weight = nn.Parameter(torch.tensor(0.1))  # 10% magnitude influence
        
    def compute_optical_flow(self, frames):
        """
        Compute optical flow between consecutive frames
        
        Args:
            frames: (batch, num_frames, channels, height, width)
            
        Returns:
            flow_sequence: (batch, num_frames-1, 2, height, width)
        """
        batch_size, num_frames, channels, height, width = frames.shape
        flow_list = []
        
        for t in range(num_frames - 1):
            # Get consecutive frame pairs
            frame1 = frames[:, t]     # (batch, channels, H, W)
            frame2 = frames[:, t + 1] # (batch, channels, H, W)
            
            # Concatenate frames for FlowNet input (expects 6 channels: RGB + RGB)
            flow_input = torch.cat([frame1, frame2], dim=1)  # (batch, 6, H, W)
            
            # Handle potential size mismatch by ensuring dimensions are compatible
            # FlowNet often expects inputs to be divisible by 64 for proper feature map alignment
            orig_h, orig_w = flow_input.shape[2], flow_input.shape[3]
            
            # Pad to nearest multiple of 64 if needed
            pad_h = (64 - orig_h % 64) % 64
            pad_w = (64 - orig_w % 64) % 64
            
            if pad_h > 0 or pad_w > 0:
                flow_input = F.pad(flow_input, (0, pad_w, 0, pad_h), mode='reflect')            # Compute optical flow
            try:
                # Use evaluation mode to avoid potential BatchNorm issues during testing
                was_training = self.flow_net.training
                self.flow_net.eval()
                
                with torch.no_grad():  # FlowNet can be frozen for initial testing
                    flow_output = self.flow_net(flow_input)
                    if isinstance(flow_output, tuple):
                        flow = flow_output[0]  # Get the finest flow prediction
                    else:
                        flow = flow_output
                
                # FlowNetSD outputs flow at 1/4 resolution, upsample to match input
                if flow.shape[2] != orig_h or flow.shape[3] != orig_w:
                    # Upsample flow to match input resolution
                    scale_h = orig_h / flow.shape[2]
                    scale_w = orig_w / flow.shape[3]
                    
                    flow = F.interpolate(flow, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                    
                    # Scale flow values according to the upsampling ratio
                    flow[:, 0] = flow[:, 0] * scale_w  # Scale x-component
                    flow[:, 1] = flow[:, 1] * scale_h  # Scale y-component
                
                # Restore training mode
                if was_training:
                    self.flow_net.train()
                
                # Crop back to original size if we padded
                if pad_h > 0 or pad_w > 0:
                    flow = flow[:, :, :orig_h, :orig_w]
                    
            except Exception as e:
                # Fallback: simple frame difference as motion proxy
                print(f"Warning: FlowNet failed ({e}), using simple motion estimation")
                frame_diff = frame2 - frame1
                
                # Create 2-channel flow from frame difference
                # Use spatial gradients as rough flow approximation
                flow = torch.zeros(batch_size, 2, orig_h, orig_w, 
                                 device=frames.device, dtype=frames.dtype)
                
                # Simple motion estimation from frame difference
                if frame_diff.shape[1] >= 3:  # RGB channels
                    # Use average difference as motion magnitude
                    motion_mag = torch.mean(torch.abs(frame_diff), dim=1, keepdim=True)
                    flow[:, 0:1] = motion_mag  # x-component
                    flow[:, 1:2] = motion_mag * 0.5  # y-component (scaled)
            
            flow_list.append(flow)
          # Stack flows: (batch, num_frames-1, 2, H, W)
        flow_sequence = torch.stack(flow_list, dim=1)
        return flow_sequence
    
    def analyze_motion_patterns(self, flow_sequence):
        """
        Analyze motion patterns from optical flow sequence
        
        Args:
            flow_sequence: (batch, num_frames-1, 2, height, width)
            
        Returns:
            motion_features: (batch, feature_dim, height, width)
            motion_magnitude: (batch, 1, height, width)
        """
        batch_size, num_frames_minus_1, flow_channels, height, width = flow_sequence.shape
        
        # Add singleton dimension for 3D conv: (batch, 2, num_frames-1, H, W)
        flow_3d = flow_sequence.permute(0, 2, 1, 3, 4)
        
        # Extract motion features through 3D convolution
        motion_features_3d = self.motion_encoder(flow_3d)  # (batch, 64, num_frames-1, H, W)
        
        # Temporal aggregation - compress temporal dimension
        motion_features_pooled = self.temporal_motion_pool(motion_features_3d)  # (batch, 64, 1, H, W)
        motion_features_2d = motion_features_pooled.squeeze(2)  # (batch, 64, H, W)
        
        # Resize motion features to exactly match input dimensions if needed
        if motion_features_2d.shape[-2:] != (height, width):
            motion_features_2d = F.interpolate(
                motion_features_2d, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Compress to match input feature dimensions
        motion_features = self.motion_compress(motion_features_2d)  # (batch, feature_dim, H, W)
        
        # Ensure motion features exactly match input dimensions
        if motion_features.shape[-2:] != (height, width):
            motion_features = F.interpolate(
                motion_features, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Compute motion magnitude for anomaly sensitivity
        flow_magnitude = torch.sqrt(flow_sequence[:, :, 0] ** 2 + flow_sequence[:, :, 1] ** 2)  # (batch, num_frames-1, H, W)
        avg_magnitude = torch.mean(flow_magnitude, dim=1, keepdim=True)  # (batch, 1, H, W)
        
        # Ensure magnitude matches input dimensions
        if avg_magnitude.shape[-2:] != (height, width):
            avg_magnitude = F.interpolate(
                avg_magnitude, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
            
        motion_magnitude = self.magnitude_analyzer(avg_magnitude)  # (batch, 1, H, W)
        
        # Final check to ensure exact dimension match
        if motion_magnitude.shape[-2:] != (height, width):
            motion_magnitude = F.interpolate(
                motion_magnitude, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
        
        return motion_features, motion_magnitude
    
    def forward(self, input_lap, input_res):
        """
        Args:
            input_lap: Laplacian stream (batch, num_frames, channels, height, width)
            input_res: Residual stream (batch, num_frames, channels, height, width)
            
        Returns:
            Enhanced input streams with motion-aware features
        """        # Store original inputs and dimensions
        self.input_lap = input_lap
        self.input_res = input_res
        batch_size, num_frames, channels, height, width = input_lap.shape
        
        # Combine streams for motion analysis (similar to TemporalFeatureFusion)
        combined = input_lap + input_res  # (batch, num_frames, channels, H, W)
        
        # Compute optical flow between consecutive frames
        flow_sequence = self.compute_optical_flow(combined)  # (batch, num_frames-1, 2, H, W)
        
        # Analyze motion patterns
        motion_features, motion_magnitude = self.analyze_motion_patterns(flow_sequence)
        
        # Ensure motion features match input dimensions exactly
        if motion_features.shape[-2:] != (height, width):
            motion_features = F.interpolate(
                motion_features, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
        
        if motion_magnitude.shape[-2:] != (height, width):
            motion_magnitude = F.interpolate(
                motion_magnitude, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply motion attention
        motion_attention_weights = self.motion_attention(motion_features)  # (batch, feature_dim)
        motion_attention_weights = motion_attention_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, feature_dim, 1, 1)
        attended_motion_features = motion_features * motion_attention_weights
        
        # Combine motion features with magnitude awareness
        motion_enhanced = attended_motion_features * (1 + self.magnitude_weight * motion_magnitude)
        
        # Broadcast motion features to all frames (similar to TemporalFeatureFusion)
        motion_broadcasted = motion_enhanced.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
        
        # Enhance input streams with motion-aware features
        motion_weight_clamped = torch.clamp(self.motion_weight, 0.0, 1.0)  # Ensure valid range
        
        input_lap_enhanced = self.input_lap + motion_weight_clamped * motion_broadcasted
        input_res_enhanced = self.input_res + motion_weight_clamped * motion_broadcasted
        
        return input_lap_enhanced, input_res_enhanced
    
    def get_motion_info(self):
        """
        Returns motion analysis information for debugging/visualization
        """
        return {
            'motion_weight': self.motion_weight.item(),
            'magnitude_weight': self.magnitude_weight.item(),
        }
