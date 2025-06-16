import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.temporal_attention import TemporalAttention, ConvLSTM


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
