import torch
import torch.nn as nn
import torch.nn.functional as F


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
                    dim=1                )
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
