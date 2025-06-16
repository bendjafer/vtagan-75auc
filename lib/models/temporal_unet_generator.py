"""
Enhanced U-Net Generator with Temporal Attention Integration
Integrates temporal attention modules into the U-Net skip connections
for improved video anomaly detection performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from lib.models.temporal_attention import TemporalAttention
from lib.models.multiscale_temporal_attention import (
    MultiScaleTemporalAttention, 
    HierarchicalTemporalAttention,
    EnhancedTemporalFusion
)


class TemporalSkipConnectionBlock(nn.Module):
    """
    Enhanced U-Net skip connection block with temporal attention
    Processes video sequences with temporal consistency
    """
    
    def __init__(self, layer_num, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False, 
                 num_frames=16, use_temporal_attention=True):
        super(TemporalSkipConnectionBlock, self).__init__()
        
        self.outermost = outermost
        self.innermost = innermost
        self.layer_num = layer_num
        self.use_dropout = use_dropout
        self.submodule = submodule
        self.num_frames = num_frames
        self.use_temporal_attention = use_temporal_attention
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        if input_nc is None:
            input_nc = outer_nc
        
        # Downsampling path - Laplacian stream
        self.downconv_lap = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                     stride=2, padding=1, bias=use_bias)
        self.downrelu_lap = nn.LeakyReLU(0.2, True)
        self.downnorm_lap = norm_layer(inner_nc)
        
        # Downsampling path - Residual stream  
        self.downconv_res = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                     stride=2, padding=1, bias=use_bias)
        self.downrelu_res = nn.LeakyReLU(0.2, True)
        self.downnorm_res = norm_layer(inner_nc)
        
        # Upsampling path components
        self.uprelu_lap = nn.ReLU(True)
        self.upnorm_lap = norm_layer(outer_nc)
        self.uprelu_res = nn.ReLU(True) 
        self.upnorm_res = norm_layer(outer_nc)
        
        # Temporal attention modules for enhanced skip connections
        if self.use_temporal_attention and not innermost:
            # Multi-scale temporal attention for skip features
            self.temporal_attention_skip = MultiScaleTemporalAttention(
                feature_dim=inner_nc,
                num_frames=num_frames,
                num_heads=8
            )
            
            # Hierarchical attention for different temporal scales
            self.hierarchical_attention = HierarchicalTemporalAttention(
                feature_dim=inner_nc,
                num_frames=num_frames,
                num_heads=8
            )
            
            # Enhanced fusion for combining temporal features
            if layer_num <= 2:  # Use enhanced fusion for deeper layers
                self.enhanced_fusion = EnhancedTemporalFusion(
                    feature_dim=inner_nc,
                    num_frames=num_frames,
                    num_heads=8
                )
        
        # Channel Shuffling for feature interaction
        self.channel_shuffle = ChannelShuffle(inner_nc, reduction=2)
        
        # Configure layer-specific architectures
        if outermost:
            # Outermost layer
            self.upconv_lap = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                               kernel_size=4, stride=2, padding=1)
            self.upconv_res = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                               kernel_size=4, stride=2, padding=1)
            
        elif innermost:
            # Innermost layer (bottleneck)
            self.upconv_lap = nn.ConvTranspose2d(inner_nc, outer_nc,
                                               kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.upconv_res = nn.ConvTranspose2d(inner_nc, outer_nc,
                                               kernel_size=4, stride=2, padding=1, bias=use_bias)
            
            # Temporal attention in bottleneck for global temporal context
            if self.use_temporal_attention:
                self.bottleneck_temporal = TemporalAttention(
                    feature_dim=inner_nc,
                    num_frames=num_frames,
                    num_heads=8
                )
                
        else:
            # Intermediate layers
            self.upconv_lap = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                               kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.upconv_res = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                               kernel_size=4, stride=2, padding=1, bias=use_bias)
    
    def apply_temporal_attention(self, features, layer_type="skip"):
        """Apply temporal attention to features"""
        if not self.use_temporal_attention:
            return features
            
        # Features shape: (batch*frames, channels, height, width)
        batch_frames, channels, height, width = features.shape
        batch_size = batch_frames // self.num_frames
        
        # Reshape to video format: (batch, frames, channels, height, width)
        video_features = features.view(batch_size, self.num_frames, channels, height, width)
        
        if layer_type == "bottleneck" and hasattr(self, 'bottleneck_temporal'):
            # Apply temporal attention and reshape back
            attended_features = self.bottleneck_temporal(video_features)
            return attended_features.view(batch_frames, channels, height, width)
            
        elif layer_type == "skip" and hasattr(self, 'temporal_attention_skip'):
            # Apply multi-scale temporal attention
            if hasattr(self, 'enhanced_fusion') and self.layer_num <= 2:
                # Use enhanced fusion for deeper layers
                attended_features = self.enhanced_fusion(video_features)
            else:
                # Use hierarchical attention for other layers
                attended_features = self.hierarchical_attention(video_features)
            
            # attended_features is (batch, channels, height, width) - temporally aggregated
            # Expand back to all frames for skip connection
            attended_features = attended_features.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
            return attended_features.view(batch_frames, channels, height, width)
        
        return features
    
    def forward(self, input):
        """Forward pass with temporal attention integration"""
        input_lap, input_res = input
        
        if self.outermost:
            # Outermost layer processing
            d_lap = self.downrelu_lap(self.downconv_lap(input_lap))
            d_res = self.downrelu_res(self.downconv_res(input_res))
            
            # Apply channel shuffling for feature interaction
            d_lap, d_res = self.channel_shuffle((d_lap, d_res))
            
            # Apply temporal attention to downsampled features
            d_lap = self.apply_temporal_attention(d_lap, "skip")
            d_res = self.apply_temporal_attention(d_res, "skip")
            
            if self.submodule is None:
                u_lap = d_lap
                u_res = d_res
            else:
                u_lap, u_res = self.submodule((d_lap, d_res))
            
            # Upsampling with temporal consistency
            out_lap = torch.tanh(self.upconv_lap(u_lap))
            out_res = torch.tanh(self.upconv_res(u_res))
            
            return (out_lap, out_res)
            
        elif self.innermost:
            # Innermost layer (bottleneck) processing
            d_lap = self.downrelu_lap(self.downconv_lap(input_lap))
            d_res = self.downrelu_res(self.downconv_res(input_res))
            
            # Apply channel shuffling
            d_lap, d_res = self.channel_shuffle((d_lap, d_res))
            
            # Apply temporal attention in bottleneck for global context
            d_lap = self.apply_temporal_attention(d_lap, "bottleneck")
            d_res = self.apply_temporal_attention(d_res, "bottleneck")
            
            # Upsampling
            u_lap = self.upnorm_lap(self.upconv_lap(self.uprelu_lap(d_lap)))
            u_res = self.upnorm_res(self.upconv_res(self.uprelu_res(d_res)))
            
            return (u_lap, u_res)
            
        else:
            # Intermediate layer processing
            d_lap = self.downnorm_lap(self.downconv_lap(self.downrelu_lap(input_lap)))
            d_res = self.downnorm_res(self.downconv_res(self.downrelu_res(input_res)))
            
            # Apply channel shuffling
            d_lap, d_res = self.channel_shuffle((d_lap, d_res))
            
            # Apply temporal attention to downsampled features
            d_lap = self.apply_temporal_attention(d_lap, "skip")
            d_res = self.apply_temporal_attention(d_res, "skip")
            
            if self.submodule is None:
                u_lap = d_lap
                u_res = d_res
            else:
                u_lap, u_res = self.submodule((d_lap, d_res))
            
            # Upsampling with skip connections
            up_lap = self.upnorm_lap(self.upconv_lap(self.uprelu_lap(u_lap)))
            up_res = self.upnorm_res(self.upconv_res(self.uprelu_res(u_res)))
            
            # Apply dropout if specified
            if self.use_dropout:
                up_lap = F.dropout(up_lap, p=0.5, training=self.training)
                up_res = F.dropout(up_res, p=0.5, training=self.training)
            
            # Skip connections with temporal consistency
            out_lap = torch.cat([input_lap, up_lap], 1)
            out_res = torch.cat([input_res, up_res], 1)
            
            return (out_lap, out_res)


class ChannelShuffle(nn.Module):
    """
    Enhanced Channel Shuffling mechanism with temporal awareness
    """
    
    def __init__(self, features, reduction=2, num_frames=16):
        super(ChannelShuffle, self).__init__()
        
        self.features = features
        self.reduction = reduction
        self.num_frames = num_frames
        
        # Adaptive pooling for feature statistics
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers for attention computation
        d = max(int(features / reduction), 32)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([
            nn.Linear(d, features) for _ in range(2)
        ])
        self.softmax = nn.Softmax(dim=1)
        
        # Temporal context integration
        self.temporal_context = nn.Sequential(
            nn.Conv1d(features, features // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(features // 4, features, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x1, x2 = x
        x_combined = x1 + x2
        
        # Global average pooling
        batch_frames, channels, height, width = x_combined.shape
        fea_s = self.gap(x_combined).squeeze()  # (batch_frames, channels)
        
        # Handle different batch sizes
        if fea_s.dim() == 1:
            fea_s = fea_s.unsqueeze(0)
        
        # Temporal context modeling
        if batch_frames >= self.num_frames:
            # Reshape for temporal processing
            batch_size = batch_frames // self.num_frames
            temporal_features = fea_s.view(batch_size, self.num_frames, channels)
            temporal_features = temporal_features.permute(0, 2, 1)  # (batch, channels, frames)
            
            # Apply temporal context
            temporal_weights = self.temporal_context(temporal_features)  # (batch, channels, frames)
            temporal_weights = temporal_weights.permute(0, 2, 1)  # (batch, frames, channels)
            temporal_weights = temporal_weights.contiguous().view(batch_frames, channels)
            
            # Combine with spatial features
            fea_s = fea_s * temporal_weights
        
        # Feature transformation
        fea_z = self.fc(fea_s)
        
        # Attention computation for both streams
        attention_vectors = []
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(1)
            attention_vectors.append(vector)
        
        attention_vec = torch.cat(attention_vectors, dim=1)
        attention_vec = self.softmax(attention_vec)
        attention_vec = attention_vec.unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention weights
        out_x1 = x1 * attention_vec[:, 0:1]
        out_x2 = x2 * attention_vec[:, 1:2]
        
        return (out_x1, out_x2)


class TemporalUnetGenerator(nn.Module):
    """
    Enhanced U-Net Generator with Temporal Attention Integration
    """
    
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, 
                 num_frames=16, use_temporal_attention=True):
        super(TemporalUnetGenerator, self).__init__()
        
        self.num_frames = num_frames
        self.use_temporal_attention = use_temporal_attention
        
        # Construct temporal-aware U-Net structure
        unet_block = TemporalSkipConnectionBlock(
            layer_num=0, 
            outer_nc=ngf * 8, 
            inner_nc=ngf * 8, 
            input_nc=None, 
            submodule=None, 
            norm_layer=norm_layer, 
            innermost=True,
            num_frames=num_frames,
            use_temporal_attention=use_temporal_attention
        )
        
        # Add intermediate layers with temporal attention
        for i in range(num_downs - 5):
            unet_block = TemporalSkipConnectionBlock(
                layer_num=i + 1,
                outer_nc=ngf * 8, 
                inner_nc=ngf * 8, 
                input_nc=None, 
                submodule=unet_block, 
                norm_layer=norm_layer, 
                use_dropout=use_dropout,
                num_frames=num_frames,
                use_temporal_attention=use_temporal_attention
            )
        
        # Add upsampling layers
        unet_block = TemporalSkipConnectionBlock(
            layer_num=num_downs,
            outer_nc=ngf * 4, 
            inner_nc=ngf * 8, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer,
            num_frames=num_frames,
            use_temporal_attention=use_temporal_attention
        )
        
        unet_block = TemporalSkipConnectionBlock(
            layer_num=num_downs + 1,
            outer_nc=ngf * 2, 
            inner_nc=ngf * 4, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer,
            num_frames=num_frames,
            use_temporal_attention=use_temporal_attention
        )
        
        unet_block = TemporalSkipConnectionBlock(
            layer_num=num_downs + 2,
            outer_nc=ngf, 
            inner_nc=ngf * 2, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer,
            num_frames=num_frames,
            use_temporal_attention=use_temporal_attention
        )
        
        # Outermost layer
        unet_block = TemporalSkipConnectionBlock(
            layer_num=num_downs + 3,
            outer_nc=output_nc, 
            inner_nc=ngf, 
            input_nc=input_nc, 
            submodule=unet_block, 
            outermost=True, 
            norm_layer=norm_layer,
            num_frames=num_frames,
            use_temporal_attention=use_temporal_attention
        )
        
        self.model = unet_block
        
        # Global temporal consistency module
        if use_temporal_attention:
            self.global_temporal_consistency = nn.ModuleDict({
                'lap_consistency': nn.Sequential(
                    nn.Conv3d(output_nc, output_nc // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.BatchNorm3d(output_nc // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(output_nc // 2, output_nc, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.Sigmoid()
                ),
                'res_consistency': nn.Sequential(
                    nn.Conv3d(output_nc, output_nc // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.BatchNorm3d(output_nc // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(output_nc // 2, output_nc, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.Sigmoid()
                )
            })
    
    def apply_global_temporal_consistency(self, output_lap, output_res):
        """Apply global temporal consistency across entire sequence"""
        if not self.use_temporal_attention:
            return output_lap, output_res
        
        batch_frames, channels, height, width = output_lap.shape
        batch_size = batch_frames // self.num_frames
        
        # Reshape to 3D format: (batch, channels, frames, height, width)
        lap_3d = output_lap.view(batch_size, self.num_frames, channels, height, width)
        lap_3d = lap_3d.permute(0, 2, 1, 3, 4)
        
        res_3d = output_res.view(batch_size, self.num_frames, channels, height, width)
        res_3d = res_3d.permute(0, 2, 1, 3, 4)
        
        # Apply temporal consistency
        lap_consistency = self.global_temporal_consistency['lap_consistency'](lap_3d)
        res_consistency = self.global_temporal_consistency['res_consistency'](res_3d)
        
        # Apply consistency weights
        lap_consistent = lap_3d * lap_consistency
        res_consistent = res_3d * res_consistency
        
        # Reshape back to original format
        lap_consistent = lap_consistent.permute(0, 2, 1, 3, 4).contiguous().view(batch_frames, channels, height, width)
        res_consistent = res_consistent.permute(0, 2, 1, 3, 4).contiguous().view(batch_frames, channels, height, width)
        
        return lap_consistent, res_consistent
    
    def forward(self, input):
        """Forward pass with temporal attention integration"""
        output_lap, output_res = self.model(input)
        
        # Apply global temporal consistency
        output_lap, output_res = self.apply_global_temporal_consistency(output_lap, output_res)
        
        return output_lap, output_res


def define_temporal_G(opt, norm='batch', use_dropout=False, init_type='normal', num_frames=16):
    """
    Define temporal-aware generator network
    """
    from lib.models.networks import get_norm_layer, init_net
    import numpy as np
    
    norm_layer = get_norm_layer(norm_type=norm)
    num_layer = int(np.log2(opt.isize))
    
    netG = TemporalUnetGenerator(
        input_nc=opt.nc, 
        output_nc=opt.nc, 
        num_downs=num_layer, 
        ngf=opt.ngf, 
        norm_layer=norm_layer, 
        use_dropout=use_dropout,
        num_frames=num_frames,
        use_temporal_attention=getattr(opt, 'use_temporal_attention', True)
    )
    
    return init_net(netG, init_type, opt.gpu_ids)


# Integration helper functions
def enhance_existing_generator_with_temporal_attention(existing_generator, num_frames=16):
    """
    Enhance existing generator with temporal attention capabilities
    """
    # This function can be used to retrofit existing generators
    # Implementation would depend on the specific architecture
    pass


def temporal_generator_test():
    """Test function for temporal generator"""
    import torch
    from types import SimpleNamespace
    
    # Create mock options
    opt = SimpleNamespace()
    opt.nc = 3
    opt.nz = 100
    opt.ngf = 64
    opt.isize = 64
    opt.gpu_ids = [0] if torch.cuda.is_available() else []
    opt.use_temporal_attention = True
    
    # Create generator
    netG = define_temporal_G(opt, num_frames=16)
    
    # Test input
    batch_size, num_frames = 2, 16
    input_lap = torch.randn(batch_size * num_frames, 3, 64, 64)
    input_res = torch.randn(batch_size * num_frames, 3, 64, 64)
    
    if torch.cuda.is_available():
        netG = netG.cuda()
        input_lap = input_lap.cuda()
        input_res = input_res.cuda()
    
    # Forward pass
    with torch.no_grad():
        output_lap, output_res = netG((input_lap, input_res))
    
    print(f"✅ Temporal Generator Test Passed")
    print(f"   Input shape: {input_lap.shape}")
    print(f"   Output Lap shape: {output_lap.shape}")
    print(f"   Output Res shape: {output_res.shape}")
    
    return netG


if __name__ == "__main__":
    temporal_generator_test()
