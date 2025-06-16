from collections import OrderedDict
from collections import OrderedDict
import os
import time
import numpy as np
# Use regular tqdm for compatibility
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lib.evaluation.visualizer import Visualizer
from lib.evaluation.loss import l2_loss
from lib.evaluation.evaluate import roc, pre_recall, save_curve
from lib.models.basemodel import BaseModel
from lib.models.attention_modules import TemporalAttention, TemporalFeatureFusion, OpticalFlowFeatureFusion
from lib.evaluation.loss import CombinedTemporalLoss
import pdb


class Gan_Model(BaseModel):
    """
    GAN for Video Anomaly Detection
    Processes video snippets (16 frames) instead of single images
    """

    def __init__(self, opt, data, classes):
        super(Gan_Model, self).__init__(opt, data, classes)
          # -- Misc attributes
        self.name = 'gan_model'
        self.classes = classes
        self.add_noise = True
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        self.num_frames = opt.num_frames if hasattr(opt, 'num_frames') else 16
          # Set temporal attention flag early
        self.use_temporal_attention = getattr(opt, 'use_temporal_attention', True)

        # Networks are already created in BaseModel
        # Just load weights if resuming
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'),strict=False)['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'),strict=False)['state_dict'])
            print("\tDone.\n")

        if self.opt.verbose:
            print(self.netg)
            print(self.netd)

        # Loss Functions
        self.l_adv = nn.BCELoss()
        self.l_con = nn.L1Loss()
        self.l_lat = l2_loss
          # Temporal loss for enhanced video processing
        if self.use_temporal_attention:
            self.temporal_loss = CombinedTemporalLoss(
                consistency_weight=getattr(opt, 'w_temporal_consistency', 0.1),
                motion_weight=getattr(opt, 'w_temporal_motion', 0.05),
                reg_weight=getattr(opt, 'w_temporal_reg', 0.01)
            )
            # Apply device placement and DataParallel if needed
            if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] >= 0:
                self.temporal_loss = self.temporal_loss.to(opt.gpu_ids[0])
                if len(opt.gpu_ids) > 1:
                    self.temporal_loss = torch.nn.DataParallel(self.temporal_loss, opt.gpu_ids)
            else:
                self.temporal_loss = self.temporal_loss.to(self.device)
            print("✅ Temporal loss functions initialized")
        
        # Real and fake labels
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        
        # Video-specific tensor attributes
        self.fake = torch.FloatTensor().to(self.device)
        self.fake_lap = torch.FloatTensor().to(self.device)
        self.fake_res = torch.FloatTensor().to(self.device)
        self.fake_aug = torch.FloatTensor().to(self.device)        # Override noise tensor with correct dimensions for video (3 channels instead of nz)
        self.noise = torch.FloatTensor(self.opt.batchsize, self.num_frames, 3, self.opt.isize, self.opt.isize).to(self.device)
          # Temporal attention modules for enhanced video processing
        if self.use_temporal_attention:
            # Calculate compatible number of heads for the feature dimension
            # Find the largest divisor of opt.nz that's <= 8 for optimal performance
            possible_heads = [8, 4, 2, 1]
            num_heads = next(h for h in possible_heads if opt.nz % h == 0)
            
            print(f"🔧 Using {num_heads} attention heads for feature_dim={opt.nz}")
              # Temporal attention for generator features (at bottleneck)
            self.temporal_attention_gen = TemporalAttention(
                feature_dim=opt.nz, 
                num_frames=self.num_frames, 
                num_heads=num_heads
            )
            # Apply device placement and DataParallel if needed
            if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] >= 0:
                self.temporal_attention_gen = self.temporal_attention_gen.to(opt.gpu_ids[0])
                if len(opt.gpu_ids) > 1:
                    self.temporal_attention_gen = torch.nn.DataParallel(self.temporal_attention_gen, opt.gpu_ids)
            else:
                self.temporal_attention_gen = self.temporal_attention_gen.to(self.device)
            
            # Temporal attention for discriminator features  
            self.temporal_attention_disc = TemporalAttention(
                feature_dim=opt.nz,
                num_frames=self.num_frames,
                num_heads=num_heads
            )
            # Apply device placement and DataParallel if needed
            if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] >= 0:
                self.temporal_attention_disc = self.temporal_attention_disc.to(opt.gpu_ids[0])
                if len(opt.gpu_ids) > 1:
                    self.temporal_attention_disc = torch.nn.DataParallel(self.temporal_attention_disc, opt.gpu_ids)
            else:
                self.temporal_attention_disc = self.temporal_attention_disc.to(self.device)
              # Multi-scale temporal fusion for input features
            self.temporal_fusion = TemporalFeatureFusion(
                feature_dim=3,  # RGB channels
                num_frames=self.num_frames
            )
            # Apply device placement and DataParallel if needed
            if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] >= 0:
                self.temporal_fusion = self.temporal_fusion.to(opt.gpu_ids[0])
                if len(opt.gpu_ids) > 1:
                    self.temporal_fusion = torch.nn.DataParallel(self.temporal_fusion, opt.gpu_ids)
            else:
                self.temporal_fusion = self.temporal_fusion.to(self.device)
            
            # Optical Flow Feature Fusion for motion analysis
            self.use_optical_flow = getattr(opt, 'use_optical_flow', True)  # Enable by default
            if self.use_optical_flow:
                self.optical_flow_fusion = OpticalFlowFeatureFusion(
                    feature_dim=3,  # RGB channels
                    num_frames=self.num_frames
                )
                # Apply device placement and DataParallel if needed
                if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] >= 0:
                    self.optical_flow_fusion = self.optical_flow_fusion.to(opt.gpu_ids[0])
                    if len(opt.gpu_ids) > 1:
                        self.optical_flow_fusion = torch.nn.DataParallel(self.optical_flow_fusion, opt.gpu_ids)
                else:
                    self.optical_flow_fusion = self.optical_flow_fusion.to(self.device)
                print("✅ Optical Flow Feature Fusion initialized")
            
            print(f"✅ Temporal attention modules initialized for {self.num_frames} frames")
          # Temporal loss for regularization
        self.use_temporal_loss = getattr(opt, 'use_temporal_loss', False)
        if self.use_temporal_loss:
            self.temporal_loss = CombinedTemporalLoss(
                in_channels=opt.nz,
                num_frames=self.num_frames
            )
            # Apply device placement and DataParallel if needed
            if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] >= 0:
                self.temporal_loss = self.temporal_loss.to(opt.gpu_ids[0])
                if len(opt.gpu_ids) > 1:
                    self.temporal_loss = torch.nn.DataParallel(self.temporal_loss, opt.gpu_ids)
            else:
                self.temporal_loss = self.temporal_loss.to(self.device)
            print("✅ Temporal loss module initialized")
          # Additional setup for video model
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            if self.use_temporal_attention:
                self.temporal_attention_gen.train()
                self.temporal_attention_disc.train()
                self.temporal_fusion.train()
                if hasattr(self, 'optical_flow_fusion'):
                    self.optical_flow_fusion.train()
                
                # Update optimizers to include temporal attention parameters
                self._update_optimizers_with_temporal_params()

    def _update_optimizers_with_temporal_params(self):
        """Update optimizers to include temporal attention parameters"""        # Collect all parameters for generator (including temporal modules)
        g_params = list(self.netg.parameters())
        if hasattr(self, 'temporal_attention_gen'):
            g_params.extend(list(self.temporal_attention_gen.parameters()))
        if hasattr(self, 'temporal_fusion'):
            g_params.extend(list(self.temporal_fusion.parameters()))
        if hasattr(self, 'optical_flow_fusion'):
            g_params.extend(list(self.optical_flow_fusion.parameters()))
        
        # Collect all parameters for discriminator (including temporal modules)
        d_params = list(self.netd.parameters())
        if hasattr(self, 'temporal_attention_disc'):
            d_params.extend(list(self.temporal_attention_disc.parameters()))
        
        # Add temporal loss parameters if using temporal loss
        if hasattr(self, 'temporal_loss'):
            # Note: CombinedTemporalLoss contains learnable parameters (Sobel filters)
            # but they're frozen, so no need to include them in optimization
            pass
            
        # Recreate optimizers with temporal parameters
        self.optimizer_g = optim.Adam(g_params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(d_params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        
        # Update schedulers
        from lib.models.networks import get_scheduler
        self.schedulers = [get_scheduler(self.optimizer_g, self.opt), get_scheduler(self.optimizer_d, self.opt)]
        self.optimizers = [self.optimizer_g, self.optimizer_d]
        
        print("✅ Optimizers updated with temporal attention parameters")

    def get_errors(self):
        """Get current error values including temporal losses"""
        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_lat', self.err_g_lat.item())
        ])        # Add temporal loss if available
        if hasattr(self, 'err_g_temporal') and self.err_g_temporal is not None:
            errors['err_g_temporal'] = self.err_g_temporal.item()
            
        return errors
    
    def forward(self):
        self.forward_g()
        self.forward_d()
    
    def forward_g(self):
        """Forward propagate through netG for video frames with temporal attention"""
        # Reshape from (batch, frames, channels, h, w) to (batch*frames, channels, h, w)
        batch_size, num_frames = self.input_lap.shape[:2]
        
        # Dynamically resize noise to match input batch size
        if self.noise.shape[0] != batch_size or self.noise.shape[1] != num_frames:
            self.noise = torch.randn(batch_size, num_frames, 3, self.opt.isize, self.opt.isize).to(self.device)
          # Apply temporal attention to input features if enabled
        if self.use_temporal_attention:
            # Combine laplacian and residual for temporal fusion
            combined_input = self.input_lap + self.input_res  # (batch, frames, 3, h, w)
            temporal_fused = self.temporal_fusion(combined_input)  # (batch, 3, h, w)
            
            # Expand temporal fused features back to all frames
            temporal_fused = temporal_fused.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)  # (batch, frames, 3, h, w)
              # Apply optical flow enhancement if enabled
            if hasattr(self, 'optical_flow_fusion') and self.use_optical_flow:
                # Apply optical flow feature fusion for motion-aware enhancement
                input_lap_flow_enhanced, input_res_flow_enhanced = self.optical_flow_fusion(
                    self.input_lap, self.input_res
                )
                
                # Combine temporal and motion features
                input_lap_enhanced = input_lap_flow_enhanced + 0.1 * temporal_fused  # Motion + Temporal
                input_res_enhanced = input_res_flow_enhanced + 0.1 * temporal_fused
            else:
                # Use only temporal-enhanced features for generation
                input_lap_enhanced = self.input_lap + 0.1 * temporal_fused  # Residual connection
                input_res_enhanced = self.input_res + 0.1 * temporal_fused
        else:
            input_lap_enhanced = self.input_lap
            input_res_enhanced = self.input_res
        
        input_lap_flat = input_lap_enhanced.view(-1, *input_lap_enhanced.shape[2:])  # (batch*frames, 3, h, w)
        input_res_flat = input_res_enhanced.view(-1, *input_res_enhanced.shape[2:])  # (batch*frames, 3, h, w)
        noise_flat = self.noise.view(-1, *self.noise.shape[2:])  # (batch*frames, 3, h, w)
        
        # Process all frames through the generator
        fake_lap_flat, fake_res_flat = self.netg((input_lap_flat + noise_flat, input_res_flat + noise_flat))
        
        # Reshape back to video format
        self.fake_lap = fake_lap_flat.view(batch_size, num_frames, *fake_lap_flat.shape[1:])
        self.fake_res = fake_res_flat.view(batch_size, num_frames, *fake_res_flat.shape[1:])
        self.fake = self.fake_lap + self.fake_res
          # Create augmented fake for discriminator (same as fake for now)
        self.fake_aug = self.fake.detach().clone()
        
    def forward_d(self):
        """Forward propagate through netD for video frames with temporal attention"""
        # Reshape and process frames
        batch_size, num_frames = self.input_lap.shape[:2]
        
        # Flatten for processing
        real_flat = (self.input_lap + self.input_res).view(-1, *self.input_lap.shape[2:])
        fake_flat = self.fake.view(-1, *self.fake.shape[2:])
        fake_aug_flat = self.fake_aug.view(-1, *self.fake_aug.shape[2:])
        
        # Process through discriminator
        self.pred_real, feat_real_flat = self.netd(real_flat)
        self.pred_fake, feat_fake_flat = self.netd(fake_flat)
        self.pred_fake_aug, feat_fake_aug_flat = self.netd(fake_aug_flat)
        
        # Reshape features back to video format
        feat_real_video = feat_real_flat.view(batch_size, num_frames, *feat_real_flat.shape[1:])
        feat_fake_video = feat_fake_flat.view(batch_size, num_frames, *feat_fake_flat.shape[1:])
        feat_fake_aug_video = feat_fake_aug_flat.view(batch_size, num_frames, *feat_fake_aug_flat.shape[1:])
        
        # Apply temporal attention to discriminator features if enabled
        if self.use_temporal_attention:
            # Apply temporal attention to enhance feature consistency across frames
            feat_real_attended = self.temporal_attention_disc(feat_real_video)
            feat_fake_attended = self.temporal_attention_disc(feat_fake_video)
            feat_fake_aug_attended = self.temporal_attention_disc(feat_fake_aug_video)
            
            # Use attended features for better temporal consistency
            self.feat_real = feat_real_attended
            self.feat_fake = feat_fake_attended
            self.feat_fake_aug = feat_fake_aug_attended
        else:
            self.feat_real = feat_real_video
            self.feat_fake = feat_fake_video
            self.feat_fake_aug = feat_fake_aug_video
    
    def backward_g(self):
        """Backpropagate netg for video"""
        # Flatten predictions for loss calculation
        pred_fake_flat = self.pred_fake.view(-1)
        real_label_expanded = self.real_label.repeat_interleave(self.num_frames)
        
        self.err_g_adv = self.opt.w_adv * self.l_adv(pred_fake_flat, real_label_expanded)
        
        # Reconstruction loss over all frames
        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input_lap + self.input_res)
        
        # Feature matching loss averaged over all frames
        feat_real_flat = self.feat_real.view(-1, *self.feat_real.shape[-1:])
        feat_fake_flat = self.feat_fake.view(-1, *self.feat_fake.shape[-1:])
        self.err_g_lat = self.opt.w_lat * self.l_lat(feat_fake_flat, feat_real_flat)

        # Initialize total generator loss
        self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat
        
        # Temporal loss if enabled
        if self.use_temporal_attention:
            # Calculate temporal loss on video sequences
            real_input = self.input_lap + self.input_res  # Ground truth video
            temporal_losses = self.temporal_loss(
                real_frames=real_input,
                fake_frames=self.fake,
                features=self.feat_fake
            )
            self.err_g_temporal = temporal_losses['total_temporal']
            self.err_g = self.err_g + self.err_g_temporal
        
        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        """Backpropagate netd for video"""
        # Flatten predictions for loss calculation
        pred_fake_flat = self.pred_fake.view(-1)
        pred_fake_aug_flat = self.pred_fake_aug.view(-1)
        pred_real_flat = self.pred_real.view(-1)
        
        # Get actual batch size from predictions
        batch_size = pred_fake_flat.shape[0] // self.num_frames
        
        # Dynamically create labels with correct batch size
        if self.fake_label.shape[0] != batch_size:
            self.fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32, device=self.device)
            self.real_label = torch.ones(size=(batch_size,), dtype=torch.float32, device=self.device)
        
        fake_label_expanded = self.fake_label.repeat_interleave(self.num_frames)
        real_label_expanded = self.real_label.repeat_interleave(self.num_frames)
        
        # Fake losses
        self.err_d_fake = self.l_adv(pred_fake_flat, fake_label_expanded)
        self.err_d_fake_aug = self.l_adv(pred_fake_aug_flat, fake_label_expanded)
        
        # Real loss
        self.err_d_real = self.l_adv(pred_real_flat, real_label_expanded)        # Combine losses
        self.err_d = self.err_d_real + self.err_d_fake + self.err_d_fake_aug
        self.err_d.backward(retain_graph=True)

    def update_netg(self):
        """Update Generator Network with gradient clipping."""       
        self.optimizer_g.zero_grad()
        self.backward_g()
        
        # Add gradient clipping if enabled
        if hasattr(self.opt, 'grad_clip_norm') and self.opt.grad_clip_norm > 0:
            # Clip gradients for main generator
            torch.nn.utils.clip_grad_norm_(self.netg.parameters(), self.opt.grad_clip_norm)
            
            # Clip gradients for temporal modules if they exist
            if hasattr(self, 'temporal_attention_gen'):
                torch.nn.utils.clip_grad_norm_(self.temporal_attention_gen.parameters(), self.opt.grad_clip_norm)
            if hasattr(self, 'temporal_fusion'):
                torch.nn.utils.clip_grad_norm_(self.temporal_fusion.parameters(), self.opt.grad_clip_norm)
        
        self.optimizer_g.step()

    def update_netd(self):
        """Update Discriminator Network with gradient clipping."""       
        self.optimizer_d.zero_grad()
        self.backward_d()
        
        # Add gradient clipping if enabled
        if hasattr(self.opt, 'grad_clip_norm') and self.opt.grad_clip_norm > 0:
            # Clip gradients for main discriminator
            torch.nn.utils.clip_grad_norm_(self.netd.parameters(), self.opt.grad_clip_norm)
            
            # Clip gradients for temporal modules if they exist
            if hasattr(self, 'temporal_attention_disc'):
                torch.nn.utils.clip_grad_norm_(self.temporal_attention_disc.parameters(), self.opt.grad_clip_norm)
        
        self.optimizer_d.step()
        if self.err_d < 1e-5: 
            self.reinit_d()

    def optimize_params(self):
        """Optimize netD and netG networks."""
        self.forward()
        self.update_netg()
        self.update_netd()

    def test(self, plot_hist=False):
        """Test model for video snippets."""
        self.netg.eval()
        self.netd.eval()
        
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                self.load_weights(is_best=True)

            self.opt.phase = 'test'

            scores = {}

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.features = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            print("   Testing %s" % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
              # Add progress bar for testing
            test_bar = tqdm(enumerate(self.data.valid, 0), 
                          total=len(self.data.valid),
                          desc=f"Testing {self.name}",
                          unit="batch")
            
            for i, data in test_bar:
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()                # Forward - Pass
                self.set_input(data)
                
                # Dynamically resize noise to match input batch size
                input_batch_size, input_num_frames = self.input_lap.shape[:2]
                if self.noise.shape[0] != input_batch_size or self.noise.shape[1] != input_num_frames:
                    self.noise = torch.randn(input_batch_size, input_num_frames, 3, self.opt.isize, self.opt.isize).to(self.device)
                
                self.fake_lap, self.fake_res = self.netg((self.input_lap.view(-1, *self.input_lap.shape[2:]) + 
                                                        self.noise.view(-1, *self.noise.shape[2:]),
                                                        self.input_res.view(-1, *self.input_res.shape[2:]) + 
                                                        self.noise.view(-1, *self.noise.shape[2:])))
                
                # Reshape back to video format
                batch_size, num_frames = self.input_lap.shape[:2]
                self.fake_lap = self.fake_lap.view(batch_size, num_frames, *self.fake_lap.shape[1:])
                self.fake_res = self.fake_res.view(batch_size, num_frames, *self.fake_res.shape[1:])
                self.fake = self.fake_lap + self.fake_res

                # Get features for anomaly scoring
                real_flat = (self.input_lap + self.input_res).view(-1, *self.input_lap.shape[2:])
                fake_flat = self.fake.view(-1, *self.fake.shape[2:])
                
                _, self.feat_real = self.netd(real_flat)
                _, self.feat_fake = self.netd(fake_flat)

                # Calculate the anomaly score for the entire video snippet
                si = self.input_lap.size()
                sz = self.feat_real.size()
                
                # Reconstruction error over all frames
                rec = (self.input_lap + self.input_res - self.fake).view(si[0], si[1] * si[2] * si[3] * si[4])
                
                # Feature matching error over all frames  
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                lat = lat.view(si[0], -1)  # Group by video snippet
                
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9 * rec + 0.1 * lat

                time_o = time.time()

                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))
                

                self.times.append(time_o - time_i)
                  # Update progress bar - clean and minimal
                test_bar.set_postfix({
                    'Batch': f"{i+1}/{len(self.data.valid)}"
                })

                # Save test images (save middle frame of each snippet)
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst): 
                        os.makedirs(dst)
                    
                    # Save middle frame (frame 8 out of 16)
                    middle_frame_idx = self.num_frames // 2
                    real_vis = self.input_lap[:, middle_frame_idx] + self.input_res[:, middle_frame_idx]
                    fake_vis = self.fake[:, middle_frame_idx]
                    fake_lap_vis = self.fake_lap[:, middle_frame_idx]
                    fake_res_vis = self.fake_res[:, middle_frame_idx]
                    
                    vutils.save_image(real_vis, '%s/real_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake_vis, '%s/fake_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake_lap_vis, '%s/fake_lap_%03d.png' % (dst, i+1), normalize=True)
                    vutils.save_image(fake_res_vis, '%s/fake_res_%03d.png' % (dst, i+1), normalize=True)

            # Close progress bar
            test_bar.close()

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / \
                             (torch.max(self.an_scores) - torch.min(self.an_scores))
            
            # Fix label convention: sklearn expects anomaly=1, normal=0
            # But our dataset has bad=0, good=1 due to alphabetical ordering
            # So we need to invert the labels for correct AUC calculation
            corrected_labels = 1 - self.gt_labels
            
            
            auc = roc(corrected_labels, self.an_scores)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            if self.opt.load_weights:
                self.visualizer.print_current_performance(performance, auc)

            return performance
    
    def train(self):
        """Train the video model"""
        print(f"\n>> Training {self.name} on {self.opt.dataset}")
        
        best_auc = 0
          # Add progress bar for epochs
        epoch_bar = tqdm(range(self.opt.niter), 
                        desc="Training Progress", 
                        unit="epoch")
        
        # Train for niter epochs
        for epoch in epoch_bar:
            self.epoch = epoch
            epoch_bar.set_description(f"Epoch {epoch+1}/{self.opt.niter}")
            
            # Set to training mode
            self.netg.train()
            self.netd.train()
            
            epoch_iter = 0
              # Track training losses for epoch summary
            epoch_losses = []
            recent_losses = []  # For running average in progress bar
            epoch_start_time = time.time()
            
            # Add progress bar for training batches
            train_bar = tqdm(enumerate(self.data.train, 0),
                           total=len(self.data.train),
                           desc=f"Training Epoch {epoch+1}/{self.opt.niter}",
                           unit="batch",
                           leave=False)
            
            for i, data in train_bar:
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                
                # Set input and optimize parameters
                self.set_input(data, noise=True)
                self.optimize_params()
                
                # Collect losses for epoch summary
                errors = self.get_errors()
                epoch_losses.append(errors)
                recent_losses.append(errors)
                  # Always log detailed losses to file for every batch (but not console)
                if self.total_steps % self.opt.print_freq == 0:
                    self.visualizer.print_current_errors(epoch, errors, print_to_console=False)
                
                # Update progress bar with running average every few batches
                if len(recent_losses) >= 5 or i == len(self.data.train) - 1:
                    # Calculate running average for progress bar
                    avg_g_loss = np.mean([loss['err_g'] for loss in recent_losses])
                    avg_d_loss = np.mean([loss['err_d'] for loss in recent_losses])
                    
                    train_bar.set_postfix({
                        'G_avg': f"{avg_g_loss:.3f}",
                        'D_avg': f"{avg_d_loss:.3f}",
                        'Batch': f"{i+1}/{len(self.data.train)}"
                    })
                    
                    # Reset recent losses for next interval
                    recent_losses = []
                
                # Save images
                if self.total_steps % self.opt.save_image_freq == 0:
                    reals, fakes, fake_lap, fake_res = self.get_current_images()
                    self.visualizer.save_current_images(epoch, reals, fakes, fake_lap, fake_res)
            
            # Close training progress bar
            train_bar.close()
            
            # Calculate epoch training summary
            epoch_training_time = time.time() - epoch_start_time
            
            # Calculate mean training losses for the epoch
            mean_training_losses = {}
            if epoch_losses:
                for key in epoch_losses[0].keys():
                    mean_training_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            # Log training summary
            self.visualizer.log_epoch_training_summary(
                epoch, self.opt.niter, mean_training_losses, epoch_training_time
            )
            
            # Test model at end of epoch
            performance = self.test()
            auc = performance['AUC']
            
            # Check if this is the best epoch
            is_best_epoch = auc > best_auc
            
            # Update learning rate
            if self.opt.lr_policy != 'constant':
                self.update_learning_rate()
            
            # Save weights if best
            if is_best_epoch:
                best_auc = auc
                self.save_weights(epoch, is_best=True)
            self.save_weights(epoch, is_best=False)
            
            # Log testing summary
            self.visualizer.log_epoch_testing_summary(
                epoch, self.opt.niter, performance, best_auc, is_best_epoch
            )
              # Update epoch progress bar - clean display
            epoch_bar.set_postfix({
                'AUC': f"{auc:.4f}",
                'Best': f"{best_auc:.4f}"
            })
        
        # Close epoch progress bar
        epoch_bar.close()
        
        print(f"\nTraining completed. Best AUC: {best_auc:.4f}")
        return best_auc
