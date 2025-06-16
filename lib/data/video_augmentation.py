#!/usr/bin/env python3
"""
Video data augmentation for surveillance anomaly detection.

Designed for surveillance video characteristics:
- Static surveillance camera (no camera motion)
- Consistent lighting conditions
- Clear anomaly detection requirements
- Various video quality scenarios
"""

import torch
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
import torch.nn.functional as F 


class VideoAugmentation:
    """
    General video augmentation pipeline for surveillance anomaly detection.
    
    Key Design Principles:
    1. Conservative approach - preserve anomaly characteristics
    2. Static camera simulation - minimal spatial transformations
    3. Quality-aware - adjustable noise/artifacts
    4. Motion-focused - maintain object motion patterns
    """
    
    def __init__(self, mode='conservative'):
        """
        Initialize video augmentation pipeline.        
        Args:
            mode (str): 'conservative' (default), 'moderate', or 'minimal'
        """
        self.mode = mode
        self.augmentations = self._setup_augmentations()
    
    def _setup_augmentations(self):
        """Setup augmentation pipeline based on mode"""
        if self.mode == 'minimal':
            return transforms.Compose([
                LightingAdjustment(brightness_range=0.05, probability=0.3),
                SensorNoise(noise_std=0.005, probability=0.2),
            ])
        elif self.mode == 'moderate':
            return transforms.Compose([
                LightingAdjustment(brightness_range=0.1, probability=0.4),
                MinimalJitter(max_translate=1, probability=0.3),
                SensorNoise(noise_std=0.01, probability=0.3),
                SubtleMotionBlur(kernel_size=3, probability=0.2),
            ])
        else:  # conservative (default)
            return transforms.Compose([
                LightingAdjustment(brightness_range=0.08, probability=0.35),
                MinimalJitter(max_translate=2, probability=0.25),
                SensorNoise(noise_std=0.008, probability=0.25),
                SubtleMotionBlur(kernel_size=3, probability=0.15),
                RareFrameDropout(drop_probability=0.02),
            ])
    
    def __call__(self, img):
        """Apply video augmentation"""
        return self.augmentations(img)


class LightingAdjustment:
    """
    Simulate natural lighting variations in surveillance video.
    Handles varying lighting conditions common in surveillance environments.
    """
    
    def __init__(self, brightness_range=0.08, contrast_range=0.05, probability=0.35):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        if isinstance(img, torch.Tensor):
            # Tensor-based adjustment
            # Brightness adjustment
            brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            img = torch.clamp(img * brightness_factor, 0, 1)
            
            # Contrast adjustment
            if random.random() < 0.5:  # 50% chance for contrast change
                contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
                mean = img.mean()
                img = torch.clamp((img - mean) * contrast_factor + mean, 0, 1)
        else:
            # PIL Image adjustment
            # Brightness
            brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
            
            # Contrast
            if random.random() < 0.5:
                contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_factor)
        
        return img


class MinimalJitter:
    """
    Simulate very minor camera vibrations and mounting instability.
    Common in fixed surveillance camera setups (1-2 pixels max movement).
    """
    
    def __init__(self, max_translate=2, rotate_range=0.5, probability=0.25):
        self.max_translate = max_translate
        self.rotate_range = rotate_range
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        if isinstance(img, torch.Tensor):
            # Convert to PIL for transformation
            img = transforms.ToPILImage()(img)
            tensor_input = True
        else:
            tensor_input = False
        
        # Very small random translation
        dx = random.randint(-self.max_translate, self.max_translate)
        dy = random.randint(-self.max_translate, self.max_translate)
        
        # Very small rotation (simulates minor camera tilt)
        angle = random.uniform(-self.rotate_range, self.rotate_range)
        
        # Apply transformations
        if abs(dx) > 0 or abs(dy) > 0 or abs(angle) > 0.1:
            # Use affine transformation for combined translation and rotation
            img = transforms.functional.affine(img, 
                                             angle=angle,
                                             translate=[dx, dy],
                                             scale=1.0,
                                             shear=0)
        
        if tensor_input:
            img = transforms.ToTensor()(img)
        
        return img


class SensorNoise:
    """
    Add minimal sensor noise typical of surveillance cameras.
    High quality surveillance systems generally have minimal noise.
    """
    
    def __init__(self, noise_std=0.008, probability=0.25):
        self.noise_std = noise_std
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        if isinstance(img, torch.Tensor):
            # Add Gaussian noise to tensor
            noise = torch.randn_like(img) * self.noise_std
            img = torch.clamp(img + noise, 0, 1)
        else:
            # Convert PIL to tensor, add noise, convert back
            img_tensor = transforms.ToTensor()(img)
            noise = torch.randn_like(img_tensor) * self.noise_std
            noisy_tensor = torch.clamp(img_tensor + noise, 0, 1)
            img = transforms.ToPILImage()(noisy_tensor)
        
        return img


class SubtleMotionBlur:
    """
    Add very subtle motion blur to simulate object movement.
    Common for objects moving at normal speeds in surveillance video.
    """
    
    def __init__(self, kernel_size=3, probability=0.15):
        self.kernel_size = kernel_size
        self.probability = probability
    
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        if isinstance(img, torch.Tensor):
            # Convert to PIL for OpenCV processing
            img = transforms.ToPILImage()(img)
            tensor_input = True
        else:
            tensor_input = False
        
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Create simple motion blur kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[self.kernel_size//2, :] = 1  # Horizontal motion blur
        kernel = kernel / self.kernel_size
        
        # Apply blur
        blurred = cv2.filter2D(img_cv, -1, kernel)
        
        # Convert back to PIL
        img = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        
        if tensor_input:
            img = transforms.ToTensor()(img)
        
        return img


class RareFrameDropout:
    """
    Simulate very rare frame corruption or transmission issues.
    High quality surveillance systems have rare dropouts.
    """
    
    def __init__(self, drop_probability=0.02, mask_value=0.0):
        self.drop_probability = drop_probability
        self.mask_value = mask_value
    
    def __call__(self, img):
        if random.random() > self.drop_probability:
            return img
        
        if isinstance(img, torch.Tensor):
            # Mask entire frame
            return torch.full_like(img, self.mask_value)
        else:
            # Create black PIL image
            return Image.new(img.mode, img.size, color=int(self.mask_value * 255))


class TemporalAugmentation:
    """
    Simple temporal augmentations for surveillance video patterns.
    """
    
    @staticmethod
    def apply_minimal_temporal_jitter(frames, probability=0.2, max_jitter=1):
        """
        Very minimal frame reordering for surveillance video.
        Args:
            frames: List of frame filenames
            probability: Chance to apply jitter
            max_jitter: Maximum frames to jitter (1 for minimal changes)
        """
        if random.random() > probability or len(frames) <= max_jitter:
            return frames
        
        jittered_frames = frames.copy()
        
        # Only swap adjacent frames occasionally
        for i in range(0, len(frames) - 1, 2):
            if random.random() < 0.3:  # 30% chance to swap adjacent frames
                if i + 1 < len(frames):
                    jittered_frames[i], jittered_frames[i + 1] = jittered_frames[i + 1], jittered_frames[i]
        
        return jittered_frames
    
    @staticmethod
    def apply_object_speed_variation(frames, probability=0.15):
        """
        Simulate different object movement speeds by frame sampling.
        Very conservative approach for surveillance video.
        """
        if random.random() > probability:
            return frames
        
        # Very minimal speed variation - just skip one frame occasionally
        if len(frames) > 8:  # Only if we have enough frames
            # Remove one random frame from middle section
            middle_start = len(frames) // 4
            middle_end = 3 * len(frames) // 4
            remove_idx = random.randint(middle_start, middle_end)
            
            modified_frames = frames[:remove_idx] + frames[remove_idx+1:]
            
            # Duplicate a nearby frame to maintain length
            duplicate_idx = min(remove_idx, len(modified_frames) - 1)
            modified_frames.insert(remove_idx, modified_frames[duplicate_idx])
            
            return modified_frames
        
        return frames


def create_video_transforms(opt, mode='conservative'):
    """
    Create general video transform pipeline for surveillance anomaly detection.
    
    Args:
        opt: Options object with isize, aspect_method, etc.
        mode: 'minimal', 'conservative', or 'moderate'
    
    Returns:
        tuple: (basic_transforms, augmented_transforms)    """
    # Import required components (avoid circular import)
    from lib.data.data_loading import VideoAspectRatioResize, calculate_optimal_dimensions
    
    # Calculate optimal dimensions for aspect ratio preservation
    target_size = calculate_optimal_dimensions(opt)
    aspect_method = getattr(opt, 'aspect_method', 'maintain_3_2')
    
    # Standard video normalization
    mean_video = (0.5, 0.5, 0.5)
    std_video = (0.5, 0.5, 0.5)
    
    # Basic transform pipeline (no augmentation)
    basic_transforms = transforms.Compose([
        VideoAspectRatioResize(opt.isize, target_size, method=aspect_method),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_video, std=std_video)
    ])
      # General video augmented pipeline
    video_augmentation = VideoAugmentation(mode=mode)
    
    augmented_transforms = transforms.Compose([
        VideoAspectRatioResize(opt.isize, target_size, method=aspect_method),
        video_augmentation,  # Apply general video augmentations
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_video, std=std_video)
    ])
    
    print(f"✅ Video Augmentation Pipeline ({mode} mode)")
    print(f"   Aspect Ratio: {aspect_method} | Target: {target_size[0]}x{target_size[1]} → {opt.isize}x{opt.isize}")
    print(f"   Augmentations: Tailored for static camera surveillance")
    
    return basic_transforms, augmented_transforms


# Example usage and testing
if __name__ == "__main__":
    print("🎯 Video Augmentation Module")
    print("=" * 40)
    
    # Test with dummy image
    test_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    
    # Test different modes
    for mode in ['minimal', 'conservative', 'moderate']:
        print(f"\n🔧 Testing {mode} mode:")
        augmentation = VideoAugmentation(mode=mode)
        
        # Apply augmentation multiple times to see variation
        for i in range(3):
            aug_img = augmentation(test_img)
            print(f"   Sample {i+1}: Applied successfully")
    
    print(f"\n✅ All video augmentation modes working!")
