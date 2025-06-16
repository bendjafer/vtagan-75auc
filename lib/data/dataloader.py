"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import numpy as np
import torch
import random
import math
from torchvision.transforms import *
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from lib.data.datasets import ImageFolder_FD, ImageFolder, ImageFolder_FD_Aug
from lib.data.video_datasets import VideoSnippetDataset, VideoSnippetDatasetAug
class VideoFrameDropout(object):
    """
    Video-specific augmentation: Randomly drop/mask frames to simulate missing data
    Useful for video anomaly detection as it forces temporal robustness
    """
    def __init__(self, drop_prob=0.1, mask_value=0.0):
        self.drop_prob = drop_prob
        self.mask_value = mask_value
    
    def __call__(self, img):
        """Apply frame dropout with given probability"""
        if random.random() < self.drop_prob:
            # Create a masked version of the frame
            if isinstance(img, torch.Tensor):
                return torch.full_like(img, self.mask_value)
            else:  # PIL Image
                return Image.new(img.mode, img.size, color=int(self.mask_value * 255))
        return img


class MotionBlur(object):
    """
    Simulate motion blur to augment video frames
    Common in surveillance videos due to camera movement or object motion
    """
    def __init__(self, kernel_size=5, blur_prob=0.3):
        self.kernel_size = kernel_size
        self.blur_prob = blur_prob
    
    def __call__(self, img):
        """Apply motion blur with given probability"""
        if random.random() < self.blur_prob:
            # Use GaussianBlur instead of MotionBlur (which doesn't exist in PIL)
            blur_radius = random.uniform(0.5, self.kernel_size)
            if isinstance(img, torch.Tensor):
                # Convert to PIL for processing
                img_pil = transforms.ToPILImage()(img)
                blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                return transforms.ToTensor()(blurred)
            else:  # PIL Image
                return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return img


class FrameNoise(object):
    """
    Add realistic noise to video frames (Gaussian noise)
    Simulates sensor noise and poor lighting conditions in surveillance
    """
    def __init__(self, noise_std=0.02, noise_prob=0.4):
        self.noise_std = noise_std
        self.noise_prob = noise_prob
    
    def __call__(self, img):
        """Add Gaussian noise to frame"""
        if random.random() < self.noise_prob:
            if isinstance(img, torch.Tensor):
                noise = torch.randn_like(img) * self.noise_std
                return torch.clamp(img + noise, 0, 1)
            else:  # PIL Image
                img_tensor = transforms.ToTensor()(img)
                noise = torch.randn_like(img_tensor) * self.noise_std
                noisy_tensor = torch.clamp(img_tensor + noise, 0, 1)
                return transforms.ToPILImage()(noisy_tensor)
        return img


class LightingVariation(object):
    """
    Simulate lighting changes common in surveillance videos
    Includes brightness, contrast, and gamma adjustments
    """
    def __init__(self, brightness_range=0.15, contrast_range=0.15, gamma_range=0.2):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
    
    def __call__(self, img):
        """Apply lighting variations"""
        # Random brightness adjustment
        brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
        
        # Random contrast adjustment  
        contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
        
        # Random gamma correction
        gamma = 1.0 + random.uniform(-self.gamma_range, self.gamma_range)
        
        if isinstance(img, torch.Tensor):
            # Apply gamma correction
            img_gamma = torch.pow(img, gamma)
            # Apply brightness and contrast
            img_enhanced = torch.clamp(img_gamma * contrast_factor + (brightness_factor - 1), 0, 1)
            return img_enhanced
        else:  # PIL Image
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
            # Gamma correction for PIL
            img_array = np.array(img).astype(np.float32) / 255.0
            img_gamma = np.power(img_array, gamma)
            img_gamma = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(img_gamma)


class SpatialJitter(object):
    """
    Small spatial perturbations to simulate camera vibrations
    Common in surveillance systems
    """
    def __init__(self, max_translate=4, rotate_range=2.0, scale_range=0.05):
        self.max_translate = max_translate
        self.rotate_range = rotate_range
        self.scale_range = scale_range
    
    def __call__(self, img):
        """Apply spatial jittering"""
        # Random translation
        translate_x = random.randint(-self.max_translate, self.max_translate)
        translate_y = random.randint(-self.max_translate, self.max_translate)
        
        # Random scaling
        scale = 1.0 + random.uniform(-self.scale_range, self.scale_range)
        
        if isinstance(img, torch.Tensor):
            img_pil = transforms.ToPILImage()(img)
        else:
            img_pil = img
            
        # Apply transformations
        # Normalize translation to [0, 1] range for RandomAffine
        translate_x_norm = abs(translate_x) / max(img_pil.size[0], 1)
        translate_y_norm = abs(translate_y) / max(img_pil.size[1], 1)
        translate_x_norm = min(translate_x_norm, 0.1)  # Cap at 10% of image size
        translate_y_norm = min(translate_y_norm, 0.1)
        
        transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=(-self.rotate_range, self.rotate_range),
                translate=(translate_x_norm, translate_y_norm),
                scale=(scale, scale),
                fillcolor=0
            )
        ])
        
        transformed = transform(img_pil)
        
        if isinstance(img, torch.Tensor):
            return transforms.ToTensor()(transformed)
        else:
            return transformed


class PixelShuffle(object):
    """
    Randomly shuffle small patches to create subtle anomalies
    Helps model learn to detect spatial inconsistencies
    """
    def __init__(self, patch_size=8, shuffle_prob=0.2, num_patches=3):
        self.patch_size = patch_size
        self.shuffle_prob = shuffle_prob
        self.num_patches = num_patches
    
    def __call__(self, img):
        """Apply pixel shuffling in small patches"""
        if random.random() < self.shuffle_prob:
            if isinstance(img, torch.Tensor):
                img_copy = img.clone()
                _, h, w = img.shape
            else:  # PIL Image
                img_array = np.array(img)
                h, w = img_array.shape[:2]
                img_copy = img_array.copy()
            
            for _ in range(self.num_patches):
                # Random patch location
                y = random.randint(0, h - self.patch_size)
                x = random.randint(0, w - self.patch_size)
                
                if isinstance(img, torch.Tensor):
                    # Extract and shuffle patch
                    patch = img_copy[:, y:y+self.patch_size, x:x+self.patch_size]
                    patch_flat = patch.reshape(patch.shape[0], -1)
                    shuffled_indices = torch.randperm(patch_flat.shape[1])
                    patch_shuffled = patch_flat[:, shuffled_indices].reshape(patch.shape)
                    img_copy[:, y:y+self.patch_size, x:x+self.patch_size] = patch_shuffled
                else:
                    # For PIL/numpy arrays
                    patch = img_copy[y:y+self.patch_size, x:x+self.patch_size]
                    patch_flat = patch.reshape(-1, patch.shape[-1] if len(patch.shape) == 3 else 1)
                    np.random.shuffle(patch_flat)
                    img_copy[y:y+self.patch_size, x:x+self.patch_size] = patch_flat.reshape(patch.shape)
            
            if isinstance(img, torch.Tensor):
                return img_copy
            else:
                return Image.fromarray(img_copy.astype(np.uint8))
        
        return img

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

##
def load_data(opt, classes):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        if opt.dataset == 'all':
            opt.dataroot = './data/{}'.format(classes)
        else:
            opt.dataroot = './data/{}'.format(opt.dataset)

    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    train_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
    valid_ds = ImageFolder(os.path.join(opt.dataroot, 'test'), transform)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)

def load_data_FD(opt, classes):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        if opt.dataset == 'all':
            opt.dataroot = './data/{}'.format(classes)
        else:
            opt.dataroot = './data/{}'.format(opt.dataset)

    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    train_ds = ImageFolder_FD(os.path.join(opt.dataroot, 'train'), transform)
    valid_ds = ImageFolder_FD(os.path.join(opt.dataroot, 'test'), transform)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)

def load_data_FD_aug(opt, classes):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        if opt.dataset == 'all':
            opt.dataroot = './data/{}'.format(classes)
        else:
            opt.dataroot = './data/{}'.format(opt.dataset)

    # Use aspect ratio preserving transforms to avoid distortion
    # For UCSD videos (360×240 → optimal resolution maintaining 3:2 ratio)
    aspect_method = getattr(opt, 'aspect_method', 'maintain_3_2')  # Default to maintain aspect ratio
    
    if aspect_method == 'maintain_3_2':
        # Maintain 3:2 aspect ratio - no distortion
        video_resize = VideoAspectRatioResize(method='maintain_3_2', base_size=opt.isize)
        print(f"✅ Using aspect ratio preserving transforms: {video_resize.target_size}")
    elif aspect_method == 'center_crop':
        # Center crop to square (may lose some content)
        video_resize = VideoAspectRatioResize(method='center_crop', base_size=opt.isize)
        print(f"⚠️ Using center crop to square: {video_resize.target_size}")
    else:
        # Fallback to original behavior (stretch - causes distortion)
        video_resize = VideoAspectRatioResize(method='stretch', base_size=opt.isize)
        print(f"⚠️ Using stretch method (may cause distortion): {video_resize.target_size}")
    
    transform = transforms.Compose([video_resize,
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    transform_aug = transforms.Compose([video_resize,
                                        # Video-specific augmentations optimized for surveillance/anomaly detection
                                        LightingVariation(brightness_range=0.15, contrast_range=0.15),  # Subtle lighting changes
                                        SpatialJitter(max_translate=2, rotate_range=1.0),              # Minimal camera movement
                                        MotionBlur(kernel_size=3, blur_prob=0.2),                      # Occasional motion blur
                                        transforms.ToTensor(),
                                        FrameNoise(noise_std=0.01, noise_prob=0.3),                    # Light sensor noise
                                        VideoFrameDropout(drop_prob=0.03),                             # Rare frame dropout
                                        PixelShuffle(patch_size=4, shuffle_prob=0.1),                  # Minimal patch shuffling
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    train_ds = ImageFolder_FD_Aug(os.path.join(opt.dataroot, 'train'), transform, transform_aug)
    valid_ds = ImageFolder_FD_Aug(os.path.join(opt.dataroot, 'test'), transform, transform_aug)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)

def load_video_data_FD(opt, classes):
    """ Load Video Snippet Data for OCR-GAN Video

    Args:
        opt ([type]): Argument Parser
        classes: List of class names

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        if opt.dataset == 'all':
            opt.dataroot = './data/{}'.format(classes)
        else:
            opt.dataroot = './data/{}'.format(opt.dataset)

    # Calculate optimal dimensions for aspect ratio preservation
    target_size = calculate_optimal_dimensions(opt)
    
    # Define normalization parameters for video data
    mean_video = (0.5, 0.5, 0.5)
    std_video = (0.5, 0.5, 0.5)

    # Basic transform pipeline: Resize → Center Crop → Normalize
    data_transforms = transforms.Compose([
        transforms.Resize((target_size[1], target_size[0])),  # Resize to target size (height, width)
        transforms.CenterCrop(opt.isize),                     # Center crop to square
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_video, std=std_video)
    ])

    # Get number of frames from options
    num_frames = opt.num_frames if hasattr(opt, 'num_frames') else 16
    
    # Print aspect ratio information
    aspect_method = getattr(opt, 'aspect_method', 'maintain_3_2')
    if aspect_method == 'maintain_3_2':
        print(f"✅ Using aspect ratio preserving transforms: {target_size[0]}x{target_size[1]}")
    else:
        print(f"ℹ️ Using {aspect_method} method: {target_size[0]}x{target_size[1]}")

    train_ds = VideoSnippetDataset(os.path.join(opt.dataroot, 'train'), data_transforms, num_frames)
    valid_ds = VideoSnippetDataset(os.path.join(opt.dataroot, 'test'), data_transforms, num_frames)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)


def load_video_data_FD_aug(opt, classes):
    """ Load Video Snippet Data with Augmentation for OCR-GAN Video

    Args:
        opt ([type]): Argument Parser
        classes: List of class names

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        if opt.dataset == 'all':
            opt.dataroot = './data/{}'.format(classes)
        else:
            opt.dataroot = './data/{}'.format(opt.dataset)

    # Calculate optimal dimensions for aspect ratio preservation
    target_size = calculate_optimal_dimensions(opt)
    
    # Define normalization parameters for video data
    mean_video = (0.5, 0.5, 0.5)
    std_video = (0.5, 0.5, 0.5)
    
    # Standard computer vision pipeline: Resize (shortest side) → Center Crop → ToTensor → Normalize
    data_transforms = transforms.Compose([
        transforms.Resize(opt.isize),         # Resize shortest side to isize (maintains aspect ratio)
        transforms.CenterCrop(opt.isize),     # Center crop to square (isize x isize)
        transforms.ToTensor(),                # Convert PIL Image to tensor [0,1]
        transforms.Normalize(mean=mean_video, std=std_video)  # Normalize to [-1,1]
    ])
    
    # Augmented transform pipeline: Resize → Center Crop → Augment → Normalize
    data_transforms_aug = transforms.Compose([
        VideoAspectRatioResize(opt.isize, target_size, method=opt.aspect_method),  # Aspect ratio aware resize + crop
        # Video-specific augmentations (applied before ToTensor)
        LightingVariation(brightness_range=0.1, contrast_range=0.1),   # Minimal lighting changes
        SpatialJitter(max_translate=1, rotate_range=0.5),              # Tiny camera jitter  
        MotionBlur(kernel_size=3, blur_prob=0.15),                     # Rare motion blur
        transforms.ToTensor(),
        # Tensor-based augmentations (applied after ToTensor)
        FrameNoise(noise_std=0.008, noise_prob=0.25),                  # Light sensor noise
        VideoFrameDropout(drop_prob=0.02),                             # Very rare dropout
        PixelShuffle(patch_size=4, shuffle_prob=0.08),                 # Minimal artifacts
        transforms.Normalize(mean=mean_video, std=std_video)
    ])

    # Get number of frames from options
    num_frames = opt.num_frames if hasattr(opt, 'num_frames') else 16
    
    # Print transform pipeline information
    aspect_method = getattr(opt, 'aspect_method', 'maintain_3_2')
    print(f"✅ Transform pipeline: Resize({target_size[0]}x{target_size[1]}) → CenterCrop({opt.isize}x{opt.isize}) → Normalize")
    print(f"   Original: 360x240 → Resized: {target_size[0]}x{target_size[1]} → Final: {opt.isize}x{opt.isize}")
    
    if aspect_method == 'maintain_3_2':
        print(f"   Aspect ratio preservation: {aspect_method} method selected")

    train_ds = VideoSnippetDatasetAug(os.path.join(opt.dataroot, 'train'), data_transforms, data_transforms_aug, num_frames)
    valid_ds = VideoSnippetDatasetAug(os.path.join(opt.dataroot, 'test'), data_transforms, data_transforms_aug, num_frames)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)

def load_video_data_FD_ucsd_ped2(opt, classes, augmentation_mode='conservative'):
    """ Load Video Snippet Data with UCSD Ped2 Simplified Augmentation

    Args:
        opt ([type]): Argument Parser
        classes: List of class names
        augmentation_mode (str): 'minimal', 'conservative', or 'moderate'

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        if opt.dataset == 'all':
            opt.dataroot = './data/{}'.format(classes)
        else:
            opt.dataroot = './data/{}'.format(opt.dataset)

    # Import UCSD Ped2 specific augmentation
    from lib.data.ucsd_ped2_augmentation import create_ucsd_ped2_transforms
    
    # Create UCSD Ped2 specific transforms
    data_transforms, data_transforms_aug = create_ucsd_ped2_transforms(opt, mode=augmentation_mode)
    
    # Get number of frames from options
    num_frames = 16
    
    # Print information
    print(f"🎯 UCSD Ped2 Simplified Augmentation Loader")
    print(f"   Mode: {augmentation_mode}")
    print(f"   Frames per video: {num_frames}")
    print(f"   Dataset: {opt.dataroot}")

    # Import UCSD Ped2 specific dataset class
    from lib.data.video_datasets import VideoSnippetDatasetUCSD_Ped2
    
    train_ds = VideoSnippetDatasetUCSD_Ped2(os.path.join(opt.dataroot, 'train'), data_transforms, data_transforms_aug, num_frames, temporal_mode=augmentation_mode)
    valid_ds = VideoSnippetDatasetUCSD_Ped2(os.path.join(opt.dataroot, 'test'), data_transforms, data_transforms_aug, num_frames, temporal_mode=augmentation_mode)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)

class AspectRatioResize:
    """
    Resize images while maintaining aspect ratio, then center crop or pad to target size
    """
    def __init__(self, target_size, method='crop'):
        """
        Args:
            target_size (int or tuple): Target size for the image
            method (str): 'crop' to center crop, 'pad' to pad with zeros, 'stretch' to ignore aspect ratio
        """
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        self.method = method
    
    def __call__(self, img):
        """
        Apply aspect ratio preserving resize
        """
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL for processing
            img = transforms.ToPILImage()(img)
            tensor_input = True
        else:
            tensor_input = False
        
        original_width, original_height = img.size
        target_width, target_height = self.target_size
        
        if self.method == 'stretch':
            # Just resize to target - causes distortion but keeps all content
            resized = img.resize((target_width, target_height), Image.LANCZOS)
        
        elif self.method == 'crop':
            # Maintain aspect ratio, resize and center crop
            # Calculate scale to fit the larger dimension
            scale = max(target_width / original_width, target_height / original_height)
            
            # Resize maintaining aspect ratio
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Center crop to target size
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            resized = resized.crop((left, top, right, bottom))
        
        elif self.method == 'pad':
            # Maintain aspect ratio, resize and pad
            # Calculate scale to fit the smaller dimension
            scale = min(target_width / original_width, target_height / original_height)
            
            # Resize maintaining aspect ratio
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create target sized image with padding
            padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            
            # Paste resized image in center
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            padded.paste(resized, (paste_x, paste_y))
            resized = padded
        
        elif self.method == 'maintain_ratio':
            # Resize to maintain original aspect ratio - different output size
            # For 360x240 → maintain 3:2 ratio
            aspect_ratio = original_width / original_height  # 360/240 = 1.5
            
            if target_width / target_height > aspect_ratio:
                # Target is wider than source, fit to height
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            else:
                # Target is taller than source, fit to width
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            
            resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        if tensor_input:
            # Convert back to tensor
            resized = transforms.ToTensor()(resized)
        
        return resized


class VideoAspectRatioResize:
    """
    Video-specific aspect ratio handling implementing the standard pipeline:
    Resize → Center Crop → Result in square images
    """
    def __init__(self, final_size, target_size, method='maintain_3_2'):
        """
        Args:
            final_size (int): Final square output size (e.g., 64 for 64x64)
            target_size (tuple): Intermediate resize target (width, height) before center crop
            method (str): 
                - 'maintain_3_2': Resize to target_size maintaining aspect, then center crop to square
                - 'center_crop': Crop to square, then resize
                - 'pad_square': Pad to square, then resize
                - 'stretch': Stretch to square (original behavior)
        """
        self.final_size = final_size
        self.target_size = target_size
        self.method = method
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL for processing
            img = transforms.ToPILImage()(img)
            tensor_input = True
        else:
            tensor_input = False
            
        if self.method == 'maintain_3_2':
            # Standard pipeline: Resize to target_size → Center crop to square
            # Step 1: Resize to target dimensions (e.g., 360×240 → 96×64)
            resized = img.resize(self.target_size, Image.LANCZOS)
            
            # Step 2: Center crop to final square size (e.g., 96×64 → 64×64)
            width, height = resized.size
            final_size = self.final_size
            
            # Calculate crop coordinates for center crop
            left = (width - final_size) // 2
            top = (height - final_size) // 2
            right = left + final_size
            bottom = top + final_size
            
            # Ensure crop is within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            
            result = resized.crop((left, top, right, bottom))
            
            # If the crop result is not exactly final_size, resize to ensure exact dimensions
            if result.size != (final_size, final_size):
                result = result.resize((final_size, final_size), Image.LANCZOS)
        
        elif self.method == 'center_crop':
            # Crop to square first, then resize
            aspect_resize = AspectRatioResize((self.final_size, self.final_size), method='crop')
            result = aspect_resize(img)
        
        elif self.method == 'pad_square':
            # Pad to square, then resize
            aspect_resize = AspectRatioResize((self.final_size, self.final_size), method='pad')
            result = aspect_resize(img)
        
        elif self.method == 'stretch':
            # Original behavior - stretch to square
            result = img.resize((self.final_size, self.final_size), Image.LANCZOS)
        
        else:
            result = img
        
        if tensor_input:
            # Convert back to tensor
            result = transforms.ToTensor()(result)
        
        return result

def calculate_optimal_dimensions(opt):
    """
    Calculate optimal input dimensions for aspect ratio preservation
    Updates opt parameters if needed for network compatibility
    """
    aspect_method = getattr(opt, 'aspect_method', 'maintain_3_2')
    base_size = opt.isize
    
    if aspect_method == 'maintain_3_2':
        # Calculate dimensions that maintain 3:2 aspect ratio and are powers of 2
        if base_size == 32:
            target_size = (48, 32)  # 48x32 maintains 3:2 ratio
        elif base_size == 64:
            target_size = (96, 64)  # 96x64 maintains 3:2 ratio
        elif base_size == 128:
            target_size = (192, 128)  # 192x128 maintains 3:2 ratio
        else:
            # For other sizes, calculate maintaining ratio and ensure divisible by 16
            width = int((base_size * 1.5) // 16) * 16
            target_size = (width, base_size)
        
        # Update opt.isize to use the height for network calculations
        # This ensures the U-Net depth calculation works correctly
        opt.effective_isize = base_size  # Use height for network depth
        
        print(f"✅ Aspect ratio preservation: {360}×{240} → {target_size[0]}×{target_size[1]}")
        print(f"   Network depth based on height: {base_size}")
        
    else:
        # For square methods, keep original size
        target_size = (base_size, base_size)
        opt.effective_isize = base_size
        print(f"ℹ️ Using square input: {target_size[0]}×{target_size[1]}")
    
    return target_size

def update_model_for_aspect_ratio(opt):
    """
    Update model parameters to handle non-square inputs properly
    """
    # Ensure isize calculations use the height (smaller dimension) for network depth
    if hasattr(opt, 'effective_isize'):
        original_isize = opt.isize
        opt.isize = opt.effective_isize
        print(f"📐 Model configuration: Using {opt.isize} for network depth calculations")
        return original_isize
    return opt.isize