"""
Optimized Video Data Loading for Anomaly Detection
Consolidated from multiple files to reduce complexity
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from lib.data.video_datasets import VideoSnippetDatasetAug, VideoSnippetDataset, VideoSnippetDatasetEnhanced
from lib.data.video_augmentation import create_video_transforms
from lib.data.utils import AspectRatioResize


class Data:
    """Simple data container class"""
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


class VideoAspectRatioResize:
    """
    Video-specific aspect ratio handling implementing the standard pipeline:
    Resize → Center Crop → Result in square images
    """
    def __init__(self, final_size, target_size, method='maintain_3_2'):
        self.final_size = final_size
        self.target_size = target_size
        self.method = method
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
            tensor_input = True
        else:
            tensor_input = False
            
        if self.method == 'maintain_3_2':
            # Standard pipeline: Resize to target_size → Center crop to square
            resized = img.resize(self.target_size, Image.LANCZOS)
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
        
        elif self.method in ['center_crop', 'pad_square']:
            aspect_resize = AspectRatioResize((self.final_size, self.final_size), 
                                            method='crop' if self.method == 'center_crop' else 'pad')
            result = aspect_resize(img)
        
        elif self.method == 'stretch':
            result = img.resize((self.final_size, self.final_size), Image.LANCZOS)
        
        else:
            result = img
        
        if tensor_input:
            result = transforms.ToTensor()(result)
        
        return result


def calculate_optimal_dimensions(opt):
    """Calculate optimal input dimensions for aspect ratio preservation"""
    aspect_method = getattr(opt, 'aspect_method', 'maintain_3_2')
    base_size = opt.isize
    
    if aspect_method == 'maintain_3_2':
        # Calculate dimensions that maintain 3:2 aspect ratio and are powers of 2
        if base_size == 32:
            target_size = (48, 32)
        elif base_size == 64:
            target_size = (96, 64)
        elif base_size == 128:
            target_size = (192, 128)
        else:
            width = int((base_size * 1.5) // 16) * 16
            target_size = (width, base_size)
        
        opt.effective_isize = base_size
        print(f"✅ Aspect ratio preservation: {360}×{240} → {target_size[0]}×{target_size[1]}")
        print(f"   Network depth based on height: {base_size}")
    else:
        target_size = (base_size, base_size)
        opt.effective_isize = base_size
        print(f"ℹ️ Using square input: {target_size[0]}×{target_size[1]}")
    
    return target_size


# Main data loading functions
def load_video_data_FD_enhanced(opt, classes, augmentation_mode='conservative'):
    """Load Video Snippet Data with Enhanced Video Augmentation"""
    if opt.dataroot == '':
        opt.dataroot = f'./data/{opt.dataset if opt.dataset != "all" else classes}'

    data_transforms, data_transforms_aug = create_video_transforms(opt, mode=augmentation_mode)
    num_frames = getattr(opt, 'num_frames', 16)
    
    print(f"🎯 Enhanced Video Data Loader")
    print(f"   Mode: {augmentation_mode}")
    print(f"   Frames per video: {num_frames}")
    print(f"   Dataset: {opt.dataroot}")

    train_ds = VideoSnippetDatasetEnhanced(
        os.path.join(opt.dataroot, 'train'), 
        data_transforms, 
        data_transforms_aug, 
        num_frames, 
        temporal_mode=augmentation_mode,
        image_size=opt.isize
    )
    valid_ds = VideoSnippetDatasetEnhanced(
        os.path.join(opt.dataroot, 'test'), 
        data_transforms, 
        data_transforms_aug, 
        num_frames, 
        temporal_mode=augmentation_mode,
        image_size=opt.isize
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)


def load_video_data_FD_aug(opt, classes):
    """Load Video Snippet Data with Augmentation"""
    if opt.dataroot == '':
        opt.dataroot = f'./data/{opt.dataset if opt.dataset != "all" else classes}'

    num_frames = getattr(opt, 'num_frames', 16)
    
    print(f"🎯 Video Data Loader with Augmentation")
    print(f"   Frames per video: {num_frames}")
    print(f"   Dataset: {opt.dataroot}")

    train_ds = VideoSnippetDatasetAug(os.path.join(opt.dataroot, 'train'), num_frames=num_frames, image_size=opt.isize)
    valid_ds = VideoSnippetDataset(os.path.join(opt.dataroot, 'test'), num_frames=num_frames, image_size=opt.isize)

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)


def load_video_data_FD(opt, classes):
    """Load Video Snippet Data without Augmentation"""
    if opt.dataroot == '':
        opt.dataroot = f'./data/{opt.dataset if opt.dataset != "all" else classes}'

    num_frames = getattr(opt, 'num_frames', 16)
    
    print(f"🎯 Video Data Loader (No Augmentation)")
    print(f"   Frames per video: {num_frames}")
    print(f"   Dataset: {opt.dataroot}")

    train_ds = VideoSnippetDataset(os.path.join(opt.dataroot, 'train'), num_frames=num_frames, image_size=opt.isize)
    valid_ds = VideoSnippetDataset(os.path.join(opt.dataroot, 'test'), num_frames=num_frames, image_size=opt.isize)

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)
