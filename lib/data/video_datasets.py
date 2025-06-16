import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from lib.data.utils import FD, find_classes, IMG_EXTENSIONS
import random


def ensure_tensor(x):
    """Convert PIL Image to tensor if needed"""
    if isinstance(x, Image.Image):
        return transforms.ToTensor()(x)
    return x


class TemporalAugmentation:
    """
    Video-specific temporal augmentation techniques for anomaly detection
    """
    
    @staticmethod
    def apply_temporal_jitter(frames, probability=0.3, max_jitter=2):
        """Randomly reorder frames within small windows"""
        if random.random() > probability or len(frames) <= max_jitter:
            return frames
            
        jittered_frames = frames.copy()
        num_frames = len(frames)
        
        # Shuffle frames in small windows
        for i in range(0, num_frames, max_jitter):
            end_idx = min(i + max_jitter, num_frames)
            window = jittered_frames[i:end_idx]
            random.shuffle(window)
            jittered_frames[i:end_idx] = window
            
        return jittered_frames
    
    @staticmethod
    def apply_frame_skip(frames, probability=0.2, skip_patterns=[2, 3]):
        """Skip frames to simulate different temporal sampling"""
        if random.random() > probability:
            return frames
            
        skip_factor = random.choice(skip_patterns)
        if len(frames) <= skip_factor:
            return frames
            
        # Take every skip_factor-th frame, then repeat to maintain length
        skipped = frames[::skip_factor]
        
        # Pad by repeating frames to maintain original length
        while len(skipped) < len(frames):
            skipped.extend(skipped[:len(frames) - len(skipped)])
            
        return skipped[:len(frames)]
    
    @staticmethod
    def apply_temporal_reverse(frames, probability=0.15):
        """Reverse temporal order"""
        if random.random() < probability:
            return frames[::-1]
        return frames


def make_snippet_dataset(dir, class_to_idx):
    """
    Create dataset of video snippets instead of individual images
    Each snippet folder contains 16 frames
    """
    snippets = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            # Check if this directory contains .tif files (is a snippet folder)
            tif_files = [f for f in fnames if f.lower().endswith(('.tif', '.tiff'))]
            if len(tif_files) > 0:
                # This is a snippet folder
                snippet_path = root
                snippets.append((snippet_path, class_to_idx[target]))

    return snippets


class VideoSnippetDataset(data.Dataset):
    """
    Dataset for loading video snippets (folders containing 16 frames each)    Each sample is a video snippet with 16 frames
    """
    
    def __init__(self, root, transform=None, num_frames=16, image_size=256):
        classes, class_to_idx = find_classes(root)
        snippets = make_snippet_dataset(root, class_to_idx)
        
        if len(snippets) == 0:
            raise(RuntimeError("Found 0 snippet folders in subfolders of: " + root + "\n"
                               "Expected folders containing .tif or .tiff files"))

        self.root = root
        self.snippets = snippets
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.num_frames = num_frames
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (video_snippet, target) where video_snippet is a tensor of shape 
                   (num_frames, 6, height, width) - 6 channels for lap and res components
        """
        snippet_path, target = self.snippets[index]
        
        # Get all .tif files in the snippet folder
        tif_files = sorted([f for f in os.listdir(snippet_path) 
                           if f.lower().endswith(('.tif', '.tiff'))])
        
        # Ensure we have exactly num_frames files
        if len(tif_files) < self.num_frames:
            # Pad with last frame if not enough frames
            while len(tif_files) < self.num_frames:
                tif_files.append(tif_files[-1])
        elif len(tif_files) > self.num_frames:
            # Take first num_frames if too many
            tif_files = tif_files[:self.num_frames]
        
        # Load and process all frames
        lap_frames = []
        res_frames = []
        
        for tif_file in tif_files:
            frame_path = os.path.join(snippet_path, tif_file)
              # Load frame
            img = cv2.imread(frame_path)
            if img is None:
                raise ValueError(f"Could not load frame: {frame_path}")
            
            # Apply frequency decomposition
            lap, res = FD(img, size=self.image_size)
              # Apply transforms
            if self.transform is not None:
                lap = self.transform(lap)
                res = self.transform(res)
            else:
                # Convert to tensor if no transform applied
                lap = ensure_tensor(lap)
                res = ensure_tensor(res)
            
            lap_frames.append(lap)
            res_frames.append(res)
        
        # Stack frames into tensors
        lap_tensor = torch.stack(lap_frames)  # (num_frames, 3, H, W)
        res_tensor = torch.stack(res_frames)  # (num_frames, 3, H, W)
        
        return lap_tensor, res_tensor, target

    def __len__(self):
        return len(self.snippets)


class VideoSnippetDatasetAug(VideoSnippetDataset):
    """
    Video snippet dataset with data augmentation
    """
    
    def __init__(self, root, transform=None, transform_aug=None, num_frames=16, image_size=256):
        super(VideoSnippetDatasetAug, self).__init__(root, transform, num_frames)
        self.transform_aug = transform_aug
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)

    def __getitem__(self, index):
        """
        Returns:
            tuple: (lap_tensor, res_tensor, aug_tensor, target)
        """
        snippet_path, target = self.snippets[index]
        
        # Get all .tif files in the snippet folder
        tif_files = sorted([f for f in os.listdir(snippet_path) 
                           if f.lower().endswith(('.tif', '.tiff'))])
          # Ensure we have exactly num_frames files
        if len(tif_files) < self.num_frames:
            while len(tif_files) < self.num_frames:
                tif_files.append(tif_files[-1])
        elif len(tif_files) > self.num_frames:
            tif_files = tif_files[:self.num_frames]        # Apply temporal augmentation to frame order
        tif_files = TemporalAugmentation.apply_temporal_jitter(tif_files, probability=0.3)
        tif_files = TemporalAugmentation.apply_frame_skip(tif_files, probability=0.2)
        tif_files = TemporalAugmentation.apply_temporal_reverse(tif_files, probability=0.15)
        
        # Load and process all frames
        lap_frames = []
        res_frames = []
        aug_frames = []
        
        for tif_file in tif_files:
            frame_path = os.path.join(snippet_path, tif_file)
            
            # Load frame
            img = cv2.imread(frame_path)
            if img is None:
                raise ValueError(f"Could not load frame: {frame_path}")
            
            # Apply frequency decomposition
            lap, res = FD(img, size=self.image_size)
            
            # Apply normal transforms
            if self.transform is not None:
                lap = self.transform(lap)
                res = self.transform(res)
            else:
                # Convert to tensor if no transform applied
                lap = ensure_tensor(lap)
                res = ensure_tensor(res)
            
            # Create augmented version
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.transform_aug is not None:
                aug_img = self.transform_aug(img_pil)
            else:
                aug_img = self.transform(img_pil) if self.transform else img_pil
            
            lap_frames.append(lap)
            res_frames.append(res)
            aug_frames.append(aug_img)
        
        # Ensure all frames are tensors before stacking
        lap_frames = [ensure_tensor(frame) for frame in lap_frames]
        res_frames = [ensure_tensor(frame) for frame in res_frames]
        aug_frames = [ensure_tensor(frame) for frame in aug_frames]
        
        # Stack frames into tensors
        lap_tensor = torch.stack(lap_frames)  # (num_frames, 3, H, W)
        res_tensor = torch.stack(res_frames)  # (num_frames, 3, H, W)
        aug_tensor = torch.stack(aug_frames)  # (num_frames, 3, H, W)
        
        return lap_tensor, res_tensor, aug_tensor, target


class VideoSnippetDatasetEnhanced(VideoSnippetDataset):
    """
    Video snippet dataset with general video augmentation
    """
    
    def __init__(self, root, transform=None, transform_aug=None, num_frames=16, temporal_mode='conservative', image_size=256):
        super(VideoSnippetDatasetEnhanced, self).__init__(root, transform, num_frames, image_size)
        self.transform_aug = transform_aug
        self.temporal_mode = temporal_mode
    
    def __getitem__(self, index):
        """
        Returns:
            tuple: (lap_tensor, res_tensor, aug_tensor, target)
        """
        snippet_path, target = self.snippets[index]
        
        # Get all .tif files in the snippet folder
        tif_files = sorted([f for f in os.listdir(snippet_path) 
                           if f.lower().endswith(('.tif', '.tiff'))])
        
        # Ensure we have exactly num_frames files
        if len(tif_files) < self.num_frames:
            while len(tif_files) < self.num_frames:
                tif_files.append(tif_files[-1])
        elif len(tif_files) > self.num_frames:
            tif_files = tif_files[:self.num_frames]
          # Apply general temporal augmentation
        from lib.data.video_augmentation import TemporalAugmentation
        
        if self.temporal_mode == 'minimal':            # Almost no temporal augmentation
            tif_files = TemporalAugmentation.apply_minimal_temporal_jitter(tif_files, probability=0.1)
        elif self.temporal_mode == 'moderate':
            # More temporal variation
            tif_files = TemporalAugmentation.apply_minimal_temporal_jitter(tif_files, probability=0.25)
            tif_files = TemporalAugmentation.apply_object_speed_variation(tif_files, probability=0.2)
        else:  # conservative
            # Balanced approach
            tif_files = TemporalAugmentation.apply_minimal_temporal_jitter(tif_files, probability=0.15)
            tif_files = TemporalAugmentation.apply_object_speed_variation(tif_files, probability=0.1)
        
        # Load and process all frames
        lap_frames = []
        res_frames = []
        aug_frames = []
        
        for tif_file in tif_files:
            frame_path = os.path.join(snippet_path, tif_file)
            
            # Load frame
            img = cv2.imread(frame_path)
            if img is None:
                raise ValueError(f"Could not load frame: {frame_path}")
              # Apply frequency decomposition
            lap, res = FD(img, size=self.image_size)
              # Apply normal transforms
            if self.transform is not None:
                lap = self.transform(lap)
                res = self.transform(res)
            else:
                # Convert to tensor if no transform applied
                lap = ensure_tensor(lap)
                res = ensure_tensor(res)
            
            # Create augmented version
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.transform_aug is not None:
                aug_img = self.transform_aug(img_pil)
            else:
                aug_img = self.transform(img_pil) if self.transform else img_pil
            
            lap_frames.append(lap)
            res_frames.append(res)
            aug_frames.append(aug_img)
        
        # Stack frames into tensors
        lap_tensor = torch.stack(lap_frames)  # (num_frames, 3, H, W)
        res_tensor = torch.stack(res_frames)  # (num_frames, 3, H, W)
        aug_tensor = torch.stack(aug_frames)  # (num_frames, 3, H, W)
        
        return lap_tensor, res_tensor, aug_tensor, target
