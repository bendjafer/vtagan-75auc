"""
Utility functions for video anomaly detection data processing.
Contains common functions for data loading, transformations, and file handling.
"""

import os
import torch
from torchvision import transforms
from PIL import Image
import random
import cv2
# Image file extensions
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']

def find_classes(dir):
    """
    Find the class folders in a dataset directory.
    
    Args:
        dir (str): Root directory path
        
    Returns:
        tuple: (classes, class_to_idx) where classes is a list of class names
               and class_to_idx is a dictionary mapping class names to indices
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def FD(img, size=None):
    """
    Frequency Decomposition with dynamic sizing
    Args:
        img: Input image
        size: Target size (width, height). If None, defaults to 256x256
    """
    if size is None:
        size = (256, 256)
    elif isinstance(size, int):
        size = (size, size)
    
    img = cv2.resize(img, size)
    #img_resize = cv2.resize(img, (128,128))
    img_resize = cv2.pyrDown(img)
    temp_pyrUp = cv2.pyrUp(img_resize)
    #pdb.set_trace()
    temp_lap = cv2.subtract(img, temp_pyrUp)
    temp_lap = Image.fromarray(temp_lap)
    temp_pyrUp = Image.fromarray(temp_pyrUp)
    return temp_lap, temp_pyrUp

class AspectRatioResize:
    """
    Resize images while maintaining aspect ratio.
    """
    def __init__(self, size, method='crop'):
        """
        Args:
            size (tuple): Target size (width, height)
            method (str): 'crop' or 'pad'
        """
        self.size = size
        self.method = method
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be resized
            
        Returns:
            PIL Image: Resized image
        """
        if self.method == 'crop':
            return self._resize_crop(img)
        elif self.method == 'pad':
            return self._resize_pad(img)
        else:
            return img.resize(self.size, Image.LANCZOS)
    
    def _resize_crop(self, img):
        """Resize and center crop to target size."""
        target_w, target_h = self.size
        img_w, img_h = img.size
        
        # Calculate ratios
        ratio_w = target_w / img_w
        ratio_h = target_h / img_h
        ratio = max(ratio_w, ratio_h)
        
        # Resize based on the larger ratio
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop to target size
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        return resized.crop((left, top, right, bottom))
    
    def _resize_pad(self, img):
        """Resize and pad to target size."""
        target_w, target_h = self.size
        img_w, img_h = img.size
        
        # Calculate ratios
        ratio_w = target_w / img_w
        ratio_h = target_h / img_h
        ratio = min(ratio_w, ratio_h)
        
        # Resize based on the smaller ratio
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create new image with target size and paste resized image
        new_img = Image.new('RGB', self.size, (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_img.paste(resized, (paste_x, paste_y))
        
        return new_img

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
        
        return target_size
    else:
        return (base_size, base_size)
