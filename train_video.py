from options import Options
from lib.data.data_loading import load_video_data_FD_aug, load_video_data_FD_enhanced
from lib.models import load_model
import numpy as np
import prettytable as pt

def train_video(opt, dataset_name):
    """Train video model with general-purpose augmentation"""
    # Check if enhanced video augmentation is requested
    if hasattr(opt, 'use_video_augmentation') and opt.use_video_augmentation:
        print(f"🎯 Using enhanced video augmentation ({opt.video_augmentation} mode)")
        data = load_video_data_FD_enhanced(opt, dataset_name, augmentation_mode=opt.video_augmentation)
    else:
        print(f"🎬 Using general video augmentation")
        data = load_video_data_FD_aug(opt, dataset_name)
    
    model = load_model(opt, data, dataset_name)
    auc = model.train()
    return auc

def main():
    """ Training for video model
    """
    # Set up for any video dataset
    dataset_name = "ucsd2"
    opt = Options().parse()
    # Override model to use video version
    opt.model = 'gan_model'
    opt.dataset = dataset_name
    opt.num_frames = 8  # Number of frames per snippet
    
    print(f"Training GAN Video on {dataset_name}")
    auc = train_video(opt, dataset_name)
    print(f"Training completed. AUC: {auc:.4f}")

if __name__ == '__main__':
    main()