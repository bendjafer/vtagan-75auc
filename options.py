import argparse
import os
import torch

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='ucsd2', help='folder')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')        
        self.parser.add_argument('--path', default='', help='path to the folder or image to be predicted.')
        self.parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--isize', type=int, default=32, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='gan_model', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=1, type=int, help='manual seed')
        self.parser.add_argument('--note', default='bad', help='note for experiments')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')

        ##

         # VAD Specific:
        self.parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video clip for VAD.') # New argument
        
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='gradient clipping norm (0 to disable)')
        self.parser.add_argument('--w_adv', type=float, default=1, help='Weight for adversarial loss. default=1')
        self.parser.add_argument('--w_con', type=float, default=50, help='Weight for reconstruction loss. default=50')
        self.parser.add_argument('--w_lat', type=float, default=1, help='Weight for latent space loss. default=1')
        
        # Temporal attention and loss weights for video models
        self.parser.add_argument('--use_temporal_attention', action='store_true', help='Enable temporal attention modules')
        self.parser.add_argument('--w_temporal_consistency', type=float, default=0.1, help='Weight for temporal consistency loss')
        self.parser.add_argument('--w_temporal_motion', type=float, default=0.05, help='Weight for temporal motion loss')
        self.parser.add_argument('--w_temporal_reg', type=float, default=0.01, help='Weight for temporal attention regularization')
        
        # Video aspect ratio handling
        self.parser.add_argument('--aspect_method', type=str, default='maintain_3_2', 
                                choices=['maintain_3_2', 'center_crop', 'pad_square', 'stretch'],
                                help='Method for handling video aspect ratio: '
                                     'maintain_3_2 (keep 3:2 ratio, recommended), '
                                     'center_crop (crop to square), '
                                     'pad_square (pad to square), '
                                     'stretch (original behavior, causes distortion)')
          # Video augmentation options
        self.parser.add_argument('--video_augmentation', type=str, default='conservative',
                                choices=['minimal', 'conservative', 'moderate'],                                help='Video augmentation mode: '
                                     'minimal (almost no augmentation), '
                                     'conservative (default, balanced), '
                                     'moderate (more variation)')
        self.parser.add_argument('--use_video_augmentation', action='store_true', 
                                help='Use specialized video augmentation instead of general video augmentation')
        
        # Optical Flow Enhancement options
        self.parser.add_argument('--use_optical_flow', action='store_true', default=True,
                                help='Enable optical flow feature fusion for motion-aware anomaly detection')
        self.parser.add_argument('--optical_flow_weight', type=float, default=0.2,
                                help='Weight for optical flow features in fusion (0.0-1.0)')
        self.parser.add_argument('--motion_magnitude_weight', type=float, default=0.1,
                                help='Weight for motion magnitude sensitivity (0.0-1.0)')
        
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
        self.opt = None

        
    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids - only if using GPU device and CUDA is available
        if len(self.opt.gpu_ids) > 0 and self.opt.device == 'gpu' and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])
        elif self.opt.device == 'cpu' or not torch.cuda.is_available():
            # Force CPU mode if explicitly requested or CUDA not available
            self.opt.gpu_ids = []

        args = vars(self.opt)

        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt