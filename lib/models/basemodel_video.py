import os
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

from lib.models.networks import weights_init, define_G, define_D, get_scheduler
from lib.visualizer import Visualizer
from lib.evaluate import roc


class BaseModel_Video():
    """ Base Model for OCR-GAN Video
    """
    def __init__(self, opt, data, classes):
        # Seed for deterministic behavior
        self.seed(opt.manualseed)        # Initialize variables
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.data = data
        self.classes = classes
        self.name = opt.name
        
        # Set device based on options - use first GPU ID if available and GPU mode
        if opt.device != 'cpu' and len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
        else:
            self.device = torch.device("cpu")
            
        self.trn_dir = os.path.join(opt.outf, opt.name, 'train')
        self.tst_dir = os.path.join(opt.outf, opt.name, 'test')
        self.num_frames = opt.num_frames if hasattr(opt, 'num_frames') else 16        # Initialize networks - define_G and define_D already handle device placement
        self.netg = define_G(opt)
        self.netd = define_D(opt)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        # Optimizers and schedulers
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.schedulers = [get_scheduler(self.optimizer_g, opt), get_scheduler(self.optimizer_d, opt)]
        self.optimizers = [self.optimizer_g, self.optimizer_d]

        # Buffers for video data
        self.total_steps = 0
        self.epoch = 0
        self.input_lap = torch.FloatTensor().to(self.device)
        self.input_res = torch.FloatTensor().to(self.device)
        self.fake_aug = torch.FloatTensor().to(self.device)
        self.gt = torch.FloatTensor().to(self.device)
        self.label = torch.FloatTensor().to(self.device)
        self.noise = torch.FloatTensor(opt.batchsize, self.num_frames, opt.nz, 1, 1).to(self.device)
        self.fixed_input_lap = torch.FloatTensor().to(self.device)
        self.fixed_input_res = torch.FloatTensor().to(self.device)

    def seed(self, seed_value):
        if seed_value == -1:
            return
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def set_input(self, input: torch.Tensor, noise: bool = False):
        with torch.no_grad():
            # For video data: input is (lap_tensor, res_tensor, [aug_tensor], target)
            if len(input) == 4:  # With augmentation
                lap_tensor, res_tensor, aug_tensor, target = input
                self.fake_aug.resize_(aug_tensor.size()).copy_(aug_tensor)
            else:  # Without augmentation
                lap_tensor, res_tensor, target = input
                
            self.input_lap.resize_(lap_tensor.size()).copy_(lap_tensor)
            self.input_res.resize_(res_tensor.size()).copy_(res_tensor)
            self.gt.resize_(target.size()).copy_(target)
            self.label.resize_(target.size())
            
            if noise:
                self.noise.data.copy_(torch.randn(self.noise.size()))
            if self.total_steps == self.opt.batchsize:
                self.fixed_input_lap.resize_(lap_tensor.size()).copy_(lap_tensor)
                self.fixed_input_res.resize_(res_tensor.size()).copy_(res_tensor)

    def get_errors(self):
        return OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_lat', self.err_g_lat.item())
        ])

    def reinit_d(self):
        self.netd.apply(weights_init)
        print('Reloading d net')

    def get_current_images(self):
        # Return middle frame from video sequences
        middle_frame = self.num_frames // 2
        reals = self.input_lap[:, middle_frame].data + self.input_res[:, middle_frame].data
        fakes = self.fake[:, middle_frame].data
        fake_lap = self.fake_lap[:, middle_frame].data
        fake_res = self.fake_res[:, middle_frame].data
        return reals, fakes, fake_lap, fake_res

    def save_weights(self, epoch: int, is_best: bool = False):
        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        os.makedirs(weight_dir, exist_ok=True)
        if is_best:
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f'{weight_dir}/netG_best.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f'{weight_dir}/netD_best.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f"{weight_dir}/netD_{epoch}.pth")
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f"{weight_dir}/netG_{epoch}.pth")

    def load_weights(self, epoch=None, is_best: bool = False, path=None):
        if epoch is None and is_best is False:
            raise Exception('Please provide epoch to be loaded or choose the best epoch.')

        fname_g = "netG_best.pth" if is_best else f"netG_{epoch}.pth"
        fname_d = "netD_best.pth" if is_best else f"netD_{epoch}.pth"

        if path is None:
            path_g = f"./output/{self.name}/{self.opt.dataset}/train/weights/{fname_g}"
            path_d = f"./output/{self.name}/{self.opt.dataset}/train/weights/{fname_d}"

        print('>> Loading weights...')
        weights_g = torch.load(path_g, map_location=self.device)['state_dict']
        weights_d = torch.load(path_d, map_location=self.device)['state_dict']
        self.netg.load_state_dict(weights_g, strict=False)
        self.netd.load_state_dict(weights_d, strict=False)
        print('   Done.')

    def train_one_epoch(self):
        self.netg.train()
        epoch_iter = 0
        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch + 1, self.opt.niter))
        
        for data in tqdm(self.data.train, leave=False, total=len(self.data.train)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize
            self.set_input(data)
            self.optimize_params()
            
            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.data.train.dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)
                    
            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed, _ = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

    def train(self):
        self.total_steps = 0
        best_auc = 0
        print(f">> Training {self.name} on {self.classes} to detect {self.opt.note}")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            self.train_one_epoch()
            res = self.test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.save_weights(self.epoch, is_best=True)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)
        return best_auc

    def test(self):
        with torch.no_grad():
            if self.opt.load_weights:
                path = f"./output/{self.name}/{self.opt.dataset}/train/weights/netG.pth"
                pretrained_dict = torch.load(path)['state_dict']
                self.netg.load_state_dict(pretrained_dict, strict=False)
                print('   Loaded weights.')

            self.opt.phase = 'test'
            self.an_scores = torch.zeros(len(self.data.valid.dataset), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(len(self.data.valid.dataset), dtype=torch.long, device=self.device)
            self.latent_i = torch.zeros((len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o = torch.zeros((len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            
            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)
                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()
                b_size = error.size(0)
                idx_start = i * self.opt.batchsize
                idx_end = idx_start + b_size
                self.an_scores[idx_start:idx_end] = error.reshape(b_size)
                self.gt_labels[idx_start:idx_end] = self.gt.reshape(b_size)
                self.latent_i[idx_start:idx_end, :] = latent_i.reshape(b_size, self.opt.nz)
                self.latent_o[idx_start:idx_end, :] = latent_o.reshape(b_size, self.opt.nz)

                self.times.append(time_o - time_i)

                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    os.makedirs(dst, exist_ok=True)
                    real, fake, _ = self.get_current_images()
                    import torchvision.utils as vutils
                    vutils.save_image(real, f'{dst}/real_{i + 1:03d}.eps', normalize=True)
                    vutils.save_image(fake, f'{dst}/fake_{i + 1:03d}.eps', normalize=True)

            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = roc(self.gt_labels, self.an_scores)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            # For modern PyTorch, this should be called after optimizer.step()
            # Since we're at epoch level, this is acceptable
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('   LR = %.7f \n' % lr)
