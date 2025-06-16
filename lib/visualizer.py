""" This file contains Visualizer class based on Facebook's visdom.

Returns:
    Visualizer(): Visualizer class to display plots and images
"""

##
import os
import time
import numpy as np
import torchvision.utils as vutils

##
class Visualizer():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """
    # pylint: disable=too-many-instance-attributes
    # Reasonable.

    ##
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)
        now  = time.strftime("%c")
        title = f'================ {now} ================\n'
        info  = f'{opt.note}, {opt.nz}, {opt.w_adv}, {opt.w_con}, {opt.w_lat}\n'
        self.write_to_log_file(text=title + info)


    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##
    def plot_current_errors(self, epoch, counter_ratio, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """

        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=4
        )

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None:
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=5
        )

    ##
    def print_current_errors(self, epoch, errors, print_to_console=False):
        """ Print current errors to log file (and optionally console).

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
            print_to_console (bool): Whether to also print to console
        """
        # message = '   [%d/%d] ' % (epoch, self.opt.niter)
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)

        # Only print to console if explicitly requested
        if print_to_console:
            print(message)
            
        # Always log to file
        with open(self.log_name, "a", encoding='utf-8') as log_file:
            log_file.write('%s\n' % message)

    ##
    def log_running_average(self, epoch, total_epochs, recent_losses, interval_batches):
        """Log running average of losses over a batch interval
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            total_epochs (int): Total number of epochs
            recent_losses (list): List of recent loss dictionaries
            interval_batches (int): Number of batches in this interval
        """
        if not recent_losses:
            return
            
        # Calculate running averages
        avg_losses = {}
        for key in recent_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in recent_losses])
        
        # Create concise message
        message = f"   Running Avg [{epoch+1}/{total_epochs}] ({interval_batches} batches):"
        for loss_name, loss_value in avg_losses.items():
            message += f" {loss_name}={loss_value:.4f}"
            
        print(message)
        self.write_to_log_file(message)

    ##
    def write_to_log_file(self, text):
        with open(self.log_name, "a", encoding='utf-8') as log_file:
            log_file.write('%s\n' % text)

    ##
    def log_epoch_training_summary(self, epoch, total_epochs, training_losses, training_time):
        """Log comprehensive training summary for each epoch
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            total_epochs (int): Total number of epochs
            training_losses (dict): Dictionary containing mean training losses
            training_time (float): Training time for this epoch in seconds
        """
        # Create training summary message
        message = f"[TRAIN] EPOCH {epoch+1}/{total_epochs}:"
        message += f" Time={training_time:.2f}s"
        
        # Add loss information
        for loss_name, loss_value in training_losses.items():
            message += f" {loss_name}={loss_value:.4f}"
            
        print(message)
        self.write_to_log_file(message)

    ##
    def log_epoch_testing_summary(self, epoch, total_epochs, test_metrics, best_auc, is_best_epoch=False):
        """Log comprehensive testing summary for each epoch
        
        Args:
            epoch (int): Current epoch number (0-indexed)
            total_epochs (int): Total number of epochs
            test_metrics (dict): Dictionary containing test metrics (AUC, runtime, etc.)
            best_auc (float): Best AUC achieved so far
            is_best_epoch (bool): Whether this epoch achieved new best AUC
        """
        # Create testing summary message
        message = f"[TEST] EPOCH {epoch+1}/{total_epochs}:"
        
        # Add test metrics
        for metric_name, metric_value in test_metrics.items():
            if metric_name == 'AUC':
                message += f" AUC={metric_value:.4f}"
            elif 'Time' in metric_name or 'Runtime' in metric_name:
                message += f" Time={metric_value:.1f}ms"
            else:
                message += f" {metric_name}={metric_value:.4f}"
        
        # Add best AUC tracking
        message += f" | Best_AUC={best_auc:.4f}"
        
        # Add indicator for best epoch
        if is_best_epoch:
            message += " ** NEW BEST! **"
            
        print(message)
        self.write_to_log_file(message)
        
        # Also log a separator for clarity between epochs
        separator = "=" * 80
        self.write_to_log_file(separator)

    ##
    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)
        message += 'max AUC: %.3f' % best

        print(message)
        self.write_to_log_file(text=message)

    def display_current_images(self, reals, fakes, fixed):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        # fixed = self.normalize(fixed.cpu().numpy())

        self.vis.images(reals, win=1, opts={'title': 'Reals'})
        self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        # self.vis.images(fixed, win=3, opts={'title': 'Fixed'})

    def save_current_images(self, epoch, reals, fakes, fixed, extra=None):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image or additional output
            extra ([FloatTensor], optional): Additional output for video models
        """
        vutils.save_image(reals, '%s/reals.png' % self.img_dir, normalize=True)
        vutils.save_image(fakes, '%s/fakes.png' % self.img_dir, normalize=True)
        vutils.save_image(fixed, '%s/fixed_fakes_%03d.png' %(self.img_dir, epoch+1), normalize=True)
        
        # Save additional output if provided (for video models)
        if extra is not None:
            vutils.save_image(extra, '%s/extra_output_%03d.png' %(self.img_dir, epoch+1), normalize=True)
