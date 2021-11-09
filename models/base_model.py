import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from .deraining import build_networks
import time


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.save_dir_autoencoder = os.path.join(opt.checkpoints_dir, opt.autoencoder_checkpoint)
        self.epoch_autoencoder = 'latest'
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.test_loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def cal_test_loss(self):
        """calculate loss only used for printing"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def hasNumbers(self, str_input):
        return any(char.isdigit() for char in str_input)

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [build_networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain:
            self.continue_train = False
        else:
            self.continue_train = opt.continue_train
        self.load_networks()
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def zero_grad_and_step(self):
        for i in range(len(self.optimizers)):
            self.optimizers[i].zero_grad()
            self.optimizers[i].step()

    def test(self, no_grad=True):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        if no_grad is True:
            with torch.no_grad():
                torch.cuda.synchronize()
                t1 = time.time()
                self.forward()
                torch.cuda.synchronize()
                t2 = time.time()
                interval = t2 - t1
                self.compute_visuals()
                return interval
        else:
            torch.cuda.synchronize()
            t1 = time.time()
            self.forward()
            torch.cuda.synchronize()
            t2 = time.time()
            interval = t2 - t1
            self.compute_visuals()
            return interval

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self, epoch=None):
        """Update learning rates for all the networks; called at the end of every epoch"""
        self.epoch = epoch
        for scheduler in self.schedulers:
            scheduler.step(epoch)

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        return lr

    def get_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self, in_test=False):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        if not in_test:
            for name in self.loss_names:
                if isinstance(name, str):
                    errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        for name in self.test_loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        # if train autoencoder, not load model
        if len(self.model_names) == 1 and 'autoencoder' == self.model_names[0].lower():
            return

        for name in self.model_names:
            if isinstance(name, str):
                if 'autoencoder' == name.lower():
                    # load autoencoder's weight
                    load_filename = '%s_net_%s.pth' % (str(self.epoch_autoencoder), 'Autoencoder')
                    load_path = os.path.join(self.save_dir_autoencoder, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('Load Autoencoder...')
                    print('loading the model from %s' % load_path, name)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    net.load_state_dict(state_dict)
                else:
                    # load ECNet's weights
                    if not self.continue_train and self.isTrain:
                        # if it is first time for training, load autoencoder's decoder weights to ECNet
                        # print('Load Autoencoder...')
                        load_filename = '%s_net_%s.pth' % (str(self.epoch_autoencoder), 'Autoencoder')
                        load_path = os.path.join(self.save_dir_autoencoder, load_filename)
                    else:
                        # if continue training, load ECNet's weight of a specific epoch
                        load_filename = '%s_net_%s.pth' % (self.opt.epoch, 'G')
                        load_path = os.path.join(self.save_dir, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path, name)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    print('original keys', state_dict.keys())
                    if not self.continue_train and self.isTrain:
                        # load autoencoder's decoder weights to ECNet
                        state_dict_new = dict({})
                        for key in state_dict.keys():
                            if 'Up' in key:
                                # load weights for decoder
                                state_dict_new[key] = state_dict[key]
                            if 'Conv.' in key:
                                # load weights of last Conv
                                state_dict_new[key] = state_dict[key]
                        print('new keys', state_dict_new.keys())
                        model_dict = net.state_dict()
                        model_dict.update(state_dict_new)
                        net.load_state_dict(model_dict)
                    else:
                        # load ECNet's weight of a specific epoch
                        net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
