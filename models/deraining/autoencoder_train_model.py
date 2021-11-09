import torch

import torch.nn as nn

from models.base_model import BaseModel
from models.deraining import build_networks

from util.standard_derain_metrics import SSIM_Derain_GPU, PSNR_Derain_GPU
from models.deraining.RLCN import RLCN

from models.deraining.loss_func import SSIM_Loss


class AutoencoderTrainModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(norm='batch')
        parser.add_argument('--gradient_clipping', type=float, default=-1,
                            help='gradient clipping for lstm network')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        self.test_loss_names = ['ssim', 'psnr']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        self.visual_names = ['rain_map', 'output_rain_residual', 'real_A', 'fake_B', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['Autoencoder']
        else:  # during test time, only load G
            self.model_names = ['Autoencoder']

        self.lcn_window_size = self.opt.lcn_window_size

        self.upsampling = 'bilinear'

        self.SSIM_metric = SSIM_Derain_GPU().cuda()
        self.PSNR_metric = PSNR_Derain_GPU().cuda()

        self.netAutoencoder = build_networks.define_G('Autoencoder', opt.norm,
                                                      opt.init_type, opt.init_gain, self.gpu_ids,
                                                      opt.pool,
                                                      opt.leaky,
                                                      upsampling=self.upsampling,
                                                      n1=opt.nb_filter,
                                                      init=not opt.not_init,
                                                      return_embedding=False)
        if self.isTrain:
            self.loss_names.append('G_Self')

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = SSIM_Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netAutoencoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.rain_map = self.real_A - self.real_B

        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output_rain_residual = self.netAutoencoder(self.rain_map)
        self.fake_B = self.real_A - self.output_rain_residual

    # calculate loss only used in printing (no grad)
    def cal_test_loss(self):
        self.loss_ssim = self.SSIM_metric(self.fake_B, self.real_B)
        self.loss_psnr = self.PSNR_metric(self.fake_B, self.real_B)

    def cal_G_loss(self):

        self.loss_G = 0.

        # self.loss_G_Self = self.criterionSSIM(self.real_A - self.output_rain_residual, self.real_B)
        self.loss_G_Self = self.criterionL2(self.output_rain_residual, self.rain_map)
        self.loss_G += self.loss_G_Self

    def backward_G(self):
        self.loss_G.backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()
        self.cal_G_loss()
        self.backward_G()

        if self.opt.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.netAutoencoder.parameters(), self.opt.gradient_clipping)
        self.optimizer_G.step()
