import torch

import torch.nn as nn

from models.base_model import BaseModel
from models.deraining import build_networks

from util.standard_derain_metrics import SSIM_Derain_GPU, PSNR_Derain_GPU
from models.deraining.RLCN import RLCN

from models.deraining.loss_func import SSIM_Loss


class ECNetTrainTestModel(BaseModel):

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

        if is_train:
            parser.add_argument('--lambda_L1_embedding', type=float, default=0.02, help='weight for embedding L1 loss')
            parser.add_argument('--lambda_ssim_image', type=float, default=1,
                                help='weight for negative ssim loss')
            parser.add_argument('--lambda_L2_att', type=float, default=0.1,
                                help='weight for negative ssim loss')

        parser.add_argument('--mask_threshold', type=float, default=1e-3,
                            help='the threshold to select rain pixels')
        parser.add_argument('--iters', type=int, default=6,
                            help='the iterations of recurrent network')
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

        self.visual_names = ['real_A', 'fake_B', 'real_B', 'rlcn']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'Autoencoder']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.lcn_window_size = self.opt.lcn_window_size

        self.upsampling = 'bilinear'

        self.SSIM_metric = SSIM_Derain_GPU().cuda()
        self.PSNR_metric = PSNR_Derain_GPU().cuda()

        self.netG = build_networks.define_G(opt.netG, opt.norm,
                                            opt.init_type, opt.init_gain,
                                            self.gpu_ids,
                                            opt.pool,
                                            opt.leaky, upsampling=self.upsampling,
                                            n1=opt.nb_filter,
                                            iters=opt.iters, init=not opt.not_init,
                                            return_att=False, return_embedding=True)

        if self.isTrain:
            self.loss_names.append('G_Embedding')
            self.loss_names.append('G_Image')
            self.loss_names.append('G_Attention')
            self.netAutoencoder = build_networks.define_G('Autoencoder', opt.norm,
                                                          opt.init_type, opt.init_gain, self.gpu_ids,
                                                          opt.pool,
                                                          opt.leaky,
                                                          upsampling=self.upsampling,
                                                          n1=opt.nb_filter,
                                                          init=not opt.not_init,
                                                          return_embedding=True)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = SSIM_Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
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

        self.rain_mask = ((self.real_A - self.real_B) / 2.) > self.opt.mask_threshold
        self.rain_mask = self.rain_mask.float()
        self.rain_mask = torch.max(self.rain_mask, dim=1, keepdim=True)[0]

        self.image_paths = input['A_paths']

        real_A_before_LCN = (self.real_A + 1) / 2.0

        self.rlcn_r, _, _ = RLCN(real_A_before_LCN[:, [0], ...], kSize=self.lcn_window_size, input_nc=1, output_nc=1,
                                 cuda=True)
        self.rlcn_g, _, _ = RLCN(real_A_before_LCN[:, [1], ...], kSize=self.lcn_window_size, input_nc=1, output_nc=1,
                                 cuda=True)
        self.rlcn_b, _, _ = RLCN(real_A_before_LCN[:, [2], ...], kSize=self.lcn_window_size, input_nc=1, output_nc=1,
                                 cuda=True)
        self.rlcn = torch.cat([self.rlcn_r, self.rlcn_g, self.rlcn_b], dim=1)
        self.rlcn = torch.clamp(self.rlcn, min=0.0, max=1.0)

    def get_list_last_item(self, input):
        if isinstance(input, list):
            return input[-1]
        else:
            return input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.rain_residuals, self.embeddings, self.atts = self.netG(self.real_A, self.rlcn)
        self.rain_residual, self.embedding, self.att = self.get_list_last_item(self.rain_residuals), \
                                                       self.get_list_last_item(self.embeddings), \
                                                       self.get_list_last_item(self.atts)
        self.fake_B = self.real_A - self.rain_residual

        if self.isTrain:
            self.fake_rainmap, self.autoencoder_embedding = self.netAutoencoder(self.rain_map)
            if isinstance(self.rain_residuals, list):
                self.recurrent_sup = True
                self.fake_Bs = [self.real_A - self.rain_residuals[i] for i in range(len(self.rain_residuals))]
            else:
                self.recurrent_sup = False

    # calculate loss only used in printing (no grad)
    def cal_test_loss(self):
        self.loss_ssim = self.SSIM_metric(self.fake_B, self.real_B)
        self.loss_psnr = self.PSNR_metric(self.fake_B, self.real_B)

    def cal_G_loss(self):

        self.loss_G = 0.

        self.loss_G_Embedding = self.criterionL1(self.autoencoder_embedding,
                                                 self.embedding) * self.opt.lambda_L1_embedding
        self.loss_G += self.loss_G_Embedding

        self.loss_G_Image = -self.criterionSSIM((self.fake_B + 1) / 2.,
                                                (self.real_B + 1) / 2.) * self.opt.lambda_ssim_image
        self.loss_G += self.loss_G_Image

        self.loss_G_Attention = self.criterionL2(self.att, self.rain_mask) * self.opt.lambda_L2_att
        self.loss_G += self.loss_G_Attention

    def cal_G_loss_recur(self):

        self.loss_G = 0.
        self.loss_G_Image = 0.
        self.loss_G_Embedding = 0.
        self.loss_G_Attention = 0.

        self.stage_loss_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 1.5]

        for i in range(6):
            self.loss_G_Image += -self.criterionSSIM((self.fake_Bs[i] + 1) / 2.,
                                                     (self.real_B + 1) / 2.) * self.opt.lambda_ssim_image * self.stage_loss_weights[i]

            self.loss_G_Attention += self.criterionL2(self.atts[i], self.rain_mask) * self.opt.lambda_L2_att * self.stage_loss_weights[i]

            self.loss_G_Embedding += self.criterionL1(self.autoencoder_embedding, self.embeddings[i]) * self.opt.lambda_L1_embedding * self.stage_loss_weights[i]

        self.loss_G = self.loss_G_Image + self.loss_G_Attention + self.loss_G_Embedding

    def backward_G(self):
        self.loss_G.backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()
        if self.recurrent_sup:
            self.cal_G_loss_recur()
        else:
            self.cal_G_loss()
        self.backward_G()

        if self.opt.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.gradient_clipping)
        self.optimizer_G.step()
