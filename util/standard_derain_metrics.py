from numpy.lib.stride_tricks import as_strided as ast

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp


def RGBTensor_to_Y(tensor):
    n, c, h, w = tensor.size()
    if n > 1:
        assert 'Test batch size is larger than 1!'

    # convert the tensor to an uint8 image
    img = tensor.detach()[0]
    img = (img + 1.) / 2. * 255.
    img = torch.clamp(img, min=0., max=255.)
    img = torch.floor(img)

    # convert the RGB image into Y, by simulating the RGB->Y process in Matlab
    img_gray = img[[0]] * 0.299 + img[[1]] * 0.587 + img[[2]] * 0.114
    img_gray = img_gray / 255 * 219 + 16
    img_gray = torch.unsqueeze(img_gray, dim=0)

    return img_gray


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


class PSNR_Derain_GPU(torch.nn.Module):

    def __init__(self):
        super(PSNR_Derain_GPU, self).__init__()

    def forward(self, img1, img2):
        img1 = RGBTensor_to_Y(img1)
        img2 = RGBTensor_to_Y(img2)

        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100

        PIXEL_MAX = 255.0
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class SSIM_Derain_GPU(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_Derain_GPU, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, data_range=1.0):

        if data_range != 255:
            img1 = RGBTensor_to_Y(img1) / 255.
            img2 = RGBTensor_to_Y(img2) / 255.

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, data_range)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, data_range=1.0):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True, data_range=1.0):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, data_range)