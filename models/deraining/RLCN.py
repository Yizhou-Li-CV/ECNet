import torch.nn.functional as F
import torch


def RLCN(img, kSize=9, input_nc=3, output_nc=3, noise_mask=None, cuda=True):
    '''
        Args:
            img : N * C * H * W
            kSize : 9 * 9
    '''

    if cuda:
        w = torch.ones((output_nc, input_nc, kSize, kSize)).cuda()
        N_counter = torch.ones_like(img).cuda()
    else:
        w = torch.ones((output_nc, input_nc, kSize, kSize))
        N_counter = torch.ones_like(img)

    N = F.conv2d(input=N_counter, weight=w, padding=kSize // 2)

    epsilon = 1e-5

    mean_local = F.conv2d(input=img, weight=w, padding=kSize // 2)

    mean_square_local = F.conv2d(input=img ** 2, weight=w, padding=kSize // 2)

    std_local = (mean_square_local - mean_local ** 2 / N) / (N - 1) + epsilon

    # print(std_local[std_local < 0])
    if torch.isnan(torch.sum(std_local)):
        print('std_local before sqrt is nan')
    std_local = torch.sqrt(std_local)
    if torch.isnan(torch.sum(std_local)):
        print('std_local after sqrt is nan')

    if noise_mask is None:
        return (img - mean_local / N) / (std_local + epsilon), mean_local, std_local
    else:
        return (img - mean_local / N) / (std_local + epsilon) * noise_mask, mean_local, std_local
