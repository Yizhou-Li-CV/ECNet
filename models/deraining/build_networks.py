import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from models.deraining.networks import ECNetLL, ECNet, Autoencoder


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'sync_batch':
        norm_layer = functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_pool_layer(pool_type='max'):
    if pool_type == 'max':
        pool_layer = nn.MaxPool2d
    elif pool_type == 'avg':
        pool_layer = nn.AvgPool2d
    elif pool_type == 'conv':
        pool_layer = 'conv'
    else:
        raise NotImplementedError('pool layer [%s] is not found' % pool_type)
    return pool_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.2)
    elif opt.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.2)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], init=True):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(netG, norm='batch', init_type='normal', init_gain=0.02,
             gpu_ids=None, pool='max', leaky=True, upsampling='bilinear', return_att=False, n1=64,
             iters=6, init=True, return_embedding=False):
    if gpu_ids is None:
        gpu_ids = []
    norm_layer = get_norm_layer(norm_type=norm)
    pool_layer = get_pool_layer(pool_type=pool)

    if netG == 'ECNet':
        net = ECNet(norm_layer=norm_layer,
                    pool_layer=pool_layer, leaky=leaky,
                    return_att=return_att, n1=n1,
                    upsampling=upsampling,
                    return_embedding=return_embedding)
    elif netG == 'ECNetLL':
        net = ECNetLL(norm_layer=norm_layer,
                      pool_layer=pool_layer, leaky=leaky,
                      return_att=return_att, n1=n1,
                      iters=iters, upsampling=upsampling,
                      return_embedding=return_embedding)
    elif netG == 'Autoencoder':
        net = Autoencoder(norm_layer=norm_layer,
                          pool_layer=pool_layer, leaky=leaky,
                          return_att=return_att, n1=n1,
                          iters=iters, upsampling=upsampling,
                          return_embedding=return_embedding)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids, init)
