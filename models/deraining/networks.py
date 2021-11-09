from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import init


class RB(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, down_ch=False):
        super(RB, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch),
            relu(*param),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch))

        if down_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
                norm_layer(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu_out = relu(*param)

    def forward(self, x):

        identity = x

        x = self.conv(x)
        x = x + self.shortcut(identity)
        x = self.relu_out(x)

        return x


class ConvLSTM(nn.Module):

    def __init__(self, nb_channel, softsign):
        super(ConvLSTM, self).__init__()

        self.conv_i = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        self.conv_f = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        self.conv_g = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        self.conv_o = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)

        init.orthogonal(self.conv_i.weight)
        init.orthogonal(self.conv_f.weight)
        init.orthogonal(self.conv_g.weight)
        init.orthogonal(self.conv_o.weight)

        init.constant(self.conv_i.bias, 0.)
        init.constant(self.conv_f.bias, 0.)
        init.constant(self.conv_g.bias, 0.)
        init.constant(self.conv_o.bias, 0.)

        self.conv_ii = nn.Sequential(self.conv_i, nn.Sigmoid())
        self.conv_ff = nn.Sequential(self.conv_f, nn.Sigmoid())
        if not softsign:
            self.conv_gg = nn.Sequential(self.conv_g, nn.Tanh())
        else:
            self.conv_gg = nn.Sequential(self.conv_g, nn.Softsign())

        self.conv_oo = nn.Sequential(self.conv_o, nn.Sigmoid())

        self.nb_channel = nb_channel

    def forward(self, input, prev_h, prev_c):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        if prev_h is None:
            prev_h = Variable(torch.zeros(batch_size, self.nb_channel, row, col)).cuda()
            prev_c = Variable(torch.zeros(batch_size, self.nb_channel, row, col)).cuda()

        x = torch.cat((input, prev_h), 1)
        i = self.conv_ii(x)
        f = self.conv_ff(x)
        g = self.conv_gg(x)
        o = self.conv_oo(x)
        c = f * prev_c + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvGRU(nn.Module):

    def __init__(self, nb_channel, softsign):
        super(ConvGRU, self).__init__()

        self.conv_z = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)

        self.conv_b = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)

        self.conv_g = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)


        init.orthogonal(self.conv_z.weight)
        init.orthogonal(self.conv_b.weight)
        init.orthogonal(self.conv_g.weight)

        init.constant(self.conv_z.bias, 0.)
        init.constant(self.conv_b.bias, 0.)
        init.constant(self.conv_g.bias, 0.)

        self.conv_zz = nn.Sequential(self.conv_z, nn.Sigmoid())
        self.conv_bb = nn.Sequential(self.conv_b, nn.Sigmoid())
        if not softsign:
            self.conv_gg = nn.Sequential(self.conv_g, nn.Tanh())
        else:
            self.conv_gg = nn.Sequential(self.conv_g, nn.Softsign())

        self.nb_channel = nb_channel

    def forward(self, input, prev_h):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        if prev_h is None:
            prev_h = Variable(torch.zeros(batch_size, self.nb_channel, row, col)).cuda()

        x1 = torch.cat((input, prev_h), 1)
        z = self.conv_zz(x1)
        b = self.conv_bb(x1)
        s = b * prev_h
        s = torch.cat((s, input), 1)
        g = self.conv_gg(s)
        h = (1 - z) * prev_h + z * g

        return h


class RRB_LSTM(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, down_ch=False, softsign=False):
        super(RRB_LSTM, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch),
            relu(*param))

        self.ConvLSTM = ConvLSTM(out_ch, softsign=softsign)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch))

        if down_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
                norm_layer(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu_out = relu(*param)

    def forward(self, x, h, c):
        identity = x

        x = self.conv1(x)
        h, c = self.ConvLSTM(x, h, c)
        x = h
        x = self.conv2(x)

        x = self.shortcut(identity) + x
        x = self.relu_out(x)

        return x, h, c


class RRB_GRU(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, down_ch=False, softsign=False):
        super(RRB_GRU, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch),
            relu(*param))

        self.ConvGRU = ConvGRU(out_ch, softsign=softsign)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch))

        if down_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
                norm_layer(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu_out = relu(*param)

    def forward(self, x, h):
        identity = x

        x = self.conv1(x)
        h = self.ConvGRU(x, h)
        x = h
        x = self.conv2(x)

        x = self.shortcut(identity) + x
        x = self.relu_out(x)

        return x, h


class Up(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, bilinear=True):
        super(Up, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(out_ch),
                relu(*param)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                norm_layer(out_ch),
                relu(*param)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class MaskGAM(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int, norm_layer, leaky=True):
        super(MaskGAM, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            norm_layer(1),
            nn.Sigmoid()
        )

        if leaky:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out, psi


class Autoencoder(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=True,
                 upsampling='bilinear', return_att=False, n1=32, iters=3, return_embedding=False):
        super(Autoencoder, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.filters = filters

        self.iters = iters

        self.return_att = return_att
        self.return_embedding = return_embedding

        if pool_layer != 'conv':
            self.Maxpool1 = pool_layer(kernel_size=2, stride=2)
            self.Maxpool2 = pool_layer(kernel_size=2, stride=2)
            self.Maxpool3 = pool_layer(kernel_size=2, stride=2)
        else:
            self.Maxpool1 = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=2, stride=2, padding=0)
            self.Maxpool2 = nn.Conv2d(self.filters[1], self.filters[1], kernel_size=2, stride=2, padding=0)
            self.Maxpool3 = nn.Conv2d(self.filters[2], self.filters[2], kernel_size=2, stride=2, padding=0)

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.Conv_input = nn.Sequential(nn.Conv2d(3, self.filters[0], kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(self.filters[0]),
                                        relu(*param))

        self.Conv1 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=False)
        self.Conv2 = RB(self.filters[0], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Conv3 = RB(self.filters[1], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Conv4 = RB(self.filters[2], self.filters[3], norm_layer, leaky=leaky, down_ch=True)

        self.Up4 = Up(self.filters[3], self.filters[2], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv4 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)

        self.Up3 = Up(self.filters[2], self.filters[1], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv3 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)

        self.Up2 = Up(self.filters[1], self.filters[0], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)

        self.Conv = nn.Conv2d(self.filters[0], 3, kernel_size=1, stride=1, padding=0)

    def forward(self, input):

        x = input

        x_in = x

        x_in = self.Conv_input(x_in)

        e1 = self.Conv1(x_in)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        d4 = self.Up4(e4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        x = out

        if self.return_embedding:
            return x, e4
        else:
            return x


class ECNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=True,
                 upsampling='bilinear', return_att=False, n1=32, return_embedding=False):
        super(ECNet, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.filters = filters

        self.return_att = return_att
        self.return_embedding = return_embedding

        if pool_layer != 'conv':
            self.Maxpool1 = pool_layer(kernel_size=2, stride=2)
            self.Maxpool2 = pool_layer(kernel_size=2, stride=2)
            self.Maxpool3 = pool_layer(kernel_size=2, stride=2)
        else:
            self.Maxpool1 = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=2, stride=2, padding=0)
            self.Maxpool2 = nn.Conv2d(self.filters[1], self.filters[1], kernel_size=2, stride=2, padding=0)
            self.Maxpool3 = nn.Conv2d(self.filters[2], self.filters[2], kernel_size=2, stride=2, padding=0)

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.Conv_input = nn.Sequential(nn.Conv2d(6, self.filters[0], kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(self.filters[0]),
                                        relu(*param))

        self.Conv1 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=False)
        self.Conv2 = RB(self.filters[0], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Conv3 = RB(self.filters[1], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Conv4 = RB(self.filters[2], self.filters[3], norm_layer, leaky=leaky, down_ch=True)

        self.Up4 = Up(self.filters[3], self.filters[2], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Att4 = MaskGAM(F_g=self.filters[2], F_l=self.filters[2], F_int=self.filters[1],
                            norm_layer=norm_layer, leaky=leaky)

        self.Up_conv4 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv4_2 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)

        self.Up3 = Up(self.filters[2], self.filters[1], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Att3 = MaskGAM(F_g=self.filters[1], F_l=self.filters[1], F_int=self.filters[0],
                            norm_layer=norm_layer, leaky=leaky)

        self.Up_conv3 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv3_2 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)

        self.Up2 = Up(self.filters[1], self.filters[0], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Att2 = MaskGAM(F_g=self.filters[0], F_l=self.filters[0], F_int=32,
                            norm_layer=norm_layer, leaky=leaky)

        self.Up_conv2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv2_2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)

        self.Conv = nn.Conv2d(self.filters[0], 3, kernel_size=1, stride=1, padding=0)

    def forward(self, input, lcn):

        x = input

        x_in = torch.cat([x, lcn], dim=1)

        x_in = self.Conv_input(x_in)

        e1 = self.Conv1(x_in)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        d4 = self.Up4(e4)
        x3, psi3 = self.Att4(g=d4, x=e3)
        d4 = self.Up_conv4(d4) + self.Up_conv4_2(x3)

        d3 = self.Up3(d4)
        x2, psi2 = self.Att3(g=d3, x=e2)
        d3 = self.Up_conv3(d3) + self.Up_conv3_2(x2)

        d2 = self.Up2(d3)
        x1, psi1 = self.Att2(g=d2, x=e1)
        d2 = self.Up_conv2(d2) + self.Up_conv2_2(x1)

        out_res = self.Conv(d2)

        x = out_res

        if self.return_embedding:
            return x, e4, psi1
        else:
            return x, psi1


class ECNetLL(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, leaky=True,
                 upsampling='bilinear', return_att=False, n1=32, iters=3, return_embedding=False):
        super(ECNetLL, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.filters = filters

        self.iters = iters

        self.return_att = return_att
        self.return_embedding = return_embedding

        if pool_layer != 'conv':
            self.Maxpool1 = pool_layer(kernel_size=2, stride=2)
            self.Maxpool2 = pool_layer(kernel_size=2, stride=2)
            self.Maxpool3 = pool_layer(kernel_size=2, stride=2)
        else:
            self.Maxpool1 = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=2, stride=2, padding=0)
            self.Maxpool2 = nn.Conv2d(self.filters[1], self.filters[1], kernel_size=2, stride=2, padding=0)
            self.Maxpool3 = nn.Conv2d(self.filters[2], self.filters[2], kernel_size=2, stride=2, padding=0)

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.Conv_input = nn.Sequential(nn.Conv2d(9, self.filters[0], kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(self.filters[0]),
                                        relu(*param))

        self.Conv1 = RRB_LSTM(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=False)
        self.Conv2 = RRB_LSTM(self.filters[0], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Conv3 = RRB_LSTM(self.filters[1], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Conv4 = RRB_LSTM(self.filters[2], self.filters[3], norm_layer, leaky=leaky, down_ch=True)

        self.Up4 = Up(self.filters[3], self.filters[2], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Att4 = MaskGAM(F_g=self.filters[2], F_l=self.filters[2], F_int=self.filters[1],
                            norm_layer=norm_layer, leaky=leaky)

        self.Up_conv4 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv4_2 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)

        self.Up3 = Up(self.filters[2], self.filters[1], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Att3 = MaskGAM(F_g=self.filters[1], F_l=self.filters[1], F_int=self.filters[0],
                            norm_layer=norm_layer, leaky=leaky)

        self.Up_conv3 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv3_2 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)

        self.Up2 = Up(self.filters[1], self.filters[0], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Att2 = MaskGAM(F_g=self.filters[0], F_l=self.filters[0], F_int=32,
                            norm_layer=norm_layer, leaky=leaky)

        self.Up_conv2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)
        self.Up_conv2_2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)

        self.Conv = nn.Conv2d(self.filters[0], 3, kernel_size=1, stride=1, padding=0)

    def forward(self, input, lcn, detach_input=False, all_stage_out=True):

        x = input

        h1, c1, h2, c2, h3, c3, h4, c4 = None, None, None, None, None, None, None, None

        self.xs = []
        self.e4s = []
        self.atts = []

        for i in range(self.iters):

            x_in = torch.cat([input, x, lcn], dim=1)

            if detach_input:
                x_in = x_in.detach()
                # x_in.detach_()

            x_in = self.Conv_input(x_in)

            e1, h1, c1 = self.Conv1(x_in, h1, c1)

            e2 = self.Maxpool1(e1)
            e2, h2, c2 = self.Conv2(e2, h2, c2)

            e3 = self.Maxpool2(e2)
            e3, h3, c3 = self.Conv3(e3, h3, c3)

            e4 = self.Maxpool3(e3)
            e4, h4, c4 = self.Conv4(e4, h4, c4)

            d4 = self.Up4(e4)
            x3, psi3 = self.Att4(g=d4, x=e3)
            d4 = self.Up_conv4(d4) + self.Up_conv4_2(x3)

            d3 = self.Up3(d4)
            x2, psi2 = self.Att3(g=d3, x=e2)
            d3 = self.Up_conv3(d3) + self.Up_conv3_2(x2)

            d2 = self.Up2(d3)
            x1, psi1 = self.Att2(g=d2, x=e1)
            d2 = self.Up_conv2(d2) + self.Up_conv2_2(x1)

            out_res = self.Conv(d2)

            x = out_res

            self.xs.append(x)
            self.e4s.append(e4)
            self.atts.append(psi1)

        if all_stage_out:
            x = self.xs
            e4 = self.e4s
            psi1 = self.atts

        if self.return_embedding:
            return x, e4, psi1
        else:
            return x, psi1
