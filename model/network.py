"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class skip(nn.Module):
    """ Skip block """

    def __init__(self, input_c, d1, d2):
        super(skip, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=input_c, out_channels=d1, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(d1)

        self.conv_2 = nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=3, bias=False, padding=1)
        self.bn_2 = nn.BatchNorm2d(d1)

        self.conv_3 = nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(d2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.bn_3(out)

        out = x + out
        out = self.relu(out)

        return out


class projection(nn.Module):
    """ Projection block """

    def __init__(self, input_c, d1, d2, stride):
        super(projection, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=input_c, out_channels=d1, kernel_size=1, stride=stride)
        self.bn_1 = nn.BatchNorm2d(d1)

        self.conv_2 = nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(d1)

        self.conv_3 = nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=1)
        self.bn_3 = nn.BatchNorm2d(d2)

        self.conv_proj = nn.Conv2d(in_channels=input_c, out_channels=d2, kernel_size=1, stride=stride)
        self.bn_proj = nn.BatchNorm2d(d2)

        self.relu = nn.ReLU()

    def forward(self, x):
        out_proj = self.conv_proj(x)
        out_proj = self.bn_proj(out_proj)

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.bn_3(out)

        out = out + out_proj
        out = self.relu(out)

        return out


class fast_up_projection(nn.Module):
    """ Up_projection block for feature map up-sampling """

    def __init__(self, input_c, output_c):
        super(fast_up_projection, self).__init__()
        self.conv_3_3 = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=(3, 3))
        self.conv_2_3 = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=(2, 3))
        self.conv_3_2 = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=(3, 2))
        self.conv_2_2 = nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=(2, 2))

        self.conv = nn.Conv2d(in_channels=output_c, out_channels=output_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # four different filters
        out_3_3 = self.conv_3_3(F.pad(x, (1, 1, 1, 1)))
        out_2_3 = self.conv_2_3(F.pad(x, (1, 1, 1, 0)))
        out_3_2 = self.conv_3_2(F.pad(x, (1, 0, 1, 1)))
        out_2_2 = self.conv_2_2(F.pad(x, (1, 0, 1, 0)))

        # interleaving feature maps
        out_3_3_3_2 = torch.stack((out_3_3, out_3_2), dim=3).transpose(4, 3).flatten(start_dim=3)
        out_2_3_2_2 = torch.stack((out_2_3, out_2_2), dim=3).transpose(4, 3).flatten(start_dim=3)
        out_interleaved = torch.stack((out_3_3_3_2, out_2_3_2_2), dim=2).transpose(2, 3).flatten(start_dim=2, end_dim=3)

        # fast up projection
        out_interleaved_conv = self.conv(self.relu(out_interleaved))
        out = out_interleaved + out_interleaved_conv
        out = self.relu(out)

        return out


class BerHuLoss(nn.Module):
    """ the reverse Huber (berHu) Loss """
    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, pred, label):
        pred = torch.flatten(pred, start_dim=1, end_dim=-1)
        label = torch.flatten(label, start_dim=1, end_dim=-1)
        t = 0.2 * torch.max(torch.abs(pred - label))
        l1 = torch.mean(torch.mean(torch.abs(pred - label), 1), 0)
        l2 = torch.mean(torch.mean(((pred - label) ** 2 + t ** 2) / t / 2, 1), 0)

        return l2 if l1 > t else l1

