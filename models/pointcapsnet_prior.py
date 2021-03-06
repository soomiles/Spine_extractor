from __future__ import print_function

import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import pdb


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class PriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, out_channels)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)

        self.fc2 = torch.nn.Linear(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

        self.fc3 = torch.nn.Linear(out_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.fc4 = torch.nn.Linear(out_channels, out_channels)
        self.bn4 = torch.nn.BatchNorm1d(out_channels)

        self.th = torch.nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.th(self.bn4(self.fc4(x)))
        return x


class CSEModule(nn.Module):
    def __init__(self, in_channels, reduction=128):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x + x * self.cSE(x)


class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_vec_size=8, prior_size=10, prior_vec_size=64, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),
                ('bn3', nn.BatchNorm1d(1024)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])
        self.prior_ens = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('p_encoder', PriorEncoder(prior_size, prior_vec_size))
            ]))
            for _ in range(prim_vec_size)])
        self.cse_modules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('cse', CSEModule(1024 + prior_vec_size))
            ]))
            for _ in range(prim_vec_size)])

    def forward(self, x, prior):
        u = []
        for capsule, encoder, cse in zip(self.capsules, self.prior_ens, self.cse_modules):
            # pdb.set_trace()
            k = torch.cat([capsule(x), encoder(prior).unsqueeze(2)], dim=1)
            k = cse(k)
            u.append(k)
        u = torch.stack(u, dim=2)
        # if ((u ** 2).sum(-2) == 0).sum():
        #     import pdb
        #     pdb.set_trace()
        return self.squash(u.squeeze())

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
                        ((1. + squared_norm) * torch.sqrt(squared_norm))
        if (output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor


class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=16, prior_vec_size=64, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.prior_vec_size= prior_vec_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01 * torch.randn(latent_caps_size, prim_caps_size + prior_vec_size, latent_vec_size, prim_vec_size))

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size + self.prior_vec_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
                        ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size / 2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size / 2), int(self.bottleneck_size / 4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size / 4), 3, 1)
        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 4))

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = self.th(self.conv4(x3))
        return x4


class CapsDecoder(nn.Module):
    def __init__(self, latent_caps_size, latent_vec_size, num_points):
        super(CapsDecoder, self).__init__()
        self.latent_caps_size = latent_caps_size
        self.bottleneck_size = latent_vec_size
        self.num_points = num_points
        self.nb_primitives = int(num_points / latent_caps_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size + 2) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.latent_caps_size))
            rand_grid.data.uniform_(0, 1)
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()


class PointCapsNet(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size,
                 prior_size, prior_vec_size,
                 latent_caps_size, latent_vec_size, num_points):
        super(PointCapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, prior_size, prior_vec_size, num_points)
        self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prior_vec_size, prim_caps_size,
                                                 prim_vec_size, latent_vec_size)
        self.caps_decoder = CapsDecoder(latent_caps_size, latent_vec_size, num_points)

    def forward(self, data, prior):
        x1 = self.conv_layer(data)
        x2 = self.primary_point_caps_layer(x1, prior)
        latent_capsules = self.latent_caps_layer(x2)
        reconstructions = self.caps_decoder(latent_capsules)
        return latent_capsules, reconstructions


# This is a single network which can decode the point cloud from pre-saved latent capsules
class PointCapsNetDecoder(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points):
        super(PointCapsNetDecoder, self).__init__()
        self.caps_decoder = CapsDecoder(digit_caps_size, digit_vec_size, num_points)

    def forward(self, latent_capsules):
        reconstructions = self.caps_decoder(latent_capsules)
        return reconstructions
