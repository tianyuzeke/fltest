from collections import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class CNNHyper(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=1, vdim=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        self.vdim = vdim

        layers = [nn.Linear(embedding_dim, hidden_dim),]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

        # self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        # self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        # self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        # self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * (2 * self.n_kernels * 5 * 5 + vdim))
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            # "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            # "conv1.bias": self.c1_bias(features).view(-1),
            # "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            # "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * self.n_kernels * 5 * 5 + self.vdim),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        return weights, emd.detach().clone()

class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10, vdim=0):
        super(CNNTarget, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5 + vdim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x, emd):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # print(emd.shape)
        # print(x.shape)
        emd = emd.repeat((x.shape[0],1))
        x = x.view(x.shape[0], -1)
        # print(emd.shape)
        # print(x.shape)
        x = F.relu(self.fc1(torch.cat((x, emd), 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LocalLayer(nn.Module):

    def __init__(self, in_channels=3, n_kernels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x