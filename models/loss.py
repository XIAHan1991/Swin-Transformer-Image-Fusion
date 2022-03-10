import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from torchvision import models
import os, cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_out,self).__init__()
        vgg = models.vgg19(pretrained=False).to(device)
        vgg.load_state_dict(torch.load("./pretrain/vgg19-dcbb9e9d.pth"))
        vgg.eval()
        vgg_pretrained_features = vgg.features
        self.requires_grad = requires_grad
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(x)
        h_relu3 = self.slice3(x)
        h_relu4 = self.slice4(x)
        h_relu5 = self.slice5(x)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Perceptual_loss134(nn.Module):
    def __init__(self):
        super(Perceptual_loss134, self).__init__()
        self.vgg = Vgg19_out().to(device)
        self.criterion = nn.MSELoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = self.weights[0] * self.criterion(x_vgg[0], y_vgg[0].detach()) + self.weights[1] * self.criterion(x_vgg[1], y_vgg[1].detach()) + self.weights[2] * self.criterion(x_vgg[2], y_vgg[2].detach())
        return loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19_out().to(device)
        self.criterion = nn.MSELoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0.0
        for iterm, (x_fea, y_fea) in enumerate(zip(x_vgg, y_vgg)):
            print(iter + 1, self.criterion(x_fea, y_fea.detach()), x_fea.size())
            loss += self.criterion(x_fea, y_fea.detach())
        return loss