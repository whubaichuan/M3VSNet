import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class vggNet(nn.Module):
    def __init__(self, pretrained=True):
        super(vggNet, self).__init__()
        self.net = models.vgg16(pretrained=True).features.eval()

    def forward(self, x):

        out = []
        for i in range(len(self.net)):
            #x = self.net[i](x)
            x = self.net[i](x)
            #if i in [3, 8, 15, 22, 29]:
            #if i in [15]: #提取1，1/2，1/4的特征图
            if i in [8,15,22]: #提取1,1/2,1/4,1/8,1/16
                # print(self.net[i])
                out.append(x)
        return out

