import torch
import torch.nn as nn
import torch.nn.functional as F
from src.resunet import RecNet
from src.resnet import resnet18
from src.block import ACFFNet, Segment


class MFRLNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.recnet = RecNet()
        self.acffnet = ACFFNet()
        self.seg = Segment()
        self.classifier = resnet18(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        x = self.acffnet(self.recnet(x))
        return self.seg(x), self.classifier(x)
