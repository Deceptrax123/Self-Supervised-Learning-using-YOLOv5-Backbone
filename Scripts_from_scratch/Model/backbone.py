import torch
from torch import nn
from Scripts_from_scratch.Model.model_segments.modules import C3, Conv, SPPF
from torchsummary import summary


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv(c1=3, c2=32)
        self.conv2 = Conv(c1=32, c2=64, s=2)
        self.c1 = C3(c1=64, c2=64)
        self.conv3 = Conv(c1=64, c2=128, s=2)
        self.c2 = C3(c1=128, c2=128)
        self.conv4 = Conv(c1=128, c2=256, s=2)
        self.c3 = C3(c1=256, c2=256)
        self.conv5 = Conv(c1=256, c2=512)
        self.c4 = C3(c1=512, c2=512)
        self.conv6 = Conv(c1=512, c2=1024, s=2)
        self.c5 = C3(c1=1024, c2=1024)

        self.sppf = SPPF(c1=1024, c2=5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c1(x)
        x = self.conv3(x)
        x = self.c2(x)
        x = self.conv4(x)
        x = self.c3(x)
        x = self.conv5(x)
        x = self.c4(x)
        x = self.conv6(x)
        x = self.c5(x)

        x = self.sppf(x)

        return x


model = Backbone()

summary(model=model, input_size=(3, 256, 256), batch_size=8, device='cpu')
