import torch
from torch import nn
from Scripts_COCO.Model.model_segments.modules import C3, Conv, SPPF
from torchsummary import summary


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv(c1=3, c2=32, k=6, s=2, p=2)
        self.conv2 = Conv(c1=32, c2=64, k=3, s=2)
        self.c1 = C3(c1=64, c2=64, n=1)
        self.conv3 = Conv(c1=64, c2=128, k=3, s=2)
        self.c2 = C3(c1=128, c2=128, n=2)
        self.conv4 = Conv(c1=128, c2=256, k=3, s=2)
        self.c3 = C3(c1=256, c2=256, n=3)
        self.conv5 = Conv(c1=256, c2=512, k=3, s=2)
        self.c4 = C3(c1=512, c2=512, n=1)

        self.sppf = SPPF(c1=512, c2=512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.c1(x)
        x = self.conv3(x1)
        x2 = self.c2(x)
        x = self.conv4(x2)
        x3 = self.c3(x)
        x = self.conv5(x3)
        x4 = self.c4(x)

        x = self.sppf(x4)

        return x, x1, x2, x3, x4


# model = Backbone()

# summary(model=model, input_size=(3, 256, 256), batch_size=8, device='cpu')
