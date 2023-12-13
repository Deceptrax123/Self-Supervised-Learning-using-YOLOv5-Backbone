import torch
from torch import nn

from torchsummary import summary
from Scripts_from_scratch.Model.model_segments.modules import C3, Conv, SPPF, UpConv
from torch.nn import Upsample, Conv2d

# Scripts for creating symmetrical decoder to backbone of yolov5


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # reverse the spatial pyramid pooling effect
        self.cspf = Conv2d(in_channels=5, out_channels=1024,
                           kernel_size=3, stride=1, padding=1)

        # Build the decoder
        self.c1 = C3(1024, 1024)
        self.upconv1 = UpConv(1024, 512, s=2)
        self.c2 = C3(512, 512)
        self.upconv2 = UpConv(512, 256, s=2)
        self.c3 = C3(256, 256)
        self.upconv3 = UpConv(256, 128, s=2)
        self.c4 = C3(128, 128)
        self.upconv4 = UpConv(128, 64, s=2)
        self.c5 = C3(64, 64)
        self.upconv5 = UpConv(64, 32, s=2)
        self.upconv6 = UpConv(32, 3, s=1, op=0)

    def forward(self, x, x1, x2, x3, x4, x5):
        x = self.cspf(x)
        x = self.c1(x)
        # Bring in skip connections from the yolov5 backbone
        x = torch.add(x, x5)
        x = self.upconv1(x)
        x = self.c2(x)
        x = torch.add(x, x4)
        x = self.upconv2(x)
        x = self.c3(x)
        x = torch.add(x, x3)
        x = self.upconv3(x)
        x = self.c4(x)
        x = torch.add(x, x2)
        x = self.upconv4(x)
        x = self.c5(x)
        x = torch.add(x, x1)
        x = self.upconv5(x)
        x = self.upconv6(x)

        return x


# model = Decoder()
# summary(model=model, input_size=(5, 8, 8), batch_size=8, device='cpu')
