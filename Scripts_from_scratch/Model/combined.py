from Scripts_from_scratch.Model.backbone import Backbone
from Scripts_from_scratch.Model.decoder import Decoder
from torch.nn import Module
from torchsummary import summary


class Combined(Module):
    def __init__(self):
        super().__init__()

        self.backbone = Backbone()
        self.decoder = Decoder()

    def forward(self, x):
        x, x1, x2, x3, x4 = self.backbone.forward(x)
        x = self.decoder(x, x1, x2, x3, x4)

        return x


# model = Combined()
# summary(model, input_size=(3, 256, 256), batch_size=8, device='cpu')
