from torch import nn
import torch
from efficientnet_pytorch import EfficientNet

# from config import settings

class Net(nn.Module):
    def __init__(self, net_version, num_classes, settings, freeze: bool = False):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-'+net_version)
        self.backbone._fc = nn.Sequential(
            nn.Linear(settings.model.fcLayer, num_classes),
        )
        if freeze:
            # freeze backbone layers
            for name, param in self.backbone.named_parameters():
                if not name.startswith("_fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
    
class Dino(nn.Module):
    def __init__(self, num_classes, settings, freeze: bool = False):
        super(Dino, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        self.linear = nn.Sequential(
            nn.Linear(settings.model.fcLayer, num_classes),
        )
        if freeze:
            # freeze backbone layers
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.linear(self.backbone(x))
