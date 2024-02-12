import torchvision
from torch import nn


# EfficientNet
class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT').features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))

