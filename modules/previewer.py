from torch import nn


# Fast Decoder for Stage C latents. E.g. 16 x 24 x 24 -> 3 x 192 x 192
class Previewer(nn.Module):
    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1),  # 16 channels to 512 channels
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            nn.ConvTranspose2d(c_hidden, c_hidden // 2, kernel_size=2, stride=2),  # 16 -> 32
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),

            nn.Conv2d(c_hidden // 2, c_hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),

            nn.ConvTranspose2d(c_hidden // 2, c_hidden // 4, kernel_size=2, stride=2),  # 32 -> 64
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.ConvTranspose2d(c_hidden // 4, c_hidden // 4, kernel_size=2, stride=2),  # 64 -> 128
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_out, kernel_size=1),
        )

    def forward(self, x):
        return self.blocks(x)
