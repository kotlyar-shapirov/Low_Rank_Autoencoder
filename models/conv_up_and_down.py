import torch
from torch import nn

# Squeezing convolution
class DownsampleBlock(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.ELU()):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=4, stride=2, padding=1, dilation=1)
        self.nonlinearity = nonlinearity
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.nonlinearity(x1)
        return x2

# Upscaling inverse convolution
class UpsampleBlock(nn.Module):
    def __init__(self, in_features, out_features, scale_factor=2, nonlinearity=nn.ELU()):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_features, out_features,
                                           kernel_size=4, output_padding=0, padding=1, stride=2)
        self.nonlinearity = nonlinearity
        
    def forward(self, x):
        x1 = self.upsample(x)
        x2 = self.nonlinearity(x1)
        return x2