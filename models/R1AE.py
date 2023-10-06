import torch
from torch import nn

from .pants import LowRankAutoencoder
from .conv_up_and_down import UpsampleBlock, DownsampleBlock

class ConvLRAE(nn.Module):
    def __init__(self, in_features,out_features, n_bins, grid,
                 dropout, nonlinearity,
                 sampling='gumbell', temperature=0.5,
                ):
        super().__init__()
        
        self.in_features = in_features   # input features after convolutions and flattening
        self.out_features = out_features # encoded dimension
        self.n_bins = n_bins
        # network params
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        # sampling params
        self.sampling = sampling
        self.temperature = temperature

        
        self.down = nn.Sequential(nn.Dropout(0.2),
                                  DownsampleBlock(in_features=1, out_features=32, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=32, out_features=64, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=64, out_features=128, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=128, out_features=256, nonlinearity=nonlinearity),
                                  
                                  )
        
        self.low_rank = LowRankAutoencoder(in_features, out_features, n_bins, grid,
                                           dropout, nonlinearity,
                                           sampling, temperature)
        
        self.up = nn.Sequential(
                                UpsampleBlock(in_features=256, out_features=128, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=128, out_features=64, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=64, out_features=32, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=32, out_features=1,  nonlinearity=nn.Tanh()),
                                )
        
    def forward(self, x):
        # downsample
        x_down = self.down(x)
        B, C, H, W = x_down.shape
        x_flat = x_down.view(B,C*H*W)
        
        # passing theough low rank
        decoded = self.low_rank(x_flat)

        # upsample
        x_2d = decoded.view(B, C, H, W)
        x_out = self.up(x_2d)

        return x_out