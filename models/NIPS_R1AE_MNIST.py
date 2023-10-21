import torch
from torch import nn

from .NIPS_pants import LowRankAutoencoder, InternalVAutoencoder, InternalAutoencoder, InternalIRMAutoencoder
from .conv_up_and_down import UpsampleBlock, DownsampleBlock




## LOW RANK AE
class ConvLRAE(nn.Module):
    def __init__(self, in_features, bottleneck_features, out_features, n_bins, grid,
                 dropout, nonlinearity,
                 sampling='gumbell', temperature=0.5, in_channels=1, start_dropout=0,
                ):
        super().__init__()
        
        self.in_features = in_features   # input features after convolutions and flattening
        self.bottleneck_features = bottleneck_features # encoded dimension
        self.out_features = out_features # expanded before upscaling

        self.n_bins = n_bins
        # network params
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        # sampling params
        self.sampling = sampling
        self.temperature = temperature

        
        self.down = nn.Sequential(nn.Dropout(start_dropout),
                                  DownsampleBlock(in_features=in_channels, out_features=32, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=32, out_features=64, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=64, out_features=128, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=128, out_features=256, nonlinearity=nonlinearity),
                                  
                                  )
        
        self.low_rank = LowRankAutoencoder(in_features, bottleneck_features, out_features, n_bins, grid,
                                           dropout, nonlinearity,
                                           sampling, temperature)
        
        self.up = nn.Sequential(
                                # UpsampleBlock(in_features=256, out_features=128, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=128, out_features=64, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=64, out_features=32, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=32, out_features=in_channels,
                                                nonlinearity=nn.Sigmoid(), kernel_size=1, stride=1, padding=0),
                                )
        
    def forward(self, x):
        # downsample
        x_down = self.down(x)
        B, C, H, W = x_down.shape
        x_flat = x_down.view(B,C*H*W)
        
        # passing theough low rank
        decoded = self.low_rank(x_flat)

        # upsample
        x_2d = decoded.view(B, C//2, H*4, W*4)
        x_out = self.up(x_2d)

        return x_out
    
    
    
## VARIATIONAL AUTOENCODER (VAE)
class ConvVAE(nn.Module):
    def __init__(self, in_features, bottleneck_features, out_features, nonlinearity, in_channels=1, start_dropout=0 ):
        super().__init__()
        
        self.in_features = in_features   # input features after convolutions and flattening
        self.out_features = out_features # encoded dimension

        # network params
        self.nonlinearity = nonlinearity
        
        
        self.down = nn.Sequential(nn.Dropout(start_dropout),
                                  DownsampleBlock(in_features=in_channels, out_features=32, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=32, out_features=64, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=64, out_features=128, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=128, out_features=256, nonlinearity=nonlinearity),
                                  )

        
        self.low_rank = InternalVAutoencoder(in_features, bottleneck_features, out_features, nonlinearity)
        
        self.up = nn.Sequential(
                                # UpsampleBlock(in_features=256, out_features=128, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=128, out_features=64, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=64, out_features=32, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=32, out_features=in_channels,
                                                nonlinearity=nn.Sigmoid(), kernel_size=1, stride=1, padding=0),
                                )
        
    def forward(self, x):
        # downsample
        x_down = self.down(x)
        B, C, H, W = x_down.shape
        x_flat = x_down.view(B,C*H*W)
        
        # passing theough low rank
        decoded = self.low_rank(x_flat)

        # upsample
        x_2d = decoded.view(B, C//2, H*4, W*4)
        x_out = self.up(x_2d)

        return x_out
    
  
## VANILA AUTOENCODER (AE)
class ConvAE(nn.Module):
    def __init__(self, in_features, bottleneck_features, out_features, nonlinearity, in_channels=1, start_dropout=0):
        super().__init__()
        
        self.in_features = in_features   # input features after convolutions and flattening
        self.out_features = out_features # encoded dimension

        # network params
        self.nonlinearity = nonlinearity
        
        
        self.down = nn.Sequential(nn.Dropout(start_dropout),
                                  DownsampleBlock(in_features=in_channels, out_features=32, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=32, out_features=64, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=64, out_features=128, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=128, out_features=256, nonlinearity=nonlinearity),
                                  )

    
        self.low_rank = InternalAutoencoder(in_features, bottleneck_features, out_features, nonlinearity)
        
        self.up = nn.Sequential(
                                # UpsampleBlock(in_features=256, out_features=128, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=128, out_features=64, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=64, out_features=32, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=32, out_features=in_channels,
                                                nonlinearity=nn.Tanh(), kernel_size=1, stride=1, padding=0),
                                )
        
    def forward(self, x):
        # downsample
        x_down = self.down(x)
        B, C, H, W = x_down.shape
        x_flat = x_down.view(B,C*H*W)
        
        # passing theough low rank
        decoded = self.low_rank(x_flat)

        # upsample
        x_2d = decoded.view(B, C//2, H*4, W*4)
        x_out = self.up(x_2d)

        return x_out
    
# IMPLICIT RANK-MINIMIZATION AUTOENCODER (IRMAE)
class ConvIRMAE(nn.Module):
    def __init__(self, in_features, bottleneck_features, out_features, nonlinearity, in_channels=1, start_dropout=0, middle_matrixes=0):
        super().__init__()
        
        self.in_features = in_features   # input features after convolutions and flattening
        self.out_features = out_features # encoded dimension

        # network params
        self.nonlinearity = nonlinearity
        
        self.down = nn.Sequential(nn.Dropout(start_dropout),
                                  DownsampleBlock(in_features=in_channels, out_features=32, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=32, out_features=64, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=64, out_features=128, nonlinearity=nonlinearity),
                                  DownsampleBlock(in_features=128, out_features=256, nonlinearity=nonlinearity),
                                  )

    
        self.low_rank = InternalIRMAutoencoder(in_features, bottleneck_features, out_features, nonlinearity, middle_matrixes=middle_matrixes)
        
        self.up = nn.Sequential(
                                # UpsampleBlock(in_features=256, out_features=128, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=128, out_features=64, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=64, out_features=32, nonlinearity=nonlinearity),
                                UpsampleBlock(in_features=32, out_features=in_channels,
                                                nonlinearity=nn.Tanh(), kernel_size=1, stride=1, padding=0),
                                )
        
    def forward(self, x):
        # downsample
        x_down = self.down(x)
        B, C, H, W = x_down.shape
        x_flat = x_down.view(B,C*H*W)
        
        # passing theough low rank
        decoded = self.low_rank(x_flat)

        # upsample
        x_2d = decoded.view(B, C//2, H*4, W*4)
        x_out = self.up(x_2d)

        return x_out