import torch
from torch import nn
from .sampling_functions import *


# global device

## Low Rank

# LOW RANK PANTS
class LowRankPants(nn.Module):
    def __init__(self, in_features, bottleneck_features, n_bins, grid,
                 dropout=0.2, sampling='gumbell', temperature=0.5):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_features = bottleneck_features
        self.n_bins = n_bins
        self.dropout = dropout
        # sampling
        self.sampling = sampling
        self.temperature = temperature
        # coeff for logarithms
        self.eps = 1e-9/(n_bins*bottleneck_features)
        # coordinates grid
        self.grid = grid
        
        assert sampling in {'vector', 'gumbell'}, 'Select: vector, gumbell'
        
        self.layers = nn.Linear(in_features, n_bins*bottleneck_features)

        # dropout for randromized vector sampling
        self.dropout = nn.Dropout(dropout)
        
    # even faster forward - inner products with the range vec
    def forward(self,x):
        B = x.shape[0]
        factors = self.layers(x)
        factors_logits = factors.view(B, self.bottleneck_features, self.n_bins) # size = (B, out_features, n_bins)

        # choosing the sampling
        if self.sampling == 'vector':
            factors_probability = nn.Softmax(dim=-1)(factors_logits*self.temperature)
            dropped_factors = self.dropout(factors_probability) + self.eps
            normalized_drop_factors = dropped_factors/(torch.sum(dropped_factors, dim=-1, keepdim=True))
            encoded =  vector_sampling(normalized_drop_factors, self.grid)
            
        elif self.sampling == 'gumbell':
            encoded = gumbell_torch_sampling(factors_logits, self.grid, self.temperature)
        
        return encoded, factors_logits


# LOW RANK PANTS with reconstructing back to original dimension
class LowRankAutoencoder(nn.Module):
    def __init__(self, in_features, bottleneck_features, out_features, n_bins, grid,
                 dropout, nonlinearity,
                 sampling, temperature):
        super().__init__() 
        
        # low rank probabilites
        self.low_rank_pants = LowRankPants(in_features, bottleneck_features, n_bins, grid,
                                           dropout, sampling, temperature)
        
        # feedforward to original size
        self.decoder = nn.Sequential(nn.Linear(bottleneck_features, out_features),
                                     nonlinearity,
                                    )
    def forward(self, x):
        encoded_out_dim, factors_probability = self.low_rank_pants(x)
        decoded = self.decoder(encoded_out_dim)
        return decoded


######################################



### Variational autoencoder (VAE)

# VAE PANTS 
class PantsVAE(nn.Module):
    def __init__(self, in_features, bottleneck_features, nonlinearity):
        super().__init__()
        
        self.mu = nn.Sequential(nn.Linear(in_features, bottleneck_features),
                                     nonlinearity,)
        
        self.sigma = nn.Sequential(nn.Linear(in_features, bottleneck_features),
                                     nonlinearity,)
        
        
    def forward(self,x):
        device = x.device
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))
        rand = torch.randn(mu.shape).to(device)
        encoded = rand*sigma + mu
        
        KL = sigma**2 + mu**2 - torch.log(sigma) - 1/2
        
        return encoded, KL
    
    
# VAE PANTS  with reconstructing back to original dimension
class InternalVAutoencoder(nn.Module):
    def __init__(self, in_features, bottleneck_features, out_features, nonlinearity):
        super().__init__() 

        # Pants probabilites
        self.low_rank_pants = PantsVAE(in_features, bottleneck_features, nonlinearity)
        
        # intermediate_decoder
        self.decoder = nn.Sequential(nn.Linear(bottleneck_features, out_features),
                                     nonlinearity,)
        
        
    def forward(self, x):
        encoded, _ = self.low_rank_pants(x)
        decoded = self.decoder(encoded)

        return decoded

    


######################################


### Autoencoder (AE) - vanilla

# AE PANTS 
class PantsAE(nn.Module):
    def __init__(self, in_features, bottleneck_features, nonlinearity):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_features = bottleneck_features
        self.out = nn.Sequential(nn.Linear(in_features, bottleneck_features),
                                     nonlinearity,)
        
    def forward(self,x):
        encoded = self.out(x)
        return encoded, None
    
    
# AE PANTS  with reconstructing back to original dimension
class InternalAutoencoder(nn.Module):
    def __init__(self, in_features, bottleneck_features, out_features, nonlinearity):
        super().__init__() 
        
        # low rank probabilites
        self.low_rank_pants = PantsAE(in_features, bottleneck_features, nonlinearity)
        
        # intermediate_decoder
        self.decoder = nn.Sequential(nn.Linear(bottleneck_features, out_features),
                                     nonlinearity,)
        

    def forward(self, x):        
        encoded, _ = self.low_rank_pants(x)  # from flatten
        decoded = self.decoder(encoded)  # to flatten dim
        return decoded


######################################