import torch
from torch import nn
from .sampling_functions import *

# LOW RANK PANTS
class LowRankPants(nn.Module):
    def __init__(self, in_features, out_features, n_bins, grid,
                 dropout=0.2, sampling='gumbell', temperature=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bins = n_bins
        self.dropout = dropout
        # sampling
        self.sampling = sampling
        self.temperature = temperature
        # coeff for logarithms
        self.eps = 1e-9/(n_bins*out_features)
        # coordinates grid
        self.grid = grid
        
        assert sampling in {'vector', 'gumbell'}, 'Select: vector, gumbell'
        
        # mapping to final probabilities (штаны)
        self.layers = nn.ModuleList([nn.Sequential(
                                                   nn.Linear(in_features, n_bins),
                                                  )
                                                   for i in range(out_features)]
                                   )

        # dropout for randromized vector sampling
        self.dropout = nn.Dropout(dropout)
        
    # even faster forward - inner products with the range vec
    def forward(self,x):
        B = x.shape[0]

        # getting additional linear layer
        factors_list = []
        for layer in self.layers:
            factors_list.append(layer(x)) # (B, n_bins)
        # stack  them up
        factors = torch.stack(factors_list, dim=-1)
        factors_logits = factors.view(B, self.out_features, self.n_bins) # size = (B, out_features, n_bins)

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
    def __init__(self, in_features, out_features, n_bins, grid,
                 dropout, nonlinearity,
                 sampling, temperature):
        super().__init__() 
        
        # low rank probabilites
        self.low_rank_pants = LowRankPants(in_features, out_features, n_bins, grid,
                                           dropout, sampling, temperature)
        
        # feedforward to original size
        self.decoder = nn.Sequential(nn.Linear(out_features, in_features),
                                     nonlinearity,
                                    )
    def forward(self, x):
        encoded_out_dim, factors_probability = self.low_rank_pants(x)
        decoded = self.decoder(encoded_out_dim)
        return decoded


