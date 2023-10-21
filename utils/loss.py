import torch

def entropy_loss(factors_probability):
    factors_probability = torch.nn.Softmax(dim=-1)(factors_probability)
    loss_entropy = torch.sum(torch.log(factors_probability+1e-9)*factors_probability,dim=-1)            
    loss_entropy_mean =  torch.mean(torch.exp(loss_entropy)) # exp of mean entropy loss
    return loss_entropy_mean


def wasser_loss(factors_probability):
    device = factors_probability.device
    B, C, N_bins = factors_probability.shape
    factors_probability = torch.nn.Softmax(dim=-1)(factors_probability)
    factors_cdf = torch.cumsum(factors_probability, dim=-1)
    # diff_abs_cdf = torch.abs(factors_cdf - torch.arange(1,factors_cdf.shape[-1]+1).to(device)/factors_cdf.shape[-1])
    # integral_diff = torch.sum(diff_abs_cdf*1/diff_abs_cdf.shape[-1], dim=-1)
    # wasser_mean_loss = torch.mean(torch.sum(torch.sqrt(integral_diff**2), dim=-1))/(factors_cdf.shape[1])

    diff_abs_cdf = torch.abs(factors_cdf - torch.arange(1,N_bins+1).to(device)/N_bins)
    integral_diff = torch.sum(diff_abs_cdf/N_bins, dim=-1)
    wasser_mean_loss = torch.mean(torch.sum(torch.sqrt(integral_diff**2), dim=-1))/C
    return wasser_mean_loss