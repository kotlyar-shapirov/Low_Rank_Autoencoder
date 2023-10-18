import torch





def wasser_loss(factors_probability):
    device = factors_probability.device
    factors_probability = torch.nn.Softmax(dim=-1)(factors_probability)
    factors_cdf = torch.cumsum(factors_probability, dim=-1)
    diff_abs_cdf = torch.abs(factors_cdf - torch.arange(factors_cdf.shape[-1]).to(device)/factors_cdf.shape[-1])
    integral_diff = torch.sum(diff_abs_cdf*1/diff_abs_cdf.shape[-1], dim=-1)
    wasser_mean_loss = torch.mean(torch.sum(torch.sqrt(integral_diff**2), dim=-1))/(factors_cdf.shape[1])
    return wasser_mean_loss