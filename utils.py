import torch

def gaussian_noise(shape):
    return torch.normal(mean=0, std=torch.ones(shape))
