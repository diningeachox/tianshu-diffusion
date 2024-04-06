import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet import UNet
import math
from tqdm import tqdm

'''
Calculates the beta schedule for the forward diffusion
'''
def beta_schedule(num_timesteps, type="linear"):
    if type == "linear":
        beta_t = torch.linspace(1e-4, 2e-2, num_timesteps + 1)
    elif type == "cosine":
        s = 0.008 # Offset to prevent beta_t from being too small near t=0
        linspace = torch.linspace(0, 1, num_timesteps + 1)
        f_t = torch.cos((linspace + s) / (1 + s) * math.pi / 2) ** 2
        bar_alpha_t = f_t / f_t[0]
        beta_t = torch.zeros_like(bar_alpha_t)
        beta_t[1:] = (1 - (bar_alpha_t[1:] / bar_alpha_t[:-1])).clamp(min=0, max=0.999)
    return beta_t

class DiffusionModel(nn.Module):
    def __init__(self, betas, out_channels=3, channel_scales=(1, 2, 2, 2), d_model=128, device='cpu'):
        super().__init__()

        #Coefficients
        self.timesteps = betas.shape[0]
        alphas = 1. - betas
        alpha_bar = torch.cumprod(alphas, axis=0).to(device)
        alpha_bar_prev = torch.cat((torch.tensor([1.]).to(device), alpha_bar[:-1]))

        self.device = device

        #Coefficients for forward process q(x_t | x_{t-1})
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_alpha_bar_prev = alpha_bar_prev
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1. / alpha_bar)
        self.sqrt_recip_minus_one_alpha_bar = torch.sqrt(1. / alpha_bar - 1.)
        #Coefficients for posterior distribution q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alpha_bar_prev) / (1. - alpha_bar)
        self.posterior_mean_coef_0 = betas * self.sqrt_alpha_bar_prev / (1. - alpha_bar)
        self.posterior_mean_coef_t = (1. - alpha_bar_prev) * torch.sqrt(alphas) / (1. - alpha_bar)
        self.posterior_log_variance_clipped = torch.log(torch.maximum(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance).to(device)))


        # Denoising model (UNet architecture)
        self.denoiser = UNet(n_channels=128, out_channels=out_channels, n_res_blocks=1, attention_res=(160,), channel_scales=channel_scales, d_model=d_model, timesteps=[]).to(device)


    def get_batch_coeffs(self, a, t, x_shape):
        '''
        Get the coefficients at a list of timesteps specified in a batch
        Args:
            t - List of timesteps
        '''
        coeffs = torch.gather(input=a, dim=0, index=t).to(self.device)
        dims = tuple([t.shape[0]] + ((len(x_shape) - 1) * [1]))
        coeffs = torch.reshape(coeffs, dims) # Reshape for broadcasting
        return coeffs

    '''
    Forward sample q(x_t | x_{t-1})
    '''
    def forward_sample(self, x_prev, t, noise=None):
        #Standard normal
        if noise is None:
            noise = torch.normal(mean=0, std=torch.ones_like(x_prev)).to(self.device)

        assert noise.shape == x_prev.shape
        alpha_bars = self.get_batch_coeffs(self.sqrt_alpha_bar, t, x_prev.shape)
        sqrt_one_minus_alpha_bars = self.get_batch_coeffs(self.sqrt_one_minus_alpha_bar, t, x_prev.shape)
        x = alpha_bars * x_prev + sqrt_one_minus_alpha_bars * noise
        return x

    '''
    Forward Posterior q(x_{t-1} | x_t, x_0)
    '''
    def forward_posterior(self, x_start, x_t, t):
        posterior_mean_coef_0 = self.get_batch_coeffs(self.posterior_mean_coef_0, t, x_t.shape)
        posterior_mean_coef_t = self.get_batch_coeffs(self.posterior_mean_coef_t, t, x_t.shape)

        mean = posterior_mean_coef_0 * x_start + posterior_mean_coef_t * x_t
        variance = self.get_batch_coeffs(self.posterior_variance, t, x_t.shape)
        log_variance = self.get_batch_coeffs(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    '''
    Backward process p(x_{t-1} | x_t)
    The last step p(x_0 | x_1) is an independent discrete decoder
    '''
    def backward_sample(self, x, t, temb_model):

        #Predict mean and variance using the UNet model
        pred_noise = self.denoiser(x, t, temb_model)
        sqrt_recip_alpha_bar = self.get_batch_coeffs(self.sqrt_recip_alpha_bar, t, x.shape)
        sqrt_recip_minus_one_alpha_bar = self.get_batch_coeffs(self.sqrt_recip_minus_one_alpha_bar, t, x.shape)
        x_tilde = sqrt_recip_alpha_bar * x - sqrt_recip_minus_one_alpha_bar * pred_noise #Predicted start from the noise
        #x_tilde = torch.clamp(x_tilde, -1., 1.) #Clip values to range
        mean, variance, log_variance = self.forward_posterior(x_tilde, x, t)

        noise = torch.normal(mean=0, std=torch.ones_like(x)).to(self.device) # The random variable z in the paper
        nonzero_mask = ~torch.eq(t, 0).to(self.device)
        log_var = torch.exp(0.5 * log_variance).to(self.device)
        return mean + nonzero_mask * log_var * noise

    def loss(self, x_prev, t, temb_model, noise=None):
        if noise is None:
            noise = torch.normal(mean=0, std=torch.ones_like(x_prev)).to(self.device)

        x_noisy = self.forward_sample(x_prev, t, noise)
        x_recon = self.denoiser(x_noisy, t, temb_model)
        assert x_noisy.shape == x_prev.shape
        assert x_noisy.shape == x_recon.shape

        losses = nn.MSELoss()(noise, x_recon)

        return losses

    '''
    Iteratively sample backward from x_{t_max} until we reconstruct an image fully, i.e. to x_0
    '''
    def generate(self, shape, noise_fn, temb_model):
        i_0 = self.timesteps - 1
        img_0 = noise_fn(shape=shape).to(self.device)

        print("Generating sample image...")
        for t in tqdm(range(i_0 - 1, -1, -1)):
            times = (t * torch.ones(shape[0])).type(torch.LongTensor).to(self.device)
            img_0 = self.backward_sample(img_0, times, temb_model)
        assert img_0.shape == shape
        return img_0

    '''
    Interpolate between images
    '''
    def interpolate(self, shape, x1, x2, t, alpha, noise_fn, temb_model):
        bs = shape[0]
        t_batched = (t * torch.ones(shape[0])).type(torch.LongTensor).to(self.device)
        xt1 = self.forward_sample(x1, t_batched)
        xt2 = self.forward_sample(x2, t_batched)

        # Linearly interpolate images
        xt_interp = (1 - alpha) * xt1 + alpha * xt2

        # Constant variance interpolation
        #xt_interp = torch.sqrt(1 - alpha * alpha) * xt1 + alpha * xt2

        #Backward process starting from timestep=t
        for s in tqdm(range(t, -1, -1)):
            timesteps = (s * torch.ones(shape[0])).type(torch.LongTensor).to(self.device)
            xt_interp = self.backward_sample(xt_interp, timesteps, temb_model)

        assert xt_interp.shape == shape
        return xt_interp
