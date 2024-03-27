import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet import UNet

class DiffusionModel(nn.Module):
    def __init__(self, betas):
        super().__init__()

        #Coefficients
        self.timesteps = betas.shape[0]
        alphas = 1. - betas
        alpha_bar = np.cumprod(alphas, axis=0)
        alpha_bar_prev = np.append(1., alpha_bar[:-1])


        #Coefficients for forward process q(x_t | x_{t-1})
        self.sqrt_alpha_bar = np.sqrt(alpha_bar)
        self.sqrt_alpha_bar_prev = np.append(1., alpha_bar[:-1])
        self.sqrt_one_minus_alpha_bar = np.sqrt(1. - alpha_bar)
        self.sqrt_recip_alpha_bar = np.sqrt(1. / alpha_bar)
        self.sqrt_recip_minus_one_alpha_bar = np.sqrt(1. / alpha_bar - 1.)
        #Coefficients for posterior distribution q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.sqrt_alpha_bar_prev) / (1. - alpha_bar)
        self.posterior_mean_coef_0 = betas * self.sqrt_alpha_bar_prev / (1. - alpha_bar)
        self.posterior_mean_coef_t = (1. - alpha_bar_prev) * np.sqrt(alphas) / (1. - alphas_bar)
        self.posterior_log_variance_clipped = np.log(np.maximum(self.posterior_variance, 1e-20))

        # Denoising model (UNet architecture)
        self.denoiser = UNet(n_channels=128, out_channels=3, n_res_blocks=2, attention_res=(16,), channel_scales=(1, 2, 2, 2), d_model=32)

    def get_batch_coeffs(self, a, t, x_shape):
        '''
        Get the coefficients at a list of timesteps specified in a batch
        Args:
            t - List of timesteps
        '''
        coeffs = torch.gather(input=a, dim=0, index=t)
        dims = tuple([t.shape[0]] + ((len(x_shape) - 1) * [1]))
        torch.reshape(coeffs, dims) # Reshape for broadcasting
        return coeffs

    '''
    Forward sample q(x_t | x_{t-1})
    '''
    def forward_sample(self, x_prev, noise=None):
        #Standard normal
        if noise is None:
            noise = torch.normal(mean=0, std=torch.ones_like(x_prev))

        assert noise.shape = x_prev.shape
        x = self.sqrt_alpha_bar * x_prev + self.sqrt_one_minus_alpha_bar * noise
        return x

    '''
    Forward Posterior q(x_{t-1} | x_t, x_0)
    '''
    def forward_posterior(self, x_start, x_t, t, noise):
        mean = self.posterior_mean_coef_0 * x_start + self.posterior_mean_coef_t * x_t
        variance = self.posterior_variance
        log_variance = self.posterior_log_variance_clipped
        return mean, variance, log_variance

    '''
    Backward process p(x_{t-1} | x_t)
    '''
    def backward_sample(self, x, t):

        #Predict mean and variance using the UNet model
        pred_noise = self.denoiser(x, t)
        x_tilde = self.sqrt_recip_alpha_bar * x - self.sqrt_recip_minus_one_alpha_bar * pred_noise
        mean, variance, log_variance = self.forward_posterior(x_tilde, x, t)
        noise = torch.normal(mean=0, std=torch.ones_like(x)) # The random variable z in the paper
        nonzero_mask = 1. - torch.eq(t, 0)

        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    def loss(self, x_prev, t, noise=None):
        if noise is None:
            noise = torch.normal(mean=0, std=torch.ones_like(x_prev))

        x_noisy = self.forward_sample(x_prev, t, noise)
        x_recon = self.denoiser(x_noisy, t)
        assert x_noisy.shape == x_prev.shape
        assert x_noisy.shape == x_recon.shape

        losses = nn.MSELoss()(noise, x_recon)

        return losses

    '''
    Iteratively sample backward from x_{t_max} until we reconstruct an image fully, i.e. to x_0
    '''
    def generate(self, shape, noise_fn):
        i_0 = self.num_timesteps - 1
        img_0 = noise_fn(shape=shape)

        for t in range(i_0, -1, -1):
            img_0 = backward_sample(img_0, t * torch.ones(shape[0]))

        assert img_0.shape == shape
        return img_0

    '''
    Interpolate between images
    '''
    def interpolate(self, shape, x1, x2, t, alpha, noise_fn):
        bs = shape[0]
        t_batched = t * torch.ones(shape[0])
        xt1 = self.forward_sample(x1, t_batched)
        xt2 = self.forward_sample(x2, t_batched)

        # Linearly interpolate images
        xt_interp = (1 - alpha) * xt1 + alpha * xt2

        # Constant variance interpolation
        #xt_interp = torch.sqrt(1 - alpha * alpha) * xt1 + alpha * xt2
        for t in range(i_0, -1, -1):
            xt_interp = backward_sample(xt_interp, t * torch.ones(shape[0]))

        assert xt_interp.shape == shape
        return xt_interp
