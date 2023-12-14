import math
import torch
from torch import nn
from inspect import isfunction
from utils import exists, default
from functools import partial
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from torch.fft import rfft2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(
        schedule_type, 
        n_timestep, 
        linear_start=1e-4, 
        linear_end=2e-2, 
        cosine_s=8e-3
    ):
    if schedule_type == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule_type == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule_type == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule_type == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule_type == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)

    elif schedule_type == 'jsd':
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
        
    elif schedule_type == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)

    else:
        raise NotImplementedError(schedule_type)
    return betas


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def complex_mse_loss(  
        input,
        target,
    ):
    difference = input - target
    return ((difference.real**2 + difference.imag**2) / 2).mean()

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        image_size: int,
        channels:int = 3,
        loss_type:str ='l2',
    ):
        super().__init__()

        self.loss_type = loss_type
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

    def set_new_noise_schedule(
            self, 
            device,
            schedule_type, 
            n_timestep, 
            linear_start=1e-4, 
            linear_end=2e-2, 
            cosine_s=8e-3,
        ):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule_type=schedule_type,
            n_timestep=n_timestep,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s
        )

        betas = betas.detach().cpu().numpy() \
            if isinstance(betas, torch.Tensor) else betas
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)

        self.register_buffer('betas', to_torch(betas))
        
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]
        ).repeat(batch_size, 1).to(x.device)
        
        if condition_x is not None:
            predicted_noise = self.denoise_fn(
                torch.cat([condition_x, x], dim=1),
                noise_level
            )
            print('predicted_noise: ', predicted_noise.shape)
            
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=predicted_noise
            )
            print('x_recon: ', x_recon.shape)
            
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level)
            )

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, 
            t=t, 
            clip_denoised=clip_denoised, 
            condition_x=condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        print('model_mean: ', model_mean.shape)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        x = x_in
        shape = x.shape
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x)
            print('img: ', img.shape)

            break
        return img

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (
                batch_size, 
                channels, 
                image_size, 
                image_size
            ), 
            continous
        )

    @torch.no_grad()
    def super_resolution(self, x_in):
        return self.p_sample_loop(x_in)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start, device=DEVICE))

        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        B, C, H, W = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=B
            )
        )

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(B, -1)

        noise = default(noise, lambda: torch.randn_like(x_start, device=DEVICE))
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.to(DEVICE)

        x_noisy = self.q_sample(
            x_start=x_start, 
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
            noise=noise
        )

        predicted_noise = self.denoise_fn(
            torch.cat([x_in['LR'], x_noisy], dim=1), 
            continuous_sqrt_alpha_cumprod
        )

        if self.loss_type == 'l1':
            loss = F.l1_loss(
                noise, 
                x_recon
            )

        elif self.loss_type == 'l2':
            loss = F.mse_loss(
                noise, 
                x_recon
            )

        elif self.loss_type == 'fl2':
            with torch.no_grad():
                fourier_noise = ( 2**(1/2) / H ) * rfft2(noise)

            loss = complex_mse_loss(
                fourier_noise, 
                ( 2**(1/2) / H ) * rfft2(x_recon)
            )

        elif self.loss_type == 'fl2_l2':
            with torch.no_grad():
                fourier_noise = ( 2**(1/2) / H ) * rfft2(noise)

            loss = F.mse_loss(
                noise, 
                x_recon
            ) + complex_mse_loss(
                fourier_noise, 
                ( 2**(1/2) / H ) * rfft2(x_recon)
            )

        elif self.loss_type == 'sl1_l2':
            x_start_predicted = self.predict_start_from_noise(
                x_noisy, t=t, noise=predicted_noise
            )
            model_mean_predicted, posterior_log_variance_predicted = self.q_posterior(
                x_start=x_start_predicted, x_t=x_noisy, t=t
            )
            x_recon_prev = model_mean_predicted +\
                torch.randn_like(x_noisy) * (0.5 * posterior_log_variance_predicted).exp()
        
            model_mean, posterior_log_variance = self.q_posterior(
                x_start=x_in['HR'], x_t=x_noisy, t=t
            )
            x_prev = model_mean +\
                torch.randn_like(x_noisy) * (0.5 * posterior_log_variance).exp()

            loss = F.mse_loss(
                noise, 
                predicted_noise
            ) + self.sqrt_alphas_cumprod_prev[t-1].view(-1, 1, 1, 1)*\
            self.torch.abs(
                torch.pow(x_prev.real, 2) + 
                torch.pow(x_prev.imag, 2) - 
                torch.pow(x_recon_prev.real, 2) + 
                torch.pow(x_recon_prev.imag, 2)
            )


        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)