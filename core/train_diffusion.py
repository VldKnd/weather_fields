import os
import argparse
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Tuple, Optional, Dict
from datetime import datetime

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader


from data_utils import WeatherFieldsDataset
from schedule import linear_beta_schedule
from unet import Unet

from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)

timesteps = 300

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def spectral_noise_generator(shape: Tuple) -> torch.Tensor:
    noise = torch.randn(shape)
    return noise, torch.fft.rfft2(noise)

def complex_mse_loss(  
        input,
        target,
    ):
    difference = input - target
    return ((difference.real**2 + difference.imag**2) / 2).mean()

def p_spectral_losses(
        denoise_model: Unet,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        self_condition: Optional[torch.Tensor] = None
    ):
    
    batch_size, C, H, W = x_start.shape
    
    if noise is None:
        noise = torch.randn_like(x_start)

    with torch.no_grad():
        domain_fourier_noise = torch.randn_like(x_start)
        fourier_noise = ((2**1/2) / H) * torch.fft.rfft2(domain_fourier_noise)
        
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    x_noisy_transformed = q_sample(x_start=x_start, t=t, noise=domain_fourier_noise)

    if denoise_model.self_condition:
        if self_condition is None:
            raise RuntimeError("The self-conditioning is not provided. ")
        
        predicted_noise = denoise_model.forward(
            x=x_noisy, 
            time=t,
            x_self_cond=self_condition
        )
        
        predicted_domain_fourier_noise = denoise_model.forward(
            x=x_noisy_transformed, 
            time=t,
            x_self_cond=self_condition
        )
        
    else:
        predicted_noise = denoise_model.forward(
            x=x_noisy, 
            time=t
        )
        
        predicted_domain_fourier_noise = denoise_model.forward(
            x=x_noisy_transformed, 
            time=t,
            x_self_cond=self_condition
        )

    loss = (
        F.mse_loss(noise, predicted_noise) +
        complex_mse_loss(
            fourier_noise,
            ((2**1/2) / H) * torch.fft.rfft2(predicted_domain_fourier_noise)
        )
    )

    return loss

def parse_arguments() -> Dict:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-b',
        '--batch_size', 
        dest='batch_size',
        type=int,
        default=256,
    )

    parser.add_argument(
        '-e',
        '--epochs', 
        dest='epochs',
        type=int,
        default=10,
    )

    parser.add_argument(
        '-n',
        '--name', 
        dest='name',
        type=str,
        default="10",
    )

    parser.add_argument(
        '-s',
        '--seed', 
        dest='seed',
        type=int,
        default=0,
    )

    args = parser.parse_args()
    return vars(args)

def train_diffusion(batch_size, epochs, name, seed):
    basicConfig(level=INFO)
    torch.manual_seed(seed)
    
    pwd_path = os.path.abspath("..")
    save_folder_path = os.path.join(
        pwd_path,
        "experiment_informations"
    )
    
    if not os.path.exists(save_folder_path):
        raise OSError(f"Folder for saving information {save_folder_path} does not exits. ")

    dataset = WeatherFieldsDataset(
        path_to_folder=os.path.join(
            "../data",
            "wrf_data",
            "train",
        )
    )

    batch_size = batch_size
    epochs = epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    hr_image, _ = dataset[0]
    C, H, _ = hr_image.shape

    model = Unet(
        dim=H,
        channels=C,
        dim_mults=(1, 2, 4,),
        self_condition=True,
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    state = {
        "loss_train":[]
    }

    loss_float = 0.
    logger.info("Model and optimizer are defined.")
    logger.info("Starting the training. ")
    
    with logging_redirect_tqdm():
        for epoch in range(epochs):
            for _, (lr_batch, hr_batch) in tqdm(
                    enumerate(dataloader), 
                    desc=f"Loss: {loss_float}, Epoch: {epoch}",
                    total=len(dataloader)
                ):
                optimizer.zero_grad()

                batch_size, _, _, _ = lr_batch.shape
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                t = torch.randint(0, timesteps, (batch_size,), device=device).long()

                loss = p_spectral_losses(
                    denoise_model=model, 
                    x_start=hr_batch, 
                    t=t,
                    self_condition=lr_batch,
                )
                loss.backward()

                loss_float = float(loss.detach().cpu())
                state['loss_train'].append(loss_float)

                optimizer.step()
                break

    logger.info(f"Finished training. ")
    
    state["model_state_dict"] = model.state_dict()
    state["optimizer_state_dict"] = optimizer.state_dict()
    state["model_kwargs"] = {
        "dim":H,
        "channels":C,
        "dim_mults":(1, 2, 4,),
        "self_condition":True,
    }
    state["other"] = {
        "epochs":epochs,
        "batch_size":batch_size,
        "device":device,
        "seed":seed,
    }
        
    return state
