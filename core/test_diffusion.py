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

@torch.no_grad()
def p_sample(model, x, self_condition, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(
                x=x, 
                time=t,
                x_self_cond=self_condition
            ) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_hr_images_loop(model, lr_images):
    device = next(model.parameters()).device
    shape = lr_images.shape
    batch_size, _, _, _ = lr_images.shape

    img = torch.randn(shape, device=device)
    for i in reversed(range(0, timesteps)):
        img = p_sample(
            model = model,
            x = img,
            self_condition = lr_images,
            t = torch.full(
                (batch_size,),
                i,
                device=device,
                dtype=torch.long
            ), 
            t_index = i
        )
    return img

@torch.no_grad()
def get_hr_images(model, lr_images):
    return sample_hr_images_loop(
        model=model,
        lr_images=lr_images,
    )

@torch.no_grad()
def test_diffusion(model, batch_size, seed):
    basicConfig(level=INFO)
    torch.manual_seed(seed)
    
    dataset = WeatherFieldsDataset(
        path_to_folder=os.path.join(
            "../data",
            "wrf_data",
            "test",
        )
    )

    batch_size = batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)

    loss_float = 0.
    logger.info("Starting the testing. ")
    
    with logging_redirect_tqdm():
        for _, (lr_batch, hr_batch) in tqdm(
                enumerate(dataloader), 
                total=len(dataloader)
            ):
            batch_size, _, _, _ = lr_batch.shape
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            reconstructed_hr_batch = get_hr_images(
                model=model, 
                lr_images=lr_batch
            )
            
            images_mean_sqrt = ((hr_batch - reconstructed_hr_batch)**2).mean(1, 2, 3)
            loss_float += images_mean_sqrt.sum()

    logger.info(f"Finished testing. ")
         
    return loss_float / len(dataset)
