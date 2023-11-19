import os
from typing import Tuple, Optional
from datetime import datetime

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import WeatherFieldsDataset
from schedule import linear_beta_schedule
from unet import Unet

torch.manual_seed(0)

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
    if noise is None:
        noise = torch.randn_like(x_start)

    with torch.no_grad():
        domain_fourier_noise = torch.randn_like(x_start)
        fourier_noise = torch.fft.rfft2(domain_fourier_noise)
        
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
        complex_mse_loss(fourier_noise, torch.fft.rfft2(predicted_domain_fourier_noise))
    )

    return loss

if __name__ == "__main__":

    dataset = WeatherFieldsDataset(
        root_dir=os.path.abspath(".."),
        path_to_folder=os.path.join(
            "data",
            "wrf_data",
        )
    )

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 128
    channels = 3

    hr_image, lr_image = dataset[0]
    C, H, W = hr_image.shape

    model = Unet(
        dim=H,
        channels=C,
        dim_mults=(1, 2, 4,),
        self_condition=True,
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    state = {
        "loss_train":[]
    }

    for epoch in range(epochs):
        for step, (lr_batch, hr_batch) in enumerate(dataloader):
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

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            state['loss_train'].append(float(loss.detach().cpu()))
            optimizer.step()
            
    ### Save checkpoint
    now = datetime.now()
    now = now.strftime('%m_%d_%M_%S')
    file_name = now + "_checkpoint.pkl"

    pwd_path = os.path.abspath("..")
    folder_path = os.path.join(
        pwd_path,
        "checkpoints"
    )

    file_path = os.path.join(
        folder_path,
        file_name
    )

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
    }

    torch.save(state, file_path)
