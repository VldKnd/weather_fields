import os
import argparse
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Dict

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader


from data_utils import WeatherFieldsDataset
from unet import Unet
from diffusion_class import GaussianDiffusion
from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)

timesteps = 300


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

def train_diffusion(batch_size, epochs, seed, loss_type):
    basicConfig(level=INFO)
    torch.manual_seed(seed)

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

    model_dim=H
    model_channels=C
    model_dim_mults=(1, 2, 4,)
    model_self_condition=True

    model = Unet(
        dim=model_dim,
        channels=model_channels,
        dim_mults=model_dim_mults,
        self_condition=model_self_condition,
    )
    model.to(device)

    diffusion = GaussianDiffusion(
        denoise_fn=model,
        image_size=128,
        channels=3,
        loss_type=loss_type,
    )

    diffusion.set_new_noise_schedule(
        device,
        schedule_type='linear',
        n_timestep=1000,
    )

    optimizer = Adam(diffusion.parameters(), lr=1e-3)

    state = {
        "loss_train":[]
    }

    loss_float = 0.

    logger.info("Model and optimizer are defined.")
    logger.info("Starting the training. ")
    
    with logging_redirect_tqdm():
        for epoch in range(epochs):
            for _, batch in tqdm(
                    enumerate(dataloader), 
                    desc=f"Loss: {loss_float}, Epoch: {epoch}",
                    total=len(dataloader)
                ):
                optimizer.zero_grad()

                batch['LR'].to(device)
                batch['HR'].to(device)

                loss = diffusion.forward(batch)
                loss.backward()

                loss_float = float(loss.detach().cpu())
                state['loss_train'].append(loss_float)

                optimizer.step()
                break

    logger.info(f"Finished training. ")
    
    state["model_state_dict"] = diffusion.state_dict()
    state["optimizer_state_dict"] = optimizer.state_dict()
    state["beta_schedule_params"] = {
        "schedule_type":'linear',
        "n_timestep":1000,
        "linear_start":1e-4, 
        "linear_end":2e-2, 
        "cosine_s":8e-3,
    }
    state["model_kwargs"] = {
        "dim":model_dim,
        "channels":model_channels,
        "dim_mults":model_dim_mults,
        "self_condition":model_self_condition,
    }
    state["other"] = {
        "epochs":epochs,
        "batch_size":batch_size,
        "device":device,
        "seed":seed,
    }
        
    return state
