from tqdm import tqdm
from logging import basicConfig, INFO
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
from torch.fft import rfft2
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet import Unet
from data_utils import WeatherFieldsDataset
from diffusion_class import GaussianDiffusion

def test_diffusion(diffusion_state_path, test_dataset_path, seed):
    basicConfig(level=INFO)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    saved_state = torch.load(
        diffusion_state_path
    )
    model = Unet()

    diffusion = GaussianDiffusion(
        denoise_fn=model,
        image_size=128,
        channels=3,
        loss_type="fl2",
    )

    diffusion.set_new_noise_schedule(
        device,
        schedule_type='linear',
        n_timestep=1000,
    )

    diffusion.load_state_dict(saved_state['model_state_dict'])
    
    diffusion = diffusion.to(device)
    test_dataset = WeatherFieldsDataset(
        path_to_data=test_dataset_path
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False,
    )

    metrics = {
        'n':0,
        'l2':0.,
        'fourier_real_l2':0.,
        'fourier_imag_l2':0.,
        'PSD_l2':0.,
    }

    with torch.no_grad():
        with logging_redirect_tqdm():
            tqdm_loader = tqdm(
                enumerate(test_dataloader), 
                total=len(test_dataloader)
            )
            for _, batch in tqdm_loader:

                batch['LR'] = batch['LR'].to(device)
                batch_HR = batch['HR'].to(device)
                batch_SR = diffusion.super_resolution(
                    batch['LR'],
                    continous=False
                )

                fourier_SR = rfft2(batch_SR)
                fourier_HR = rfft2(batch_HR)

                psd_SR = fourier_SR.real**2 + fourier_SR.imag**2
                psd_HR = fourier_HR.real**2 + fourier_HR.imag**2

                n = batch_HR.shape[0] 

                metrics['n'] += n

                metrics['l2'] += F.mse_loss(
                    batch_SR, batch_HR, 
                    reduction='none').mean(
                        dim=(1, 2, 3)
                ).sum()

                metrics['fourier_real_l2'] += F.mse_loss(
                    fourier_SR.real, fourier_HR.real, 
                    reduction='none').mean(
                        dim=(1, 2, 3)
                ).sum()

                metrics['fourier_imag_l2'] += F.mse_loss(
                    fourier_SR.imag, fourier_HR.imag, 
                    reduction='none').mean(
                        dim=(1, 2, 3)
                ).sum()

                metrics['PSD_l2'] += F.mse_loss(
                    psd_SR, psd_HR, 
                    reduction='none').mean(
                        dim=(1, 2, 3)
                ).sum()

            metrics['l2'] /= n
            metrics['fourier_real_l2'] /= n
            metrics['fourier_imag_l2'] /= n
            metrics['PSD_l2'] /= n
            return metrics