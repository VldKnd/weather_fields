import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Lambda,
    Compose,
    Resize
)
from torchvision.transforms import functional as TV_F
from tqdm.contrib.logging import logging_redirect_tqdm

from typing import Dict
from tqdm import tqdm
from logging import getLogger
from datetime import datetime
import numpy as np
import random
import os

logger = getLogger(__name__)

class WeatherFieldsDataset(Dataset):
    def __init__(self, path_to_data, transform=None):
        """
        Retrieves all npy tensors from path_to_data folder as dictionary by matching there names.
        """
        date_idx_to_file_pathes = {}

        path = os.path.abspath(path_to_data)

        logger.info("Creating dataset. ")
        with logging_redirect_tqdm():
            for root, _, files in tqdm(os.walk(path)):
                for file in files:
                    if file.endswith("npy"):
                        if "high_res" in file:
                            hr_file_name = file
                            hr_file_name = hr_file_name.replace("_high_res_", "_")
                            hr_file_name = hr_file_name.replace(".npy", "")
                            _, date, number = hr_file_name.split("_")
                            index = int("".join(date.split("-")) + number)
                            if index in date_idx_to_file_pathes:
                                date_idx_to_file_pathes[index]['hr'] = os.path.join(
                                    root, 
                                    file
                                )
                            else:
                                date_idx_to_file_pathes[index]= {
                                    'hr':os.path.join(
                                        root, 
                                        file
                                    )
                                }
                        else:
                            lr_file_name = file
                            lr_file_name = lr_file_name.replace("_low_res_", "_")
                            lr_file_name = lr_file_name.replace(".npy", "")
                            _, date, number = lr_file_name.split("_")
                            index = int("".join(date.split("-")) + number)
                            if index in date_idx_to_file_pathes:
                                date_idx_to_file_pathes[index]['lr'] = os.path.join(
                                    root, 
                                    file
                                )
                            else:
                                date_idx_to_file_pathes[index]= {
                                    'lr':os.path.join(
                                        root, 
                                        file
                                    )
                                }
                            
        self.transform = transform
        self.date_idx_to_file_pathes = date_idx_to_file_pathes 
        self.sorted_date_idx = list(date_idx_to_file_pathes.keys())
        self.sorted_date_idx.sort()
        self.lr_transform = Compose([
            Lambda(lambda t: (t / 255.)),
            Lambda(lambda t: (t*2) - 1),
            Lambda(lambda t: t.permute(2, 0, 1)),
            Resize(size=(128, 128), antialias=True),
        ])

        self.hr_transform = Compose([
            Lambda(lambda t: (t / 255.)),
            Lambda(lambda t: (t*2) - 1),
            Lambda(lambda t: t.permute(2, 0, 1))
        ])
  
    def __len__(self):
        return len(self.sorted_date_idx)

    def __getitem__(self, index) -> Dict[str, torch.TensorType]:
        date_idx = self.sorted_date_idx[index]
        
        lr_file_path = self.date_idx_to_file_pathes[date_idx]['lr']
        hr_file_path = self.date_idx_to_file_pathes[date_idx]['hr']

        with torch.no_grad():
            lr_image = torch.from_numpy(np.load(lr_file_path))
            hr_image = torch.from_numpy(np.load(hr_file_path))

            lr_image = self.lr_transform(lr_image)
            hr_image = self.hr_transform(hr_image)

            if random.random() < 0.5:
                lr_image = TV_F.hflip(lr_image)
                hr_image = TV_F.hflip(hr_image)
                
            if random.random() < 0.5:
                lr_image = TV_F.vflip(lr_image)
                hr_image = TV_F.vflip(hr_image)

        return { 'LR' : lr_image, 'HR' : hr_image }
    
def save_state(state: Dict, folder_path: str = "", name: str = ""):
    now = datetime.now()
    now = now.strftime('%m_%d_%M_%S')
    if name != "":
        file_name = name + "_" + now + ".pkl"
    else:
        file_name = now + ".pkl"
        
    if folder_path == "":
        pwd_path = os.path.abspath("..")
        folder_path = os.path.join(
            pwd_path,
            "saves"
        )

    file_path = os.path.join(
        folder_path,
        file_name
    )

    logger.info(f"Saving the checkpoint at {file_path}. ")

    torch.save(state, file_path)