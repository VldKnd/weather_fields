import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Lambda,
    Compose,
    Resize
)
from torchvision.transforms import functional as TV_F

from logging import getLogger
from datetime import datetime
import numpy as np
import random
import os

logger = getLogger(__name__)

class WeatherFieldsDataset(Dataset):
    def __init__(self, root_dir, path_to_folder, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        lr_data_folder = os.path.join(
            root_dir, 
            path_to_folder,
            "train_2017_lr",
        )
        hr_data_folder = os.path.join(
            root_dir, 
            path_to_folder,
            "train_2017_hr",
        )
        
        date_idx_to_hr_file_names = {}

        for hr_file_name in os.listdir(hr_data_folder):
            hr_file_name_copy = hr_file_name
            hr_file_name = hr_file_name.replace("_high_res_", "_")
            hr_file_name = hr_file_name.replace(".npy", "")
            _, date, number = hr_file_name.split("_")
            index = int("".join(date.split("-")) + number)
            date_idx_to_hr_file_names[index] = hr_file_name_copy

        date_idx_to_file_pathes = {}

        for lr_file_name in os.listdir(lr_data_folder):
            lr_file_name_copy = lr_file_name
            lr_file_name = lr_file_name.replace("_low_res_", "_")
            lr_file_name = lr_file_name.replace(".npy", "")
            _, date, number = lr_file_name.split("_")
            index = int("".join(date.split("-")) + number)
            hr_file_name = date_idx_to_hr_file_names.get(index)
            if hr_file_name is not None:
                date_idx_to_file_pathes[index] = (
                    os.path.join(
                        lr_data_folder,
                        lr_file_name_copy,
                    ),
                    os.path.join(
                        hr_data_folder,
                        hr_file_name,
                    )
                )

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

    def __getitem__(self, index) -> torch.TensorType:
        date_idx = self.sorted_date_idx[index]
        lr_file_path, hr_file_path = self.date_idx_to_file_pathes[date_idx]

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

        return lr_image, hr_image
    
def save_state(state, folder_path: str = "", name: str = ""):
    now = datetime.now()
    now = now.strftime('%m_%d_%M_%S')
    if name != "":
        file_name = now + ".pkl"
    else:
        file_name = name + now + ".pkl"
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