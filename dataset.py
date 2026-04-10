import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

def make_train_transform(img_size: int):
    return A.Compose([
        # TODO: Add train augmentation logic here
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])

def make_val_test_transform(img_size: int):
    return A.Compose([
        # TODO: Add val/test resize logic here
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])

class AlzeyeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing metadata.
            img_dir (str): Path to the root directory where images are stored.
            transform (callable, optional): Augmentation function to apply.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # TODO: Combine CSV info and img_dir to create the actual file paths
        # Example: cfp_img_path = os.path.join(self.img_dir, f"{row['imageHash_cfp']}.jpeg")
        cfp_img_path = row['image_path']

        # TODO: Load images (using OpenCV or PIL)
        cfp_image = cv2.imread(cfp_img_path)
        cfp_image = cv2.cvtColor(cfp_image, cv2.COLOR_BGR2RGB)
        
        cfp_image = self.transform(image=cfp_image)['image']
        
        cvd_label = row['cvd_label']

        return_dict = {
            'cfp_image': cfp_image,
            'cvd_label': torch.tensor(cvd_label, dtype=torch.float32)
        }
        return return_dict


class AlzeyeDataModule(L.LightningDataModule):
    def __init__(self, data_dir='./csv', img_dir='./images', img_size=224, batch_size=32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # TODO: Connect train/val/test CSV paths and assign Dataset instances
        pass
    
    def train_dataloader(self):
        # Use standard DataLoader (shuffle=True is required for training)
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )