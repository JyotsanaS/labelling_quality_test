import os
import tifffile
from torch.utils.data import Dataset
import torch
import numpy as np
import torch
import os

def apply_horizontal_flip(image_array):
    if np.random.rand() < 0.5:  # 50% chance of applying horizontal flip
        return np.flip(image_array, axis=1)
    return image_array

def apply_vertical_flip(image_array):
    if np.random.rand() < 0.5:  # 50% chance of applying vertical flip
        return np.flip(image_array, axis=0)
    return image_array

def apply_channel_drop(image_array, drop_probability=0.2):
    if np.random.rand() < drop_probability:
        # Choose a random channel to drop
        channel_to_drop = np.random.randint(0, image_array.shape[2])
        # Set the chosen channel to zero
        image_array[:, :, channel_to_drop] = 0
    return image_array

def apply_random_transformations(image_array):
    image_array = apply_horizontal_flip(image_array)
    image_array = apply_vertical_flip(image_array)
    image_array = apply_channel_drop(image_array)

    return image_array

class CustomDataset(Dataset):
    def __init__(self, root_folder, transform=False):
        self.root_folder = root_folder
        self.image_paths = self.get_image_paths()
        self.transform = transform
        self.class_to_int = {'Pasture': 0,
                             'HerbaceousVegetation': 1,
                             'AnnualCrop': 2,
                             'PermanentCrop': 3,
                             'Highway': 4,
                             'Residential': 5,
                             'Industrial': 6,
                             'River': 7,
                             'Forest': 8,
                             'SeaLake': 9}

    def get_image_paths(self):
        image_paths = []
        for class_folder in os.listdir(self.root_folder):
            class_folder_path = os.path.join(self.root_folder, class_folder)
            if os.path.isdir(class_folder_path):
                for image_file in os.listdir(class_folder_path):
                    if image_file.endswith(".tif"):
                        image_paths.append(os.path.join(class_folder_path, image_file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = tifffile.imread(image_path)
        image = image.astype(float)
        
        if self.transform:
            image = apply_random_transformations(image)
            
        image = np.transpose(image, (2, 1, 0))
        image = torch.from_numpy(np.flip(image,axis=0).copy())

        # Get label as an integer
        label = torch.tensor(self.class_to_int[os.path.basename(os.path.dirname(image_path))])

        return image, label