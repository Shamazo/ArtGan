import os
from torch.utils.data import Dataset
from skimage import io
import pickle
import numpy as np
import torch


class ArtDataset(Dataset):
    def __init__(self, data_dir, from_pickle=True, transform=None):
        """
        Args
            :param data_dir location of data with pngs {artist name}_{object id}.png:
            :param transform:
        """
        self.data_dir = data_dir
        self.frame_names = os.listdir(data_dir)
        self.transform = transform
        # self.frame_names = [self.frame_names[9]] * 65
        # self.frame_names=['Blinky Palermo_60262.png'] * 129

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):
        frame_name = self.frame_names[idx]
        frame_path = os.path.join(self.data_dir, frame_name)
        # print(frame_path)
        frame = io.imread(frame_path)
        #channels first for pytorch
        frame = np.moveaxis(frame, -1, 0)
        return torch.tensor(frame)
        if self.transform:
            frame = self.transform(frame)
        return {'frame': frame}

