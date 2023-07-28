import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image

import params



class HomeDataset(data.Dataset):
    def __init__(self, root, text_path, train=True, transform=None):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.text_path = text_path
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        self.train_data, self.train_labels, self.train_domaoins = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size]]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
            self.train_domains = self.train_domains[indices[0:self.dataset_size]]
        # self.train_data *= 255.0
        # self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, label, domain = self.train_data[index], self.train_labels[index], self.train_domains[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        domain = torch.LongTensor([np.int64(domain).item()])
        return img, label, domain

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def load_samples(self):
        """Load sample images from dataset."""
        with open(self.text_path, "r") as f:
            lines = f.readlines()
            lines = np.array([l.split(' ') for l in lines], dtype=np.object_)
        paths = lines[:, 0]
        images = [transforms.ToTensor()(Image.open(os.path.join(self.root, path))) for path in paths]
        labels = lines[:, 1].astype(np.int32)
        domains = lines[:, 2].astype(np.int32)
        self.dataset_size = labels.shape[0]
        return images, labels, domains



def get_home(text_path, train):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    usps_dataset = HomeDataset(root=params.data_root,
                        train=train,
                        text_path=text_path,
                        transform=pre_process,)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return usps_data_loader
