import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import params


class OfficeDataset(data.Dataset):
    def __init__(self, root, text_path, image_size, train=True, transform=None):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.text_path = text_path
        self.image_size = image_size
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        self.train_data, self.train_labels, self.train_domains = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
            self.train_domains = self.train_domains[indices[0:self.dataset_size]]
        # self.train_data = self.train_data.transpose(0, 2, 3, 1)  # convert to HWC

    def __getitem__(self, index):
        img, label, domain = self.train_data[index, ::], self.train_labels[index], self.train_domains[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        domain = torch.LongTensor([np.int64(domain).item()])
        # img = img.permute(1, 2, 0)
        # img *= 255.0
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

        with ThreadPoolExecutor(max_workers=16) as executor:
            image_list = [Image.open(os.path.join(self.root, p)).resize((self.image_size, self.image_size)) for p in tqdm(paths)]
        images = np.array([np.array(im) for im in image_list])
        labels = lines[:, 1].astype(np.int32)
        domains = lines[:, 2].astype(np.int32)
        self.dataset_size = labels.shape[0]
        return images, labels, domains



def get_office(text_path, image_size, train):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    usps_dataset = OfficeDataset(root=params.data_root,
                        train=train,
                        text_path=text_path,
                        image_size=image_size,
                        transform=pre_process,)

    if train:
        usps_data_loader = torch.utils.data.DataLoader(
            dataset=usps_dataset,
            batch_size=params.batch_size,
            drop_last=True,
            shuffle=True)
    else:
        usps_data_loader = torch.utils.data.DataLoader(
            dataset=usps_dataset,
            batch_size=params.batch_size,
            shuffle=False)

    return usps_data_loader
