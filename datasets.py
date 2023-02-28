import torch
from torch import nn
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from alg.vcl_net import MultiHeadVCLSplitNotMNIST, Initialization
from alg.kcenter import KCenter
import os, tarfile
from torchvision import datasets

def get_split_notMNIST(task_idx, device, train, test):
    # "The notMNIST dataset contains 400,000 images of the characters from 
    #  A to J with different font styles. We consider ﬁve binary classiﬁcation 
    #  tasks: A/F, B/G, C/H, D/I, and E/J"

    labels = (task_idx, task_idx + 5) # labels are 0 to 9 for A to J in order
    ds_train_filtered = list(filter(lambda item: item[1] in labels, train))
    ds_test_filtered = list(filter(lambda item: item[1] in labels, test))
    train_x = nn.Flatten()(torch.cat([d[0] for d in ds_train_filtered]))
    train_y = torch.tensor([0 if d[1] == task_idx else 1 for d in ds_train_filtered])
    
    test_x = nn.Flatten()(torch.cat([d[0] for d in ds_test_filtered]))
    test_y = torch.tensor([0 if d[1] == task_idx else 1 for d in ds_test_filtered])

    return train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)

def get_split_MNIST(task_idx, device, train, test):
    labels = np.array_split(range(0, 10), 5)[task_idx]
    train_filtered = list(filter(lambda item: item[1] in labels, train))
    test_filtered = list(filter(lambda item: item[1] in labels, test))
    train_x = nn.Flatten()(torch.cat([d[0] for d in train_filtered]))
    train_y = torch.tensor([d[1] - task_idx * 2 for d in train_filtered])
    
    test_x = nn.Flatten()(torch.cat([d[0] for d in test_filtered]))
    test_y = torch.tensor([d[1] - task_idx * 2 for d in test_filtered])

    return train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)

def get_split_CIFAR10(task_idx, device, train, test):
    labels = np.array_split(range(0, 10), 5)[task_idx]
    train_filtered = list(filter(lambda item: item[1] in labels, train))
    test_filtered = list(filter(lambda item: item[1] in labels, test))
    train_x = torch.cat([d[0][None, :, :, :] for d in train_filtered])
    train_y = torch.tensor([d[1] - task_idx * 2 for d in train_filtered])
    
    test_x = torch.cat([d[0][None, :, :, :] for d in test_filtered])
    test_y = torch.tensor([d[1] - task_idx * 2 for d in test_filtered])

    return train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)


class MultiTaskDataset:
    def __init__(self, dataset_name: str, device: str):
        if dataset_name == 'split MNIST':
            self.train = datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
            self.test = datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)
        elif dataset_name == 'split notMNIST':
            tar_file = './data/notMNIST_small.tar.gz'
            folder = './data/notMNIST_small'
            if not os.path.exists(folder):
                if not os.path.exists(tar_file):
                    raise Exception("notMNIST dataset is missing, download at: https://www.kaggle.com/datasets/lubaroli/notmnist")
                with tarfile.open(tar_file) as f:
                    f.extractall('./data/')
                assert os.path.exists(folder), "Something went wrong with extraction."

            # these are some empty images
            empty_files = ['A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png', 'F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png']
            for file_name in empty_files:
                file_name = f"{folder}/{file_name}"
                if os.path.exists(file_name):
                    os.remove(file_name)


            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])

            image_folder = ImageFolder(folder, transform=transform)

            # Same split ratio as in MNIST
            self.train, self.test = random_split(image_folder, [1 - 0.143, 0.143], generator=torch.Generator().manual_seed(0))
        elif dataset_name == 'split CIFAR-10':
            self.train = datasets.CIFAR10("./data", train=True, transform=transforms.ToTensor(), download=True)
            self.test = datasets.CIFAR10("./data", train=False, transform=transforms.ToTensor(), download=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.dataset_name = dataset_name
        self.device = device
        
    def get_task_dataset(self, task_idx: int):
        if self.dataset_name == 'split MNIST':
            return get_split_MNIST(task_idx, self.device, self.train, self.test)
        elif self.dataset_name == 'split notMNIST':
            return get_split_notMNIST(task_idx, self.device, self.train, self.test)
        elif self.dataset_name == 'split CIFAR-10':
            return get_split_CIFAR10(task_idx, self.device, self.train, self.test)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")