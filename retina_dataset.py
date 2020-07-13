from __future__ import print_function
import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
import csv
from collections import Counter
from torchvision import transforms
from PIL import Image

def get_transform(name):

    if name == 'train':
        data_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.Resize([256, 256]),
             transforms.RandomCrop([224, 224]),
             transforms.ToTensor(),
             # transforms.Normalize([0.485], [0.229])
             ])
    else:
        data_transforms = transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.CenterCrop([224, 224]),
             transforms.ToTensor(),
             # transforms.Normalize([0.485], [0.229])
             ])
    return data_transforms

class Retina_Dataset(Dataset):
    def __init__(self, name, args):
        super(Retina_Dataset, self).__init__()
        assert name in ['train', 'test']
        self.name = name
        self.args = args
        self.transform = get_transform(name)

        # Loading images
        files = glob.glob(os.path.join(args.data_dir, name, "*"))
        files = files[:100]
        self.images = {}
        for file in files:
            filename = os.path.basename(os.path.splitext(file)[0])
            self.images[filename] = Image.fromarray(np.load(file).astype(np.uint8))

        # Loading labels
        reader = csv.DictReader(open(os.path.join(args.data_dir, name+"Labels.csv")), delimiter=',')
        self.labels = {}
        for row in reader:
                self.labels[row['image']] = int(row['level'])

        print("Label balance for " + name, Counter(self.labels.values()))

        self.set = list(self.images.keys())

    def __getitem__(self, idx):
        key = self.set[idx]
        return {'image': self.transform(self.images[key]),
                'label':  np.array([self.labels[key]]),
                'img_name': key}

    def __len__(self):
        return len(self.set)