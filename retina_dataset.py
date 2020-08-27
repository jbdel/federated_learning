from __future__ import print_function
from torch.utils.data import Dataset
import glob
import os
import numpy as np
import pandas as pd
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
    def __init__(self, name, args, num_site):
        super(Retina_Dataset, self).__init__()
        assert name in ['train', 'val']
        self.name = name
        self.args = args
        self.num_site = num_site
        self.transform = get_transform(name)

        # Loading and categorizing labels
        reader = pd.read_csv(os.path.join(args.data_dir, name + "_liang.csv"))  # to shuffle ?
        reader.drop(reader.loc[reader['image'] == '492_right'].index, inplace=True)  # bad sample
        reader.drop(reader.loc[reader['image'] == '25313_right'].index, inplace=True)  # bad sample
        reader.drop(reader.loc[reader['image'] == '27096_right'].index, inplace=True)  # bad sample

        labels = []
        if args.task_binary:
            labels.append(reader.loc[reader['level'] == 0])
            labels.append(reader.loc[reader['level'] > 0])
        else:
            for i in range(reader['level'].nunique()):
                labels.append(reader.loc[reader['level'] == i])

        assert len(args.distribution) == len(labels), "labels distribution doesnt match " \
                                                      "the number of different labels"
        self.labels = {}
        self.images = {}

        # Feed the dictionary of the dataloader according to its site number and distribution
        for i, label in enumerate(labels):
            # eval or test
            if num_site == 'val':
                num_samples_for_label = len(label)  # take all
                start_feeding = 0
            else:  # train
                num_samples_for_label = int(self.args.samples_site * self.args.distribution[i])
                start_feeding = self.num_site * num_samples_for_label

            for j, (image, c) in enumerate(zip(label['image'], label['level'])):
                if j >= start_feeding:
                    gt = (int(c) > 0) if args.task_binary else int(c)
                    self.labels[image] = int(gt)
                    self.images[image] = Image.fromarray(np.load(os.path.join(args.data_dir,
                                                                              'images',
                                                                              image + ".npy")))
                    if j == (start_feeding + num_samples_for_label - 1):
                        break

        print("Label balance for site " + str(self.num_site), Counter(self.labels.values()))
        self.set = list(self.labels.keys())

    def __getitem__(self, idx):
        key = self.set[idx]
        return {'image': self.transform(self.images[key]),
                'label': np.array([self.labels[key]]),
                'img_name': key}

    def __len__(self):
        return len(self.set)
