import torch.utils.data as data
import torch
import numpy as np

from PIL import Image
import os
import os.path
from torchvision import transforms
from utils.convert_dicom_png import process_dicom
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.mat', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid data directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images



def make_dataset_dicom(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid data directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.dcm'):
                path = os.path.join(root, fname)
                Img = process_dicom(path)
                if Img is not None:
                    images.append(path)
    print('We use dicom for train--updated')
    return images


class DatasetDist(data.Dataset):
    def __init__(self, opt, folder):
        super(DatasetDist, self).__init__()
        self.img_paths = make_dataset_dicom(os.path.join(opt.data_path, folder))
        print('Loading', len(self.img_paths), 'training images from institution', opt.inst_id, 'for', folder, '------')
        self.opt = opt
        self.labels = opt.labels

        if opt.phase == 'train':
            data_transforms = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5),
                 transforms.Resize([opt.load_size, opt.load_size]),
                 transforms.RandomCrop([opt.fine_size, opt.fine_size]),
                 transforms.ToTensor(),
                 # transforms.Normalize([0.485], [0.229])
                ])
        else:
            data_transforms = transforms.Compose(
                [transforms.Resize([opt.load_size, opt.load_size]),
                 transforms.CenterCrop([opt.fine_size, opt.fine_size]),
                 transforms.ToTensor(),
                 # transforms.Normalize([0.485], [0.229])
                 ])

        self.transform = data_transforms

    def __getitem__(self, index):
        name = self.img_paths[index]

        if name.endswith('.npy'):
            Img = np.load(name)
            Img = (Img - Img.min()) / (Img.max() - Img.min()) * 224
            Img = Image.fromarray(Img.astype('uint8'))
        elif name.endswith('.dcm'):
            Img = process_dicom(name)
            Img = Img * 255
            Img = Image.fromarray(Img.astype('uint8'))

        else:
            Img = Image.open(name).convert('RGB')

        input = self.transform(Img)
        if input.shape[0] == 1:
            input = torch.cat([input, input, input])
        tmp_label = self.labels[os.path.basename(name)]

        if self.opt.regression:
            label = torch.FloatTensor(1)
            label[0] = tmp_label
        else:

            label = torch.LongTensor(3)

            label[0] = tmp_label
            label[1] = tmp_label
            label[2] = tmp_label
        return {'input': input, 'label': label, 'Img_paths': name}

    def __len__(self):
        return len(self.img_paths)



