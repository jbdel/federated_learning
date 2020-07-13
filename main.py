import numpy as np
import argparse
import torch
import random
from torch.utils.data import DataLoader
from retina_dataset import Retina_Dataset
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--dataloader', type=str, default="Retina_Dataset")
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Base on args given, compute new args
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader
    train_dset = eval(args.dataloader)('train', args)
    eval_dset = eval(args.dataloader)('test', args)
    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    eval_loader = DataLoader(eval_dset, args.batch_size, num_workers=8, pin_memory=True)

    net = eval(args.model)(pretrained=True)
    net.fc = nn.Linear(512, 5)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="sum").cuda()

    loss_tmp = 0
    for iteration, data in enumerate(train_loader):
        inputs = data['image'].cuda()
        labels = data['label'].cuda()
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels.flatten())
        loss.backward()
        optimizer.step()
        loss_tmp += loss.cpu().data.numpy()

        print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e" % (
                  1,
                  iteration,
                  int(len(train_loader.dataset) / args.batch_size),
                  loss_tmp / args.batch_size,
                  args.lr,# *[group['lr'] for group in optim.param_groups],
              ), end='          ')

    net.train(False)
    accuracy = []

    for iteration, data in enumerate(eval_loader):
        inputs = data['image'].cuda()
        labels = data['label']
        pred = net(inputs).cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        accuracy += list(np.argmax(pred, axis=1) == labels.flatten())
    print(100 * np.mean(np.array(accuracy)))
    net.train(True)


