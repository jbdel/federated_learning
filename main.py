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

NUM_OF_TRAIN_EXAMPLES = 35126

def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataloader', type=str, default="Retina_Dataset")
    parser.add_argument('--data_dir', type=str, default="data")

    # Model
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch_per', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--task_binary', type=bool, default=True)

    # Federated
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--sites', type=int, default=4)
    parser.add_argument('--samples_site', type=int, default=1000)
    parser.add_argument('--distribution', nargs='+', type=float, default=[0.5, 0.5])

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    assert args.samples_site * args.sites <= NUM_OF_TRAIN_EXAMPLES, "Not enough training samples"
    assert sum(args.distribution) == 1, "Distribution needs to sum to 1"
    assert args.task_binary == (len(args.distribution) == 2), "If task is binary, needs only two distribution"
    assert (not args.task_binary) == (len(args.distribution) == 5), "If task is not binary, needs five distribution"

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader
    site_loaders = []
    for i in range(args.sites):
        train_dset = eval(args.dataloader)('train', args, num_site=i)
        site_loaders.append(DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=4))

    eval_dset = eval(args.dataloader)('val', args, num_site='val')
    eval_loader = DataLoader(eval_dset, args.batch_size, num_workers=4)

    net = eval(args.model)(pretrained=True)
    net.fc = nn.Linear(512,
                       2 if args.task_binary else 5)
    net.cuda()

    criterion = nn.CrossEntropyLoss(reduction="sum").cuda()
    for round in range(args.rounds):
        net.train(True)
        for i, site_loader in enumerate(site_loaders):
            optimizer = optim.Adam(net.parameters(), lr=args.lr)
            for epoch in range(args.epoch_per):
                for iteration, data in enumerate(site_loader):
                    inputs = data['image'].cuda()
                    labels = data['label'].cuda()
                    optimizer.zero_grad()
    
                    outputs = net(inputs)
    
                    loss = criterion(outputs, labels.flatten())
                    loss.backward()
                    optimizer.step()
    
                    print("\r[Round %2d][Site %2d][Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e" % (
                              round + 1,
                              i + 1,
                              epoch + 1,
                              iteration,
                              int(len(site_loader.dataset) / args.batch_size),
                              loss.cpu().data.numpy() / args.batch_size,
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
        print('Evaluation accuracy', str(100 * np.mean(np.array(accuracy))))


