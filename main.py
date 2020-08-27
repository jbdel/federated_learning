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
import ast
from operator import add


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir', type=str, default="data")

    # Model
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch_per', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--task_binary', type=bool, default=True)

    # Federated
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--sites', type=int, default=4)
    parser.add_argument('--samples_site', type=int, default=1500)
    parser.add_argument('--distribution', type=str, default='['
                                                            '[[0.2,0.8],[0.2,0.8],[0.2,0.8],[0.2,0.8]],'
                                                            '[[0.8,0.2],[0.8,0.2],[0.8,0.2],[0.8,0.2]]'
                                                            ']')
    # parser.add_argument('--distribution', type=str, default='[[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]]')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Handle distribution
    args.distribution = np.array(ast.literal_eval(args.distribution))
    if len(args.distribution.shape) == 2:
        print("Using same distribution for every round")
        args.distribution = np.expand_dims(args.distribution, axis=0)

    args.same_distribution = args.distribution.shape[0] == 1
    print("Distribution is: \n", args.distribution)

    # DataLoader
    round_loaders = []
    sum_labels = [0] * args.distribution.shape[-1]  # keeping track of used labels
    for i in range(len(args.distribution)):
        site_loaders = []
        for j in range(args.sites):
            distribution = args.distribution[i][j]
            train_dset = Retina_Dataset('train', args, num_site=[i, j], distribution=distribution,
                                        index_label=sum_labels)
            sum_labels = list(map(add, sum_labels, distribution))
            site_loaders.append(DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=4))
        round_loaders.append(site_loaders)

    eval_dset = Retina_Dataset('val', args, num_site='val')
    eval_loader = DataLoader(eval_dset, args.batch_size, num_workers=4)

    net = eval(args.model)(pretrained=True)
    net.fc = nn.Linear(512, 2 if args.task_binary else 5)
    net.cuda()

    criterion = nn.CrossEntropyLoss(reduction="sum").cuda()
    for r in range(args.rounds):
        net.train(True)
        site_loaders = round_loaders[r if not args.same_distribution else 0]
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
                        r + 1,
                        i + 1,
                        epoch + 1,
                        iteration,
                        int(len(site_loader.dataset) / args.batch_size),
                        loss.cpu().data.numpy() / args.batch_size,
                        args.lr,  # *[group['lr'] for group in optim.param_groups],
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
