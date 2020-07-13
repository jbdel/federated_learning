import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import DatasetDist
from torch.utils.tensorboard import SummaryWriter

def get_central_files(ssh_client, central_path, model_file_name, only_model = False):
    ftp_client = ssh_client.open_sftp()
    print('Waiting for others client, connecting with server, loading files from server path', central_path)
    # we only pass model names or the train/test_process files
    if only_model:
        try:
            files = ftp_client.listdir(central_path)
            for file in files:
                if model_file_name in file:
                    ftp_client.get(os.path.join(central_path, file), os.path.join('.', central_path, file))
        except:
            raise ValueError('Central server donot contain trained model---')
    else:
        files = ftp_client.listdir(central_path)
        for file in files:
            if model_file_name in file or file.endswith('.csv'):
                try:
                    ftp_client.get(os.path.join(central_path, file), os.path.join('.', central_path, file))
                except:
                    print(
                        'Previous instituions donot finish train yet, waiting for them to transfer models and training process files:')
                    print(os.path.join(central_path, file))

    ftp_client.close()


def put_file(ssh_client, central_path, file):
    ftp_client = ssh_client.open_sftp()
    ftp_client.put(os.path.join(central_path, file), os.path.join(central_path, file))
    ftp_client.close()


def initization_configure(opt, ssh_client):

    #Generating folder for saving intermediate results and loading files from central server
    if not os.path.exists(opt.central_path):
        os.makedirs(opt.central_path)
        print('Generate local folder for saving models and training progress', opt.central_path)
    try:
        sftp = ssh_client.open_sftp()
        sftp.mkdir(opt.central_path)
        print('Generate central folder for saving models and training progress', opt.central_path)
    except:
        pass

    opt.device = torch.device("cuda:{gpu_id}".format(gpu_id = opt.gpu_ids) if torch.cuda.is_available() else "cpu")

    np.random.seed(opt.SEED)  # if numpy is used
    torch.manual_seed(opt.SEED)
    if not opt.device == 'cpu':
        torch.cuda.manual_seed(opt.SEED)
    log_file_tensorboard = os.path.join('.', opt.central_path, 'log', opt.dis_model_name + '/')
    if not os.path.exists(log_file_tensorboard):
        os.makedirs(log_file_tensorboard)
        print('Generator path to store logfile', log_file_tensorboard)
    # opt.writer = SummaryWriter(log_file_tensorboard)

    ## dataset configure
    label_path = os.path.join( opt.data_path, 'labels.csv')
    if not opt.regression:
        opt.labels = {line.strip().split(',')[0]: int(line.strip().split(',')[1]) for line in
                      open(label_path)}
    else:
        opt.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                      open(label_path)}

    if opt.phase == 'train':
        train_set = DatasetDist(opt, 'train')
        train_set_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
        opt.train_set_loader = train_set_loader

        if opt.val:
            opt.phase = 'test'
            val_set = DatasetDist(opt, 'val')
            val_set_loader = DataLoader(dataset=val_set, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                          shuffle=False)
            opt.phase = 'train'
            opt.val_set_loader = val_set_loader
    else:
        test_set = DatasetDist(opt, 'test')
        test_set_loader = DataLoader(dataset=test_set, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                     shuffle=False)
        opt.test_set_loader = test_set_loader

    opt.best_acc = 0


