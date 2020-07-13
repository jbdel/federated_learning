import torch
import torch.nn as nn
import os
import numpy as np
import time
import subprocess
import torch.optim as optim
import torchvision.models as models

from utils.start_config import put_file, get_central_files


class DistrSystem(object):
    def __init__(self, opt, ssh_client):
        self.opt = opt
        self.opt.log = os.path.join(self.opt.central_path, 'log_inst_' + str(self.opt.inst_id) + '.txt')
        self.opt.model_path = os.path.join(self.opt.central_path, self.opt.dis_model_name + '.pth')
        self.ssh_client = ssh_client
        self.set_up_model()
        self.set_up_criterion()
        self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr)


    def set_up_model(self):
        if self.opt.model_architecture == 'ResNet34':
            self.net = models.resnet34(pretrained=True)
            self.net.fc = nn.Linear(512, self.opt.num_classes)
        elif self.opt.model_architecture == 'ResNet18':
            self.net = models.resnet18(pretrained=True)
            self.net.fc = nn.Linear(512, self.opt.num_classes)
        elif self.opt.model_architecture == 'DesNet121':
            self.net = models.densenet121(pretrained=True)
            self.net.fc = nn.Linear(1024, self.opt.num_classes)
        elif self.opt.model_architecture == 'inception_v3':
            self.net = models.inception_v3(pretrained=True)
            self.net.AuxLogits.fc = nn.Linear(768, self.opt.num_classes)
            self.net.fc = nn.Linear(2048, self.opt.num_classes)
        elif self.opt.model_architecture == 'squeezenet1_0':
            self.net = models.squeezenet1_0(pretrained=True)
            self.net.classifier[1] = nn.Conv2d(512, self.opt.num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            raise ValueError('Not implemented model.')
        ## set to cuda and train or test phase
        self.net.to(self.opt.device)
        if self.opt.phase == 'train':
            self.net.train()
        else:
            self.net.eval()
    def set_up_criterion(self):
        if self.opt.regression:
            self.criterion = torch.nn.L1Loss().to(self.opt.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.opt.device)

    def super_print(self, msg):
        print(msg)
        with open(self.opt.log, 'a') as f:
            f.write(msg + '\n')
    def load_model(self, load_path):
        pretrained_state_dict = torch.load(load_path, map_location=str(self.opt.device))
        self.net.load_state_dict(pretrained_state_dict)
        print('Loading models from server', load_path)
    def save_model(self, name = None):
        if name is None:
            save_file_name = self.opt.model_path
        else:
            save_file_name = os.path.join(self.opt.central_path, self.opt.dis_model_name + '_' +  name + '.pth')
        torch.save(self.net.state_dict(), save_file_name)

    def train_one_epoch(self, cycle_epoch, data_set_loader):
        for iteration, data in enumerate(data_set_loader):
            inputs = data['input'].to(self.opt.device)
            labels = data['label'].to(self.opt.device)
            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            if self.opt.regression:
                labels1 = labels
            else:
                labels1 = labels[:, 1]

            loss = self.criterion(outputs, labels1)
            loss.backward()
            self.optimizer.step()

            outputs = outputs.sigmoid()
            _, predicted = torch.max(outputs, 1)
            train_acc = ((predicted == labels1).sum().item()) / labels1.size(0)

            message =  ('Cycle: %s, Inst: %s, Iter: %s, train loss: %.3f, train acc: %.3f' % (
                cycle_epoch, self.opt.inst_id, iteration, loss.item(), train_acc))
            self.super_print(message)


    def val_one_epoch(self, data_set_loader):
        self.net.eval()
        correct = 0
        total = 0
        loss_all = 0
        with torch.no_grad():
            for iteration, data in enumerate(data_set_loader):
                # print(iteration)
                inputs = data['input'].to(self.opt.device)
                labels = data['label'].to(self.opt.device)

                # forward + backward + optimize
                outputs = self.net(inputs)
                if self.opt.regression:
                    # print('We use regression with label of size', labels.shape)
                    labels1 = labels
                else:
                    labels1 = labels[:, 1]
                loss = self.criterion(outputs, labels1)
                loss_all += loss.item()

                outputs = outputs.sigmoid()
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels1).sum().item()
                total += labels.size(0)

        self.loss_test = loss_all / len(data_set_loader)
        self.acc_test = 100 * correct / total

        self.net.train()

    def train(self, train_data_loader):
        ## for the first inst: initialization or from
        if self.opt.inst_id == 1:
            with open(os.path.join(self.opt.central_path, 'train_progress.csv'), 'w') as f:
                f.write('0' + ',' * (4 * self.opt.num_inst + 1) + '\n')
            put_file(self.ssh_client, self.opt.central_path, 'train_progress.csv')

            # if continue train, then load model from central server
            if self.opt.continue_train:
                try:
                    get_central_files(self.ssh_client, self.opt.central_path, self.opt.dis_model_name, only_model=True)
                    load_filename = '%s_%s.pth' % (self.opt.dis_model_name, 'Best')
                    load_path = os.path.join(self.opt.model_path, load_filename)
                    self.load_model(load_path)
                except:
                    print('Central server donot contains previous saved model, we use random initization model')
            else:
                ## then we save the current refresh model and send them to central server
                self.save_model()
                subprocess.run('tar -zcvf %s.tar.gz %s' % (self.opt.model_path, self.opt.model_path), shell=True)
                put_file(self.ssh_client, self.opt.central_path, self.opt.dis_model_name + '.pth.tar.gz')
                subprocess.run('tar xvzf %s.tar.gz' % (self.opt.model_path), shell=True)

        ## then start standard train
        for cycle in range(self.opt.max_cycles):
            ## substep 1: loading file from central server
            while (True):
                get_central_files(self.ssh_client, self.opt.central_path, self.opt.dis_model_name)
                if not os.path.exists(os.path.join(self.opt.central_path, 'train_progress.csv')):
                    continue
                progress_lines = [line.strip().split(',') for line in
                                  open(os.path.join(self.opt.central_path, 'train_progress.csv'))]
                if len(progress_lines) == 0 or int(progress_lines[-1][0]) != cycle:
                    time.sleep(self.opt.sleep_time)
                    continue
                if self.opt.inst_id == 1:
                    if cycle == 0:
                        break
                    if self.opt.val:
                        if progress_lines[-2][-1] != '' and progress_lines[-1][1] == '':
                            break
                    elif progress_lines[-2][self.opt.num_inst] != '' and progress_lines[-1][1] == '':
                        break
                else:
                    if progress_lines[-1][self.opt.inst_id - 1] != '' and progress_lines[-1][self.opt.inst_id] == '':
                        break
            ## substep 2: train local inst
            subprocess.run('tar xvzf %s.tar.gz' % (self.opt.model_path), shell=True)
            self.load_model(self.opt.model_path)
            self.train_one_epoch(cycle, train_data_loader)

            ## substep 3: save local model and send back to server
            self.save_model()
            put_file(self.ssh_client, self.opt.central_path, os.path.basename(self.opt.log))
            ## --- just testing whether we need zip files
            # tic = time.time()
            # put_file(self.ssh_client, self.opt.central_path, self.opt.dis_model_name + '.pth')
            # toc = time.time() - tic
            # print('if we donot zip, then using time', toc )

            ##  -- testing the other zip
            subprocess.run('tar -zcvf %s.tar.gz %s' % (self.opt.model_path, self.opt.model_path), shell=True)
            put_file(self.ssh_client, self.opt.central_path, self.opt.dis_model_name + '.pth.tar.gz')
            subprocess.run('tar xvzf %s.tar.gz' % (self.opt.model_path), shell=True)

            progress_lines = [line.strip().split(',') for line in
                              open(os.path.join(self.opt.central_path, 'train_progress.csv'))]
            progress_lines[-1][self.opt.inst_id] = '1'
            with open(os.path.join(self.opt.central_path, 'train_progress.csv'), 'w') as f:
                for line in progress_lines:
                    f.write(','.join(line) + '\n')
            put_file(self.ssh_client, self.opt.central_path, 'train_progress.csv')

            ## validation or not
            if self.opt.val:
                if (cycle + 1) % self.opt.val_freq == 0:
                    self.test(self.opt.val_set_loader, 'val')
            elif self.opt.inst_id == self.opt.num_inst:
                progress_lines.append(['' for i in range(len(progress_lines[-1]))])
                progress_lines[-1][0] = str(cycle + 1)
                with open(os.path.join(self.opt.central_path, 'train_progress.csv'), 'w') as f:
                    for line in progress_lines:
                        f.write(','.join(line) + '\n')
                put_file(self.ssh_client, self.opt.central_path, 'train_progress.csv')


    ## test or validation

    def test(self, data_loader, dataset = 'val'):
        self.super_print('=' * 80)

        ## step 1: configuration and make sure all the cyclic transfer is completed
        if dataset == 'val':
            while (True):
                ## first loading from central server
                get_central_files(self.ssh_client, self.opt.central_path, self.opt.dis_model_name)
                progress_lines = [line.strip().split(',') for line in
                                  open(os.path.join(self.opt.central_path, 'train_progress.csv'))]
                if len(progress_lines) == 0:
                    time.sleep(self.opt.sleep_time)
                    continue
                if progress_lines[-1][self.opt.num_inst + self.opt.inst_id - 1] != '' and progress_lines[-1][
                    self.opt.num_inst + self.opt.inst_id] == '':
                    break
                time.sleep(self.opt.sleep_time)
        else:
            load_filename = '%s_%s.pth' % (self.opt.dis_model_name, 'Best')
            load_path = os.path.join(self.opt.model_path, load_filename)
            self.load_model(load_path)
            while (True):
                if self.opt.inst_id == 1:
                    with open(os.path.join(self.opt.central_path, 'test_progress.csv'), 'w') as f:
                        f.write(',' * (3 * self.opt.num_inst - 1) + '\n')
                    put_file(self.ssh_client, self.opt.central_path, 'test_progress.csv')
                    break
                else:
                    get_central_files(self.ssh_client, self.opt.central_path, self.opt.dis_model_name)
                    if not os.path.exists(os.path.join(self.opt.central_path, 'test_progress.csv')):
                        time.sleep(self.opt.sleep_time)
                        continue
                    progress_line = [line.strip().split(',') for line in
                                     open(os.path.join(self.opt.central_path, 'test_progress.csv'))][0]
                    if progress_line[self.opt.inst_id - 2] != '' and progress_line[self.opt.inst_id - 1] == '':
                        break
                time.sleep(self.opt.sleep_time)
        ## step 2 loading latest model and start evaluation
        subprocess.run('tar xvzf %s.tar.gz' % (self.opt.model_path), shell=True)
        self.load_model(self.opt.model_path)
        self.val_one_epoch(data_loader)

        ## then cycle the loss and weights
        if dataset == 'val':
            train_progress_lines = [line.strip().split(',') for line in
                                    open(os.path.join(self.opt.central_path, 'train_progress.csv'))]
            cycle = train_progress_lines[-1][0]
            train_progress_lines[-1][self.opt.num_inst + self.opt.inst_id] = str(len(data_loader) * self.opt.batch_size)
            train_progress_lines[-1][2 * self.opt.num_inst + self.opt.inst_id] = str(self.loss_test)
            train_progress_lines[-1][3 * self.opt.num_inst + self.opt.inst_id] = str(self.acc_test)
            self.super_print(
                'Cycle: %s, Inst: %s, val loss: %.3f, val acc: %.3f' % (cycle, self.opt.inst_id, self.loss_test, self.acc_test))
            if self.opt.inst_id == self.opt.num_inst:
                val_numbers = np.asarray([int(train_progress_lines[-1][i]) for i in
                                          range(self.opt.num_inst + 1, 2 * self.opt.num_inst + 1)], dtype=int)
                val_losses = np.asarray([float(train_progress_lines[-1][i]) for i in
                                         range(2 * self.opt.num_inst + 1, 3 * self.opt.num_inst + 1)], dtype=float)
                val_accs = np.asarray([float(train_progress_lines[-1][i]) for i in
                                       range(3 * self.opt.num_inst + 1, 4 * self.opt.num_inst + 1)], dtype=float)
                n_val_overall = np.sum(val_numbers)
                acc_val_overall = np.sum(val_numbers * val_accs) / n_val_overall
                loss_val_overall = np.sum(val_numbers * val_losses) / n_val_overall
                self.super_print('=' * 80)
                self.super_print('Cycle: %s, combined val loss: %.3f, combined val acc: %.3f' % (
                cycle, loss_val_overall, acc_val_overall))
                if cycle == '0' or loss_val_overall < float(train_progress_lines[-2][-1]):
                    train_progress_lines[-1][-1] = str(loss_val_overall)
                    self.super_print('NEW BEST VALIDATION LOSS, SAVING BEST MODEL')
                    subprocess.run(
                        'cp %s.tar.gz %s_best.tar.gz' % (self.opt.model_path, self.opt.model_path),
                        shell=True)
                    put_file(self.ssh_client, self.opt.central_path, '%s_best.tar.gz' % (self.opt.dis_model_name + '.pth'))
                else:
                    train_progress_lines[-1][-1] = train_progress_lines[-2][-1]
                self.super_print('=' * 80)
                train_progress_lines.append(['' for i in range(len(train_progress_lines[-1]))])
                train_progress_lines[-1][0] = str(int(cycle) + 1)
            with open(os.path.join(self.opt.central_path, 'train_progress.csv'), 'w') as f:
                for line in train_progress_lines:
                    f.write(','.join(line) + '\n')
            put_file(self.ssh_client, self.opt.central_path, 'train_progress.csv')
            put_file(self.ssh_client, self.opt.central_path, os.path.basename(self.opt.log))

        else:
            test_progress_lines = [line.strip().split(',') for line in
                                   open(os.path.join(self.opt.central_path, 'test_progress.csv'))]
            test_progress_lines[-1][self.opt.inst_id - 1] = str(len(data_loader) * self.opt.batch_size)
            test_progress_lines[-1][self.opt.num_inst + self.opt.inst_id - 1] = str(self.loss_test)
            test_progress_lines[-1][2 * self.opt.num_inst + self.opt.inst_id - 1] = str(self.acc_test)
            self.super_print('Inst: %s, test loss: %.3f, test acc: %.3f' % (self.opt.inst_id, self.loss_test, self.acc_test))
            if self.opt.inst_id == self.opt.num_inst:
                test_numbers = np.asarray([int(test_progress_lines[-1][i]) for i in range(0, self.opt.num_inst)],
                                          dtype=int)
                test_losses = np.asarray(
                    [float(test_progress_lines[-1][i]) for i in range(self.opt.num_inst, 2 * self.opt.num_inst)],
                    dtype=float)
                test_accs = np.asarray(
                    [float(test_progress_lines[-1][i]) for i in range(2 * self.opt.num_inst, 3 * self.opt.num_inst)],
                    dtype=float)
                n_test_overall = np.sum(test_numbers)
                acc_test_overall = np.sum(test_numbers * test_accs) / n_test_overall
                loss_test_overall = np.sum(test_numbers * test_losses) / n_test_overall
                self.super_print('=' * 80)
                self.super_print(
                    'combined test loss: %.3f, combined test acc: %.3f' % (loss_test_overall, acc_test_overall))
                self.super_print('=' * 80)
            with open(os.path.join(self.opt.central_path, 'test_progress.csv'), 'w') as f:
                for line in test_progress_lines:
                    f.write(','.join(line) + '\n')
            put_file(self.ssh_client, self.opt.central_path, 'test_progress.csv')
            put_file(self.ssh_client, self.opt.central_path, os.path.basename(self.opt.log))










