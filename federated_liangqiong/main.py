
import paramiko
import argparse
from utils.start_config import initization_configure
from utils.CWT_models import DistrSystem

parser = argparse.ArgumentParser()
## PART ONE PARAMETERS:  CENTRAL PARAMETER SERVER INFO
parser = argparse.ArgumentParser(description='PyTorch CWT')
parser.add_argument('--central_server', type=str, default='epad-public.stanford.edu', help='central parameter server')
parser.add_argument('--username', type=str, default='distributed', help='central parameter server username')
parser.add_argument('--password', type=str, default='Ch@ngeMe', help='central parameter server username')
parser.add_argument('--central_path', type=str, default='ADNI_experiment', help='path to central parameter directory (/path/to/param_dir/')

## PART TWO PARAMETERS:
# LOCAL CONFIG
parser.add_argument('--num_inst', type=int, default=2, help='number of participating training institutions')
parser.add_argument('--inst_id', type=int, default= 1, help='order of your institution among the num_inst institutions: int in the range [1,num_inst]\
	Must be different for each training institution')
parser.add_argument('--data_path', type=str,default='Data', )
parser.add_argument('--gpu_ids', type=str,default=0, help = 'used gpu ids' )

## PART THREE: TRAIN PARAMETERS THE SAME FOR ALL THE INSTS
parser.add_argument('--dis_model_name', type=str, default='CWT_ADNI_ResNet18', help='model name')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--max_cycles', type=int,default=100, help = 'maximum epochs for train' )
parser.add_argument('--SEED', type=int, default=666, help='Random seed, should be the same for all insts. ')
parser.add_argument('--lr', type=float, default=0.0001,  help='Learning Rate. Default=0.0001')
parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size')
parser.add_argument('--fine_size', type=int, default=224, help='Then crop to this size')
parser.add_argument('--val_freq', type=int, default=1, help='frequncy for CWT transfer')
parser.add_argument('--num_classes', type=int, default=2, help='num of classes')
parser.add_argument('--num_workers', type=int, default=4, help='threads for loading data')
parser.add_argument('--phase', type=str, default='train', help='train or test phase')
parser.add_argument('--val', action='store_true', default=True, help='include to validate during training')
parser.add_argument('--continue_train', action='store_true', default=False, help='Continue train from central saved model')
parser.add_argument('--model_architecture', type=str, default='ResNet18', help='Type of model to use: one of ResNet34, DesNet121, inception_v3, squeezenet1_0')
parser.add_argument('--regression', action='store_true', default = False,  help='Indicating using regression model or not')
parser.add_argument('--sleep_time', type=int, default=60, help='time in seconds to pause between github pull requests')

# FOLLOWING ARGUMENTS MUST BE THE SAME FOR ALL PARTICIPATING INSTITUTIONS DURING TRAINING

# SELECT TRUE FOR ONE OF THE FOLLOWING TO ADDRESS LABEL IMBALANCE


if __name__ == '__main__':
    opt = parser.parse_args()
    # step 1 : connecting to central server

    ## step 1: ssh connection with central server

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    while True:
        try:
            ssh_client.connect(hostname=opt.central_server, username=opt.username,
                               password=opt.password)
            break
        except:
            print('Wrong password for', opt.username, opt.central_server)

    ## step 2: initization dataset and device setting
    initization_configure(opt, ssh_client)

    ## step 3: start train or test
    classifier = DistrSystem(opt, ssh_client)
    if opt.phase == 'train':
        classifier.train(opt.train_set_loader)
    else:
        classifier.train(opt.test_set_loader)



