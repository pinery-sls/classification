import argparse

from yaml import parse

parser = argparse.ArgumentParser(description="cls")

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. Default:[0]',
)

parser.add_argument(
    '--dataset',
    type=str,
    default='ImageNet',
    help='Select dataset to train. default:ImageNet',
)

parser.add_argument(
    '--data_path',
    type=str,
    default='/data/cifar10/',
    help='The dictionary where the input is stored. default:/data/cifar10/',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored.',
)

parser.add_argument(
    '--arch',
    type=str,
    default='resnet',
    help='Architecture of model. default:resnet'
)

parser.add_argument(
    '--cfg',
    type=str,
    default='resnet56',
    help='Detail architecuture of model.'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=150,
    help='The num of epochs to train. Default:150'
)

parser.add_argument(
    '--resume',
    type=str,
    default=None,
    help='Continue training from last epoch, keep all training configurations as before.'
)

parser.add_argument('-p', '--print_freq', default=20, type=int,
                    metavar='N', help='print frequency (default:20)')

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size for training. default:256'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=128,
    help='Batch size for validataion. default:128'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9'
)


parser.add_argument(
    '--lr',
    type=float,
    default=1e-2,
    help='Learning rate for train. default:1e-2'
)

parser.add_argument(
    '--lr_type',
    default='step', 
    type=str, 
    help='lr scheduler (step/exp/cos/step3/fixed)'
)

parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

parser.add_argument(
    '--criterion',
    default='Softmax', 
    type=str, 
    help='Loss func (Softmax/SmoothSoftmax)'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+',
    default=[50, 100],
    help='the iterval of learn rate. default:50, 100'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-3,
    help='The weight decay of loss. default:5e-3'
)

parser.add_argument(
    '--pretrain_model',
    type=str,
    default=None,
    help='Path to the pretrain model . default:None'
)


args = parser.parse_args()