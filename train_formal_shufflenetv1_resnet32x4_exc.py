"""
the general training framework
"""

from __future__ import print_function

import os
import re
import argparse
import time
import random

import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
# import tensorboard_logger as tb_logger

# import apex

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed, SelfA

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
# from dataset.imagenet import get_imagenet_dataloader, imagenet_list
# from dataset.imagenet_dali import get_dali_data_loader

from helper.util import adjust_learning_rate, save_dict_to_json, reduce_tensor, Logger

from distiller_zoo import DistillKL, RKDLoss, SemCKDLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate

split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    # basic
    parser.add_argument('--print-freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--exc_epoch', type=int, default=10,help='exchange parmater epoch')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8x4',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'ResNet18', 'ResNet34', 'resnet8x4_double',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_50_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg11_imagenet', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path-t', type=str, default='./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth', help='teacher model snapshot')

    # distillation
    parser.add_argument('--mindex', type=str, default='1', help='Method Index')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'rkd', 'crd', 'semckd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])

    # transform layers for IRG
    parser.add_argument('--transform_layer_t', nargs='+', type=int, default = [])
    parser.add_argument('--transform_layer_s', nargs='+', type=int, default = [])

    # switch for edge transformation
    parser.add_argument('--no_edge_transform', action='store_true') # default=false
    
    parser.add_argument('--use-lmdb', action='store_true') # default=false

    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)

    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')

    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    parser.add_argument('--hkd_initial_weight', default=100, type=float, help='Initial layer weight for HKD method')
    parser.add_argument('--hkd_decay', default=0.7, type=float, help='Layer weight decay for HKD method')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/student_model_exc/formal_training/ShuffleV1_resnet32x4/'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_{}_{}_r', '{}_a', '{}_b', '{}'])
    opt.model_name = model_name_template.format(opt.model_s, opt.model_t, opt.dataset, opt.trial, opt.mindex, opt.distill,
                                                opt.gamma, opt.alpha, opt.beta)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    # save_txt = os.path.join(opt.save_folder, "log.txt")
    # logger = Logger(args=opt, filename = save_txt)
    
    print(opt)
    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    if segments[0] == 'resnext50':
        return segments[0] + '_' + segments[1]
    if segments[0] == 'vgg13' and segments[1] == 'imagenet':
        return segments[0] + '_' + segments[1]
    return segments[0]

def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    # TODO: reduce size of the teacher saved in train_teacher.py
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model

total_time = time.time()
best_acc1 = 0
best_acc2 = 0

def main():
    
    opt = parse_option()
    
    
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc1, best_acc2, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    save_txt = os.path.join(opt.save_folder, "log.txt")
    logger = Logger(args=opt, filename = save_txt)
# region Dataset + Model
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)

    class_num_map = {
        'cifar100': 100,
    }
    if opt.dataset not in class_num_map:
        raise NotImplementedError(opt.dataset)
    n_cls = class_num_map[opt.dataset]

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    module_args = {'num_classes': n_cls}
    model_s1 = model_dict[opt.model_s](**module_args)
    model_s2 = model_dict[opt.model_s](**module_args)

    
    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)

    model_t.eval()
    model_s1.eval()
    model_s2.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s1, _ = model_s1(data, is_feat=True)
    feat_s2, _ = model_s2(data, is_feat=True)

    module_list1 = nn.ModuleList([])
    module_list1.append(model_s1)
    trainable_list1 = nn.ModuleList([])
    trainable_list1.append(model_s1)
    module_list2 = nn.ModuleList([])
    module_list2.append(model_s2)
    trainable_list2 = nn.ModuleList([])
    trainable_list2.append(model_s2)
# endregion

# region Distill
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'semckd':
        s1_n = [f.shape[1] for f in feat_s1[1:-1]]
        s2_n = [f.shape[1] for f in feat_s2[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = SemCKDLoss()
        self_attention1 = SelfA(len(feat_s1)-2, len(feat_t)-2, opt.batch_size, s1_n, t_n)
        self_attention2 = SelfA(len(feat_s2)-2, len(feat_t)-2, opt.batch_size, s2_n, t_n)
        module_list1.append(self_attention1)
        trainable_list1.append(self_attention1)
        module_list2.append(self_attention2)
        trainable_list2.append(self_attention2)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s1[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = 50000
        criterion_kd = CRDLoss(opt)
        module_list1.append(criterion_kd.embed_s)
        module_list1.append(criterion_kd.embed_t)
        trainable_list1.append(criterion_kd.embed_s)
        trainable_list1.append(criterion_kd.embed_t)
        module_list2.append(criterion_kd.embed_s)
        module_list2.append(criterion_kd.embed_t)
        trainable_list2.append(criterion_kd.embed_s)
        trainable_list2.append(criterion_kd.embed_t)
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    module_list1.append(model_t)
    module_list2.append(model_t)
# endregion

# region Torch + Optimizer + Dataloater + Test_teacher 
    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list1.cuda(opt.gpu)
                module_list2.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list1:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                for module in module_list2:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list1 = distributed_modules
                module_list2 = distributed_modules

                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list1.cuda()
            module_list2.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    optimizer1 = optim.SGD(trainable_list1.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    optimizer2 = optim.SGD(trainable_list2.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)


    if not opt.skip_validation:
        # validate teacher accuracy
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)

        if opt.dali is not None:
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')
    # endregion
    
    # routine
    Exc_index = 0


    exc_epoch =  opt.exc_epoch   
    row1 = {
            '训练描述： [1,2,3] 150 [2,3] 240   layer1*2*3 的 weight, Exc_epoches为':str(exc_epoch)
            }      
    
    print(row1)
    logger.writerow(row1)


    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)

        now_lr = adjust_learning_rate(epoch, opt, optimizer1)
        adjust_learning_rate(epoch, opt, optimizer2)
        if epoch % exc_epoch == 0:
            if epoch <= 150:
                random_number = random.randint(1, 3)
            elif epoch > 150:
                random_number = random.randint(2, 3)
            
            if random_number == 1:
                row1 = {
                    'change parameter 选择交换层数为':str(random_number)
                }
                print(row1)
                logger.writerow(row1)
                layer_name = "conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.0.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.0.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer1.0.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.0.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer1.0.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.0.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer1.1.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.1.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer1.1.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.1.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer1.1.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.1.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer1.2.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.2.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer1.2.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.2.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer1.2.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.2.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer1.3.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.3.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer1.3.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.3.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer1.3.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer1.3.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############

            if random_number == 2:
                row1 = {
                    'change parameter 选择交换层数为':str(random_number)
                }
                print(row1)
                logger.writerow(row1)
                layer_name = "layer2.0.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.0.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.0.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.0.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.0.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.0.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer2.1.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.1.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.1.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.1.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.1.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.1.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer2.2.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.2.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.2.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.2.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.2.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.2.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer2.3.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.3.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.3.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.3.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.3.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.3.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer2.4.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.4.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.4.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.4.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.4.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.4.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer2.5.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.5.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.5.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.5.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.5.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.5.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer2.6.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.6.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.6.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.6.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.6.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.6.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer2.7.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.7.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.7.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.7.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer2.7.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer2.7.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                

            if random_number == 3:
                row1 = {
                    'change parameter 选择交换层数为':str(random_number)
                }
                print(row1)
                logger.writerow(row1)
                layer_name = "layer3.0.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.0.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer3.0.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.0.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer3.0.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.0.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer3.1.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.1.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer3.1.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.1.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer3.1.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.1.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer3.2.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.2.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer3.2.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.2.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer3.2.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.2.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "layer3.3.conv1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.3.bn1.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer3.3.conv2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.3.bn2.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ################
                layer_name = "layer3.3.conv3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                layer_name = "layer3.3.bn3.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
                ###############
                layer_name = "linear.weight"
                weights1 = model_s1.state_dict()[layer_name].clone()
                weights2 = model_s2.state_dict()[layer_name].clone()
                model_s1.state_dict()[layer_name].copy_(weights2)
                model_s2.state_dict()[layer_name].copy_(weights1)
            

        time1 = time.time()
        train_acc1, train_acc1_top5, train_loss1, data_time1 = train(epoch, train_loader, module_list1, criterion_list, optimizer1, opt)
        train_acc2, train_acc2_top5, train_loss2, data_time2 = train(epoch, train_loader, module_list2, criterion_list, optimizer2, opt)

        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc1, train_acc1_top5, train_loss1, data_time1]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss, data_time = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}, Data {:.2f}'.format(epoch, opt.gpu, train_acc1, train_acc1_top5, time2 - time1, data_time1))
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}, Data {:.2f}'.format(epoch, opt.gpu, train_acc2, train_acc2_top5, time2 - time1, data_time2))

            

        # print('GPU %d validating' % (opt.gpu))
        test_acc1, test_acc1_top5, test_loss1 = validate(val_loader, model_s1, criterion_cls, opt)        
        test_acc2, test_acc2_top5, test_loss2 = validate(val_loader, model_s2, criterion_cls, opt)        


        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc1, test_acc1_top5))
            row = { 'Epoch': str(epoch), 
            'TrainAcc@1': '%.3f'%(train_acc1), 
            'TrainAcc@5': '%.3f'%(train_acc1_top5), 
            'TestAcc@1': '%.3f'%(test_acc1), 
            'TestAcc@5': '%.3f'%(test_acc1_top5), 
            'Time': '%.2f'%(time2 - time1), 
            'lr': '%.5f'%(now_lr),
            'Data': '%.2f'%(data_time1),
            }
            print(row)
            logger.writerow(row)
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc2, test_acc2_top5))
            row = { 'Epoch': str(epoch), 
            'TrainAcc@1': '%.3f'%(train_acc2), 
            'TrainAcc@5': '%.3f'%(train_acc2_top5), 
            'TestAcc@1': '%.3f'%(test_acc2), 
            'TestAcc@5': '%.3f'%(test_acc2_top5), 
            'Time': '%.2f'%(time2 - time1), 
            'lr': '%.5f'%(now_lr),
            'Data': '%.2f'%(data_time2),
            }
            print(row)
            logger.writerow(row)

            # save the best model
            if test_acc1 > best_acc1:
                best_acc1 = test_acc1
                state = {
                    'epoch': epoch,
                    'model': model_s1.state_dict(),
                    'best_acc': best_acc1,
                }
                if opt.distill == 'semckd':
                    state['attention'] = trainable_list1[-1].state_dict()
                save_file = os.path.join(opt.save_folder, '{}_s1_best.pth'.format(opt.model_s))
                
                test_merics = {'test_loss': test_loss1,
                                'test_acc': test_acc1,
                                'test_acc_top5': test_acc1_top5,
                                'epoch': epoch}
                
                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_s1_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)
            if test_acc2 > best_acc2:
                best_acc2 = test_acc2
                state = {
                    'epoch': epoch,
                    'model': model_s2.state_dict(),
                    'best_acc': best_acc2,
                }
                if opt.distill == 'semckd':
                    state['attention'] = trainable_list2[-1].state_dict()
                save_file = os.path.join(opt.save_folder, '{}_s2_best.pth'.format(opt.model_s))
                
                test_merics = {'test_loss': test_loss2,
                                'test_acc': test_acc2,
                                'test_acc_top5': test_acc2_top5,
                                'epoch': epoch}
                
                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_s2_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)
            if epoch == 240:
                row = { 
                'BestAcc@s1': '%.3f'%(best_acc1), 
                'BestAcc@s2': '%.3f'%(best_acc2), 
                }
                print(row)
                logger.writerow(row)
                

if __name__ == '__main__':
    main()
