from os.path import join
from options.robustsirr.train_options import TrainOptions
from engine import Engine
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
torch.set_num_threads(5)

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# Path to dataset
datadir = './datasets/'

datadir_syn = join(datadir, 'VOC2012/')
datadir_real = join(datadir, 'real89')

train_dataset_voc = datasets.VOCDataset(opt, datadir_syn)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_fusion = datasets.FusionDataset([train_dataset_voc, train_dataset_real], [0.7, 0.3])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=True, 
    num_workers=opt.nThreads, pin_memory=True, drop_last=True)

"""Main Loop"""
engine = Engine(opt)
result_dir = os.path.join(f'./checkpoints/{opt.name}/results', util.get_formatted_time())

def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

engine.train(train_dataloader_fusion)
