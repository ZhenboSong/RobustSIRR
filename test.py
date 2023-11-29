from os.path import join, basename
from attack import utility
from options.robustsirr.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
opt = TrainOptions().parse()
torch.set_num_threads(5)

opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True
opt.display_id = -1
opt.verbose = False
opt.phase = 'test'
opt.nThreads = 0
opt.if_align = True

# Path to dataset
datadir = './datasets/'

eval_dataset_real = datasets.CEILTestDataset(join(datadir, 'real20'), fns=read_fns(join(datadir, 'real20/real_test.txt')), if_align=opt.if_align)
eval_dataset_solidobject = datasets.CEILTestDataset(join(datadir, 'SIR2/SolidObjectDataset'), if_align=opt.if_align)
eval_dataset_postcard = datasets.CEILTestDataset(join(datadir, 'SIR2/PostcardDataset'), if_align=opt.if_align)
eval_dataset_wild = datasets.CEILTestDataset(join(datadir, 'SIR2/WildSceneDataset'), if_align=opt.if_align)
eval_dataset_nature = datasets.CEILTestDataset(join(datadir, 'nature20'), if_align=opt.if_align)

eval_dataloader_real = datasets.DataLoader(eval_dataset_real, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_solidobject = datasets.DataLoader(eval_dataset_solidobject, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_postcard = datasets.DataLoader(eval_dataset_postcard, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_wild = datasets.DataLoader(eval_dataset_wild, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_nature = datasets.DataLoader(eval_dataset_nature, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

engine = Engine(opt)

result_dir = join(opt.save_root, opt.name, util.get_formatted_time())
print(result_dir)

savedir = join(result_dir, 'real')
ckp_real = utility.checkpoint(opt,savedir)
engine.eval(eval_dataloader_real, dataset_name='real', ckp=ckp_real, savedir=savedir)

savedir = join(result_dir, 'solidobject')
ckp_solidobject = utility.checkpoint(opt,savedir)
engine.eval(eval_dataloader_solidobject, dataset_name='solidobject', ckp=ckp_solidobject, savedir=savedir)

savedir = join(result_dir, 'postcard')
ckp_postcard = utility.checkpoint(opt, savedir)
engine.eval(eval_dataloader_postcard, dataset_name='postcard', ckp=ckp_postcard, savedir=savedir)

savedir = join(result_dir, 'wild')
ckp_wild = utility.checkpoint(opt,savedir)
engine.eval(eval_dataloader_wild, dataset_name='wild', ckp=ckp_wild, savedir=savedir)

savedir = join(result_dir, 'nature')
ckp_nature = utility.checkpoint(opt,savedir)
engine.eval(eval_dataloader_nature, dataset_name='nature', ckp=ckp_nature, savedir=savedir)