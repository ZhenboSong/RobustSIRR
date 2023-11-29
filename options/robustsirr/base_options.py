from options.base_option import BaseOptions as Base
from util import util
import os
import torch
import numpy as np
import random

class BaseOptions(Base):
    def initialize(self):
        Base.initialize(self)
        # experiment specifics
        self.parser.add_argument('--inet', type=str, default='robustsirr', help='chooses which architecture to use for inet.')
        self.parser.add_argument('--icnn_path', type=str, default=None, help='icnn checkpoint to use.')
        self.parser.add_argument('--init_type', type=str, default='edsr', help='network initialization [normal|xavier|kaiming|orthogonal|uniform]')
        # for network
        self.parser.add_argument('--hyper', action='store_true', help='if true, augment input with vgg hypercolumn feature')
        
        self.parser.add_argument('--save_root', default='results', type=str, help='results root')
        
        # for attack
        self.parser.add_argument('--crop_size', default='256', type=int, help='then crop to this size')
        self.parser.add_argument('--phase', default='train', type=str, help='train or test')
        # robust
        self.parser.add_argument('--data_test', default='nature', type=str, choices=['nature', 'wild', 'postcard','solidobject','real20'])
        self.parser.add_argument('--save_attack_dir', default='.', type=str)
        self.parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'ifgsm', 'free', 'none'])
        self.parser.add_argument('--attack_iters', default=20, type=int)
        self.parser.add_argument('--robust_epsilon', default=8, type=float)
        self.parser.add_argument('--robust_alpha', default=2, type=float)
        self.parser.add_argument('--restarts', default=1, type=int)
        self.parser.add_argument('--save_attack', action='store_true',help='save attacked images together')
        self.parser.add_argument('--target', default='output', type=str, choices=['output', 'input','down_stream','down_stream_v2','residual'])
        self.parser.add_argument('--attack_loss', default='l_2', type=str, choices=['l_2', 'lpips'])
        self.parser.add_argument('--mse_weight', default=25, type=int, help='mse weight for down_stream attack')
        self.parser.add_argument('--save_results', action='store_true',help='save output results')
        self.parser.add_argument('--attack_gt', action='store_true',help='attack gt')
        self.parser.add_argument('--save_gt', action='store_true',help='save gt')
        self.parser.add_argument('--rgb_range', type=int, default=1,help='maximum value of RGB')
        self.parser.add_argument('--attack_reflection_region', action='store_true', help='attack_reflection_region')
        self.parser.add_argument('--attack_non_reflection_region', action='store_true', help='attack_non_reflection_region')
        # robust train
        self.parser.add_argument('--start_iter', default=0, type=int, help='default 0, for resume training')
        self.parser.add_argument('--max_iter', default=700000, type=int)

        self.parser.add_argument('--mask_threshold', default=0.2, type=float, help='mask threshold')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed) # seed for every module
        random.seed(self.opt.seed)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        # TODO:
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.opt.name = self.opt.name or '_'.join([self.opt.model])
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        if self.opt.debug:
            self.opt.display_freq = 20
            self.opt.print_freq = 20
            self.opt.nEpochs = 40
            self.opt.max_dataset_size = 100
            self.opt.no_log = False
            self.opt.nThreads = 0
            self.opt.decay_iter = 0
            self.opt.serial_batches = True
            self.opt.no_flip = True
        
        return self.opt
