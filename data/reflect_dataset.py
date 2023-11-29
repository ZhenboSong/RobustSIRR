import os.path
from os.path import join
from data.image_folder import make_dataset_attack
from data.transforms import to_tensor, paired_data_transforms, get_attack_transform
from PIL import Image
import random
import torch
import math

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import util.util as util
import data.torchdata as torchdata

BaseDataset = torchdata.Dataset

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

class CEILTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None, finetune=False, if_align=False):
        super(CEILTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.finetune = finetune
        self.if_align = if_align

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2):
        h, w = x1.height, x1.width
        h, w = h // 16 * 16, w // 16 * 16
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        return x1, x2

    def __getitem__(self, index):
        fn = self.fns[index]
        if 'Solid' in self.datadir or 'Wild' in self.datadir or 'Post' in self.datadir:
            t_fn = fn
            t_fn = t_fn.replace('m', 'g', 1)

            t_img = Image.open(join(self.datadir, 'transmission_layer', t_fn)).convert('RGB')
            m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
            
        elif 'real' in self.datadir:
            t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
            m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        else:
            t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
            m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        if not self.enable_transforms:
            t_img = F.resize(t_img, 360)
            m_img = F.resize(m_img, 360)
            self.if_align = True

        if self.if_align:
            t_img, m_img = self.align(t_img, m_img)

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        B = to_tensor(t_img)
        M = to_tensor(m_img)

        # dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': M - B,'identity': self.finetune, 'identity_r': False}  # fake reflection gt
        dic = {'input': M, 'target_t': B, 'fn': fn, 'target_r': M - B}  # fake reflection gt

        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)

class VOCDataset(BaseDataset):
    def __init__(self, opt, fns):
        super(VOCDataset, self).__init__()

        self.opt = opt
        self.trans_paths, self.reflect_paths, self.blended_paths = make_dataset_attack(fns, self.opt, self.opt.max_dataset_size)
        self.trans_size = len(self.trans_paths)  # get the size of dataset A1
        self.reflect_size = len(self.reflect_paths)  # get the size of dataset A2
        self.blended_size = len(self.blended_paths)

        assert len(self.trans_paths)==len(self.reflect_paths) ,'trans != reflect'
        assert len(self.trans_paths)==len(self.blended_paths) ,'trans != blended'

    def __getitem__(self, index):

        index = index % len(self.trans_paths)
        trans_path = self.trans_paths[index]
        reflect_path = self.reflect_paths[index]
        blended_path = self.blended_paths[index]
        
        trans_img = Image.open(trans_path).convert('RGB')
        reflect_img = Image.open(reflect_path).convert('RGB')
        blended_img = Image.open(blended_path).convert('RGB')

        T, R, B = get_attack_transform(trans_img, reflect_img, blended_img)

        fn = os.path.basename(trans_path)

        return {'input': B, 'target_t': T, 'target_r': R, 'fn': fn}

    def __len__(self):
        length = min(max(self.trans_size, self.reflect_size, self.blended_size),self.opt.max_dataset_size)
        return length

class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1./len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' %(self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio/residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index%len(dataset)]
            residual -= ratio
    
    def __len__(self):
        return self.size