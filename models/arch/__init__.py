# Add your custom network here
from .default import DRNet
from .wo_aid import DRNet_wo_aid
from .wo_aff import DRNet_wo_aff
from .wo_scm import DRNet_wo_scm

import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)

def robustsirr(in_channels, out_channels, base_channels, opt, **kwargs):
    if opt.wo_aid:
        print('W/O AID')
        return DRNet_wo_aid(in_channels, out_channels, base_channels, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)
    if opt.wo_aff:
        print('W/O AFF')
        return DRNet_wo_aff(in_channels, out_channels, base_channels, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)
    if opt.wo_scm:
        print('W/O SCM')
        return DRNet_wo_scm(in_channels, out_channels, base_channels, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)
    return DRNet(in_channels, out_channels, base_channels, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)