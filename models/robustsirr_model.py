import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import itertools
from collections import OrderedDict
from models.utils import pdist_js

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join
from tqdm import tqdm

from .base import *


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor=image_tensor[0].data.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    # image_numpy = image_numpy.astype(imtype)
    return image_tensor


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        
        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class RobustSIRRModelBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
            # tqdm.write('input : '+str(input.device))
            # input = input.to('cuda:1')
            # tqdm.write('input : '+str(input.device))
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])                
        
        self.input = input
        
        # self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.data_name = data_name

        self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True
        
        # if target_t is not None:            
        #     self.target_edge = self.edge_map(self.target_t)         
            
    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            output_i = self.forward()

            output_i = tensor2im(output_i)
            target = tensor2im(self.target_t)

            if self.aligned:
                res = index.quality_assess(output_i, target)
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))
                    if suffix is not None:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'{}_{}.png'.format(self.opt.name, suffix)))
                    else:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 'transmission_layer', str(self._count)+'.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'blended', str(self._count)+'.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

#         if self.data_name is not None and savedir is not None:
#             name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
#             if not os.path.exists(join(savedir, name)):
#                 os.makedirs(join(savedir, name))

#             if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
#                 return 
        
        # with torch.no_grad():
        output_i = self.forward()
        output_i = output_i.to('cuda:0')
        return output_i

class RobustSIRRModel(RobustSIRRModelBase):
    def name(self):
        return 'robustsirr'
        
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")/

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        #if not opt.clean :
        #    self.adv_label = torch.LongTensor(([0]*(opt.batchSize//2))+([1]*(opt.batchSize//2))).to(self.device)
        #else :
        self.adv_label = torch.LongTensor(([0]*(opt.batchSize//2))+([1]*(opt.batchSize//2))).to(self.device)
        in_channels = 3
        base_channels = 48
        self.vgg = None
        
        # if opt.hyper:
        #     self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
        #     in_channels += 1472
        
        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3, base_channels, opt)
        networks.init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        # self.edge_map = EdgeMap(scale=1).to(self.device)

        
        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss
            
            if not opt.wo_aid:
                # attack loss
                self.loss_dic['tri'] = losses.OnlineTripletLoss(0.6, pdist=pdist_js)

            cxloss = losses.ContentLoss()

            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(), 
                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None
        self.loss_Tri = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
        
        if self.aligned:
            # print(self.output_i.shape)
            # print(self.target_t.shape)
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
        else:
            
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        if not self.opt.wo_aid:
            # attack train
            self.loss_Tri, _ = self.loss_dic['tri'].get_loss(self.pred_adv,self.adv_label)
            self.loss_G += self.loss_Tri*self.opt.lambda_tri
        self.loss_G.backward()

    def forward(self):
        input_i = self.input
        
        pred_adv = self.net_i.disc(self.input)
        self.pred_adv = pred_adv
        
        output_i = self.net_i(input_i, pred_adv)
        self.output_i = output_i.to('cuda:0')

        return output_i
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def zero_grad(self):
        self.optimizer_G.zero_grad()
        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
            
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        if self.loss_Tri is not None:
            ret_errors['Tri'] = self.loss_Tri.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals       

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if icnn_path is None:
            print(model.pre_dir)
            print(model.name())
            model_path = util.get_model_list(model.pre_dir, model.name(), epoch=resume_epoch)
            print('Load ',model_path)
            state_dict = torch.load(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model.net_i.load_state_dict(state_dict['icnn'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = torch.load(icnn_path)
            model.net_i.load_state_dict(state_dict['icnn'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            # if model.isTrain:
            #     model.optimizer_G.load_state_dict(state_dict['opt_g'])

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])
            
        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict