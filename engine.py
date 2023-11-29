from fileinput import filename
import torch
import util.util as util
import models
import time
import os
import sys
from os.path import join
from util.visualizer import Visualizer
import numpy as np
import json
import glob
from PIL import Image
# attack
from tqdm import tqdm
from attack import utility
from util.util import BatchIter

class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6

        self.__setup()

    def set_learning_rate(self, lr):
        for optimizer in self.model.optimizers:
            # print('[i] set learning rate to {}'.format(lr))
            util.set_opt_param(optimizer, 'lr', lr)
        
    def __setup(self):
        self.basedir = join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        opt = self.opt

        """Model"""
        self.model = models.__dict__[self.opt.model]()
        self.model.initialize(opt)

        if not opt.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer(opt)

    def train(self, train_loader, **kwargs):
        print('\nEpoch: %d' % self.epoch)

        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            iterations = self.iterations
            
            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)
            
            errors = model.get_current_errors()
            avg_meters.update(errors)
            util.progress_bar(i, len(train_loader), str(avg_meters))
            
            if not opt.no_log:
                util.write_loss(self.writer, 'train', avg_meters, iterations)
            
                if iterations % opt.display_freq == 0 and opt.display_id != 0:
                    save_result = iterations % opt.update_html_freq == 0
                    self.visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if iterations % opt.print_freq == 0 and opt.display_id != 0:
                    t = (time.time() - iter_start_time)          

            self.iterations += 1
        self.epoch += 1

        if not self.opt.no_log:
            if self.epoch % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' % (self.epoch, self.iterations))
                model.save()
            
            print('saving the latest model at the end of epoch %d, iters %d' % (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' % (time.time() - epoch_start_time))

    def eval(self, val_loader, dataset_name, ckp, savedir=None, suffix=None,  **kwargs):

        ckp.write_log('Evaluation:\n')
        ckp.add_log(torch.zeros(1,2))
        model = self.model
        opt = self.opt

        timer_test = utility.timer()
        # with torch.no_grad():
        tqdm_test = tqdm(val_loader,ncols=80)
        for i, data in enumerate(tqdm_test):
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
            data_name = data_name[0].split('.')[0]
            torch.cuda.empty_cache()

            # input = input.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            # target_t = target_t.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            # target_r = target_r.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

            input = input.cuda()
            target_t = target_t.cuda()
            target_r = target_r.cuda()

            data = {'input':input,'fn':data_name}
            output_i = model.test(data)
            output_i = torch.clamp(output_i,0,1)

            save_list = [output_i.detach()]
            psnr1 = utility.calculate_psnr(target_t, output_i)
            ssim1 = utility.calculate_ssim(target_t, output_i)
            st1 = "target_t,output_i        | PSNR:"+str(psnr1)+" | SSIM:"+str(ssim1)
            # tqdm.write('------------------------------------------------------------')
            # tqdm.write(st1)
            ckp.write_log('------------------------------------------------------------')
            ckp.write_log(str(data_name))
            ckp.write_log(st1)

            ckp.log[-1,0] += psnr1
            ckp.log[-1,1] += ssim1

            if opt.save_gt:
                save_list.extend([input.detach(), target_t.detach()])

            if opt.save_results:
                ckp.save_results(data_name, save_list)

            del input, target_t, output_i, psnr1, ssim1

        # tqdm.write(str(len(val_loader)))
        ckp.write_log('------------------------------------------------------------')
        ckp.write_log(str(ckp.log))
        ckp.log[-1] = ckp.log[-1]/len(val_loader)
        tqdm.write(dataset_name)
        tqdm.write(str(ckp.log))
        ckp.write_log('------------------------------------------------------------')
        ckp.write_log(str(ckp.log))

        a = ckp.log.detach().numpy()
        np.savetxt(os.path.join(savedir,'result.csv'),a,fmt='%.5f',delimiter=',')
        ckp.write_log('Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True)

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
