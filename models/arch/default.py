# Define network components here
from operator import xor
import torch
from torch import nn
import torch.nn.functional as F
from models.base import discriminator
from models.networks import SCM, DynamicConv2d, DynamicConvTranspose2d, AFF, BasicConv, TransformerBlock, Downsample, Upsample, FAM
from copy import deepcopy

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        
     
ssd_adv_pred = {}

class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, n_resblocks, norm=nn.BatchNorm2d, 
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False):
        super(DRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cuda:0")
        
        # attack train
        self.disc = discriminator().to(self.device)

        num_blocks = [1,2,2,4]
        # num_blocks = [4,6,6,8]
        num_refinement_blocks = 4
        heads = [1,2,4,8]
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        
        # self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        # self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        # self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        self.feat_extract = nn.Sequential(
            DynamicConvLayer(conv, in_channels, base_channels, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act),
            DynamicConvLayer(conv, base_channels, base_channels, kernel_size=3, stride=1, norm=norm, act=act),
            DynamicConvLayer(conv, base_channels, base_channels, kernel_size=3, stride=1, norm=norm, act=act)
        ).to(self.device)

        self.SCM1 = SCM(base_channels * 4).to(self.device)
        self.SCM2 = SCM(base_channels * 2).to(self.device)
        self.FAM1 = FAM(base_channels * 4).to(self.device)
        self.FAM2 = FAM(base_channels * 2).to(self.device)

        self.AFFs = nn.ModuleList([
            AFF(base_channels * 7, base_channels * 1),
            AFF(base_channels * 7, base_channels * 2)
        ]).to(self.device)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=base_channels, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])]).to(self.device)
        
        self.down1_2 = Downsample(base_channels).to(self.device)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(base_channels*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]).to(self.device)

        self.down2_3 = Downsample(int(base_channels*2**1)).to(self.device)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(base_channels*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])]).to(self.device)

        self.down3_4 = Downsample(int(base_channels*2**2)).to(self.device)
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(base_channels*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])]).to(self.device)
        
        self.up4_3 = Upsample(int(base_channels*2**3)).to(self.device1)
        self.reduce_chan_level3 = nn.Conv2d(int(base_channels*2**3), int(base_channels*2**2), kernel_size=1, bias=bias).to(self.device1)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(base_channels*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])]).to(self.device1)

        self.up3_2 = Upsample(int(base_channels*2**2)).to(self.device1)
        self.reduce_chan_level2 = nn.Conv2d(int(base_channels*2**2), int(base_channels*2**1), kernel_size=1, bias=bias).to(self.device1)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(base_channels*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]).to(self.device1)
        
        self.up2_1 = Upsample(int(base_channels*2**1)).to(self.device1)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(base_channels*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])]).to(self.device1)

        self.spatial_feat_extract = nn.Sequential(
            DynamicConvLayer(conv, base_channels * 2, base_channels, kernel_size=3, stride=1, norm=norm, act=act),
            DynamicConvLayer(conv, base_channels, base_channels, kernel_size=3, stride=1, norm=norm, act=act),
            PyramidPooling(base_channels, base_channels, scales=(4,8,16,32), ct_channels=base_channels//4),
            DynamicConvLayer(conv, base_channels, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        ).to(self.device1)

        #input dynamic convolution weights to each DynamicConv2d layer via hook
        for layer in self.modules():
            if isinstance(layer, DynamicConv2d) or isinstance(layer, DynamicConvTranspose2d):
                layer.register_forward_pre_hook(lambda module, x:(x[0], ssd_adv_pred[x[0].device]))
    
    @property
    def adv_pred(self):
        return ssd_adv_pred[self.conv1.device]
        
    def forward(self, x, adv_pred):
        
        # adv_pred=adv_pred if adv_pred is not None else self.disc(x) 
        ssd_adv_pred[adv_pred.device] = adv_pred #AID预测动态卷积权重
        adv_pred1 = adv_pred.to(self.device1)
        ssd_adv_pred[adv_pred1.device] = adv_pred1
        
        x_2 = F.interpolate(x, scale_factor=0.5)
        # print('x_2.shape',x_2.shape)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        # encoder

        inp_enc_level1 = self.feat_extract(x)
        # print('inp_enc_level1.shape',inp_enc_level1.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # print('out_enc_level1.shape',out_enc_level1.shape)
        inp_fam2 = self.down1_2(out_enc_level1)
        # print('inpfam2.shape',inp_fam2.shape)
        # print('z2.shape',z2.shape)
        inp_enc_level2 = self.FAM2(inp_fam2, z2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_fam1 = self.down2_3(out_enc_level2)
        inp_enc_level3 = self.FAM1(inp_fam1, z4)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # AFF
        out_enc_level3_x2 = F.interpolate(out_enc_level3, scale_factor=2)
        out_enc_level2_x2 = F.interpolate(out_enc_level2, scale_factor=2)
        out_enc_level3_x4 = F.interpolate(out_enc_level3_x2, scale_factor=2)
        
        out_af_0 = self.AFFs[0](out_enc_level1, out_enc_level2_x2, out_enc_level3_x4)

        out_enc_level1_x05 = F.interpolate(out_enc_level1, scale_factor=0.5)
        out_af_1 = self.AFFs[1](out_enc_level1_x05, out_enc_level2, out_enc_level3_x2)
        
        # cuda 1
        latent = latent.to(self.device1)
        out_enc_level3 = out_enc_level3.to(self.device1)
        #######
        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        
        # cuda 1
        out_af_1 = out_af_1.to(self.device1)
        out_af_0 = out_af_0.to(self.device1)
        #######

        # decoder
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = torch.cat([inp_dec_level2, out_af_1], 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_af_0], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # spatial_feat_extract
        outputs = self.spatial_feat_extract(out_dec_level1)
        
        return outputs


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))

        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)

class DynamicConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(DynamicConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('dynamic',DynamicConv2d(in_channels,out_channels,kernel_size,stride,padding,dilation=dilation))
        # self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))

        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)

class DynamicConvTranspose2dLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(DynamicConvTranspose2dLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('dynamictrans',DynamicConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,dilation=dilation))
        # self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))

        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)