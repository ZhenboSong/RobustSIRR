import torch
import math
import random
from PIL import Image
import numpy as np
import scipy.stats as st
import cv2

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


to_tensor = transforms.ToTensor()

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    
    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    # target_size = int(random.randint(224, 448) / 2.) * 2
    target_size = int(random.randint(256, 480) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    # i, j, h, w = get_params(img_1, (224,224))
    i, j, h, w = get_params(img_1, (256,256))
    img_1 = F.crop(img_1, i, j, h, w)
    
    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)
    
    return img_1,img_2

def get_attack_transform(a1_img, a2_img, b_img, crop_size=256):
        # Random Crop
        w, h = a1_img.size
        if w >= 256 and h >= 256:
            pass
        elif w < 256 and h >= 256:
            a1_img = a1_img.resize((256, h),Image.ANTIALIAS)
            a2_img = a2_img.resize((256, h),Image.ANTIALIAS) if a2_img else None
            b_img = b_img.resize((256, h),Image.ANTIALIAS)
            w = 256
        elif w >= 256 and h < 256:
            a1_img = a1_img.resize((w, 256),Image.ANTIALIAS)
            a2_img = a2_img.resize((w, 256),Image.ANTIALIAS) if a2_img else None
            b_img = b_img.resize((w, 256),Image.ANTIALIAS)
            h = 256
        else:
            a1_img = a1_img.resize((256, 256),Image.ANTIALIAS)
            a2_img = a2_img.resize((256, 256),Image.ANTIALIAS) if a2_img else None
            b_img = b_img.resize((256, 256),Image.ANTIALIAS)
            w = 256
            h = 256

        random_w = random.randint(0, w-crop_size)
        random_h = random.randint(0, h-crop_size)

        a1_img = a1_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size))
        a2_img = a2_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size)) if a2_img else None
        b_img = b_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size))

        return F.to_tensor(a1_img), F.to_tensor(a2_img), F.to_tensor(b_img)