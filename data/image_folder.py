###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import json

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, fns=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):                
                    path = os.path.join(root, fname)
                    images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)

    return images

def make_dataset_attack(dir, opt, max_dataset_size=float("inf")):
    # for attack train
    reflection_mask_paths = []
    trans_paths = []
    blended_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    
    json_path = os.path.join(dir,'VOC_results_list.json')

    with open(json_path,'r') as load_f:     
        input_list = json.load(load_f)
    load_f.close()

    reflect_mask_root = os.path.join(dir, 'reflection_mask_layer')
    reflect_root = os.path.join(dir, 'reflection_layer')
    trans_root = os.path.join(dir, 'transmission_layer')
    blend_root = os.path.join(dir, 'blended')


    for index in input_list:
        reflection_mask_paths.append(os.path.join(reflect_mask_root,index['reflection_layer']))
        trans_paths.append(os.path.join(trans_root,index['transmission_layer']))
        blended_paths.append(os.path.join(blend_root,index['blended']))

    return trans_paths[:min(max_dataset_size, len(trans_paths))],reflection_mask_paths[:min(max_dataset_size, len(reflection_mask_paths))],\
    blended_paths[:min(max_dataset_size, len(blended_paths))]

def default_loader(path):
    return Image.open(path).convert('RGB')

