import torch, torchvision
import mmseg
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import mmcv
import pickle
import pandas as pd
import argparse

from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm
from mmseg.apis import set_random_seed
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS
from mmcv import Config
from sklearn.model_selection import train_test_split

from model_runner import ModelRunner

opj = lambda *args: os.path.join(*args)

def main(args):
    cfg = __import__(name = f'configs.{args.model}', fromlist=['Config'])
    # print(cfg.Config().cfg)
    model = ModelRunner(cfg.Config(), args.device, [args.gpu_id])
    model.load_dataset()
    if args.mode == 'train':
        model.train(resume=True)
    if args.mode == 'test':
        model.infer(args.test_folder, save_dir=args.save_dir, img_suffix=args.img_suffix, video=True)
    
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='upernet_convnextb', type=str, help='python file name in the configs folder', required=False)
    parser.add_argument('--mode', default='train', type=str, help='train | test', required=False)
    parser.add_argument('--device', default='cuda', type=str, help='cuda | cpu', required=False)
    parser.add_argument('--gpu_id', default=0, type=int, help='the cuda device', required=False)
    parser.add_argument('--test_folder', default='/common/users/dm1487/segmentation_data/open_bag/cam-00', type=str, help='folder where the image is', required=False)
    parser.add_argument('--img_suffix', default='.color.png', type=str, help='what images to select based on the suffix, for example, .png', required=False)
    parser.add_argument('--save_dir', default='', type=str, help='where the output will be saved', required=False)
    # parser.add_argument('resume', default=False, type=str, help='resume training from latest checkpoint')
    args = parser.parse_args()
    main(args)
