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
import glob
import cv2

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from mmseg.apis import set_random_seed
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS
from mmcv import Config
from sklearn.model_selection import train_test_split
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from utils import *

class ModelRunner():
    def __init__(self, cfg, device='cuda', gpu_ids=[0], seed=42):
        self.cfg = None
        self.model_meta = getattr(cfg, 'model_meta')
        self._prepare_cfg(getattr(cfg, 'cfg', None), device, gpu_ids, seed)
        self.model = None
        mmcv.mkdir_or_exist(os.path.abspath(self.cfg.work_dir))
        self.CLASSES = cfg.model_meta.classes
        self.PALETTE = cfg.model_meta.palette
        
        
    def _prepare_cfg(self, cfg, device, gpu_ids, seed):
        self.cfg = Config.fromfile(self.model_meta.model_config)
        print(self.cfg.data['train'])
        self._overwrite_cfg(self.cfg, class_to_dict(cfg))
        self._add_params(self.cfg, class_to_dict(cfg))
        print(self.cfg.data['train'])
        self.cfg.device = device
        self.cfg.gpu_ids = gpu_ids
        self.cfg.workflow = [('train', 1)]
        self.cfg.seed = seed
        set_random_seed(seed, deterministic=False)
        
    def _overwrite_cfg(self, cfg, cfg_ov):
        '''
        overwriting and additng config with parameters as set in the config file for the model
        '''
        for key in cfg.keys():
            c_keys = list(cfg_ov.keys())
            # c_keys = [c for c in dir(cfg_ov) if not c.startswith('__')]
            if key in c_keys:
                var = getattr(cfg, key)
                var_ov = cfg_ov[key]
                if type(var) == mmcv.utils.config.ConfigDict:
                    self._overwrite_cfg(var, var_ov)
                else:
                    setattr(cfg, key, var_ov)
                    print('setting', key, 'to', var_ov)
                    
    def _add_params(self, cfg, cfg_ov):
        for key in list(cfg_ov.keys()):
            var_new = cfg_ov[key]
            var = getattr(cfg, key, None)
            
            if type(var_new) is dict:
                if var is None:
                    try:
                        setattr(cfg, key, var_new)
                    except:
                        cfg[key] = var_new
                    print('adding', key, var_new)
                    var = getattr(cfg, key)
                self._add_params(var, var_new)
            else:
                if var is None:
                    try:
                        setattr(cfg, key, var_new)
                    except:
                        cfg[key] = var_new
                    print('adding', key, var_new)
            
    
    def load_dataset(self):
        '''
        loading the train-val-test dataset for segmentation
        '''
        self.datasets = [build_dataset(self.cfg.data.train)]
    
    def train(self, resume=False):
        '''
        train the model based on the runner cfg
        '''
        if resume:
            LATEST_CKPT = opj(sef.cfg.work_dir, 'latest.pth')
            if os.path.exists(LATEST_CKPT):
                self.cfg.resume_from = LATEST_CKPT
        self.model = build_segmentor(self.cfg.model)
        train_segmentor(self.model, self.datasets, self.cfg, distributed=False, validate=True, 
                meta=dict())
        
    def validate(self):
        '''
        validate the model for quantitative analysis.
        '''
        pass
    
    def infer(self, img_or_folder, save_dir="", device='cpu', video=False, img_suffix='.color.png', latest=True):
        '''
        infer from the model
        this can be done either on the cpu/gpu
        it can also generate a video of the infered segmentation images for qualitative analysis.
        '''
        
        if not os.path.exists(self.model_meta.save_path):
            os.makedirs(self.model_meta.save_path)
        
        CKPT = self.cfg.load_from
        if latest and os.path.exists(opj(self.cfg.work_dir, 'latest.pth')):
            CKPT = opj(self.cfg.work_dir, 'latest.pth')
        
        if self.model is None:
            self.model = init_segmentor(self.cfg, CKPT, device=device)
        
        self.model.CLASSES = self.CLASSES
        self.model.PALETTE = self.PALETTE
        
        files = img_or_folder
        if os.path.isfile(img_or_folder):
            files = [img_or_folder]
            results_path = self.model_meta.save_path
        elif os.path.isdir(img_or_folder):
            files = sorted(glob.glob(img_or_folder + f'/*{img_suffix}'))
            results_path = opj(self.model_meta.save_path, save_dir)
            if not os.path.exists(results_path):
                os.makedirs(results_path)
        
        cv2_writer = None # cv2.VideoWriter(args.video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps=30, frameSize=(320, 480))
        
        for img in tqdm(files):
            result = inference_segmentor(self.model, img)
            img_output = self.model.show_result(img, result, palette=self.PALETTE, show=False, opacity=0.5)
            np_img = np.concatenate([np.array(Image.open(img))[..., ::-1], img_output])
            out_img = Image.fromarray(np_img[..., ::-1])
            
            if video:
                if cv2_writer is None:
                    cv2_writer = cv2.VideoWriter(opj(results_path,  'video_out.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps=30, frameSize=np_img.shape[:2][::-1])
                cv2_writer.write(np_img)
            
            out_img.save(opj(results_path, Path(img).stem + '.png'))
            
        if video:
            cv2_writer.release()
            
        
        
        
    
    