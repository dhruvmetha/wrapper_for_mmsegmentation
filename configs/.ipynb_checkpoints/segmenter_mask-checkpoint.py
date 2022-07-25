from .base_config import BaseConfig, CLASSES, PALETTE
import os

opj = lambda *args: os.path.join(*args)

name = 'segmenter_mask_50k'
CONFIG_FOLDER = '/common/home/dm1487/Spring22/robotics/segmentation/mmsegmentation/configs'
MODEL_CONFIG = opj(CONFIG_FOLDER, 'segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py')

TRAINVAL_DATA = '/common/users/dm1487/ade_small/ADEChallengeData2016'
TEST_DATA = '/common/users/dm1487/ade_small/ADEChallengeData2016'

LOAD_CKPT = 'pretrain/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth'
BASE_WORK_DIR = "/common/users/dm1487/segmentation_models/"

class Config(BaseConfig):
    
    class model_meta:
        config_folder = CONFIG_FOLDER
        model_config = MODEL_CONFIG
        save_path = opj(BASE_WORK_DIR, name, 'results')
        # load_ckpt = LOAD_CKPT
        classes = CLASSES
        palette = PALETTE
        
    class cfg:
        # norm_cfg = NORM_CFG
        data_root = TRAINVAL_DATA
        load_from = LOAD_CKPT
        device = 'cuda'
        work_dir = opj(BASE_WORK_DIR, name)
        seed = 0
        # gpu_ids = [0]
        # workflow = [('train', 1)]
        class model:
            pretrained = LOAD_CKPT
            class decode_head:
                num_classes = len(CLASSES)
                # norm_cfg = NORM_CFG
                
        class data:
            samples_per_gpu = 4
            class train:
                data_root = TRAINVAL_DATA
                classes = CLASSES
            class val:
                data_root = TRAINVAL_DATA
                classes = CLASSES
            class test:
                data_root = TRAINVAL_DATA
                classes = CLASSES
                
        class runner:
            max_iters = 50000
            
        class log_config:
            interval = 500
            
        class evaluation:
            interval = 5000
            
        class checkpoint_config:
            interval = 5000