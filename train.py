import warnings
warnings.filterwarnings(action='ignore')

import time
import random
import os, sys, shutil
import multiprocessing

import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm, trange
import natsort
from natsort import natsorted

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import monai
import mclahe

import scipy
from scipy.ndimage import label
import sklearn
import sklearn.model_selection 
from sklearn.model_selection import train_test_split

import torchmetrics
import albumentations as albu
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
import segmentation_models_pytorch as smp

import dataset
from dataset import Dataset
import model_multitask
import lightening_segmenter
from config import parse_arguments


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    monai.utils.misc.set_determinism(seed=seed)
    pl.seed_everything(42)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_(img, alpha):    
    try:
        plt.imshow(img.cpu().detach().numpy(), cmap='gray', alpha=alpha)
    except:
        plt.imshow(img,cmap='gray',alpha=alpha)

def plot_color(img, alpha):
    try:
        plt.imshow((img.cpu().detach().numpy())/3, cmap='bone', alpha=alpha)
    except:
        plt.imshow(img/3,cmap='bone',alpha=alpha)
        
def visualize(**images):
    """Plot images in one row."""
    
    n = len(images)
    plt.figure(figsize=(26, 12))
    
    for i, (name, image) in enumerate(images.items()):
        if torch.is_tensor(image) and image.shape[0] == 3:
            image = image.permute(1,2,0)
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if i == 0:
            image_ori = image
            plot_(image_ori,1)
        else:
            plot_(image_ori,1)
            plot_color(image,0.7)
    plt.show()


def main(args):
    gpus = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus;
    set_seed()
    device = get_device()
    NUM_WORKERS = os.cpu_count()
    print("Number of workers:", NUM_WORKERS)
    
    DATA_DIR = args.DATA_DIR
    x_train = natsorted(glob(os.path.join(DATA_DIR, args.train_img,'*.png')))
    # b:0 , L1: 1, Spine: 2, sacrum:3
    y_train_fracture = natsorted(glob(os.path.join(DATA_DIR, args.train_fx,'*.png')))
    # b:0 , L1: 1
    y_train_position = natsorted(glob(os.path.join(DATA_DIR, args.train_level,'*.png')))

    # sklearn의 train_test_split를 사용해 데이터를 나눕니다. 반복성 있게 데이터가 나누어 집니다.
    x_train, x_valid, y_train_fracture, y_valid_fracture, y_train_position, y_valid_position = train_test_split(x_train,y_train_fracture,y_train_position,test_size=0.1,random_state=42)

    x_test = natsorted(glob(os.path.join(DATA_DIR, args.test_img, '*.png')))
    y_test_fracture = natsorted(glob(os.path.join(DATA_DIR, args.test_fx, '*.png')))
    y_test_position = natsorted(glob(os.path.join(DATA_DIR, args.test_level, '*.png')))

    train_dataset = Dataset(x_train, y_train_fracture, y_train_position, 
    augmentation=dataset.get_training_augmentation()
    )
    valid_dataset = Dataset(x_valid, y_valid_fracture, y_valid_position,
    augmentation=dataset.get_validation_augmentation()
    )
    # test_dataset = Dataset(x_test,y_test_fracture, y_test_position, 
    # augmentation=dataset.get_validation_augmentation()
    # )

    batch_size= args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    net = model_multitask.Unet()
    lossfn = [monai.losses.DiceLoss(to_onehot_y=True), monai.losses.DiceLoss(to_onehot_y=True)]
    metricfn = [torchmetrics.functional.dice_score, torchmetrics.functional.dice_score]
    experiment_name = args.experiment_name
    model = lightening_segmenter.Segmentor(network=net, lossfn=lossfn, metricfn=metricfn, experiment_name=experiment_name,)

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=f"weight/{experiment_name}/", save_top_k=2, monitor="val_loss_epoch")
    trainer = pl.Trainer(gpus=-1, strategy='dp', precision=32, max_epochs=1000, callbacks=[checkpoint_callback, early_stop_callback])
    
    trainer.fit(model, train_loader, valid_loader)



if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    print("="*30)
    print(argv)
    print("="*30)
    print()
    main(argv)