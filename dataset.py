import cv2
import numpy as np

import scipy
from scipy.ndimage import label

import mclahe
import albumentations as albu

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


def refine_TLlabel(lbl):
    """
    In vertebral level segmentation task, we have L1, other T-L spine and sacrum class.
    Therefore, we need to refine.
    input : H x W 
    result : H x W
    """
    convert_2to1 = False
    lbl_sacrum = lbl.copy()
    lbl_sacrum[lbl_sacrum!=3] = 0
    
    lbl_1 = lbl.copy()
    lbl_1[lbl_1==3] = 0
    lbl_1[lbl_1!=0] = 1
    result = np.zeros_like(lbl_1)

    spines, _ = label(lbl_1)
    for j in range(1, len(np.unique(spines))):
        spine = spines.copy()
        spine[spine!=j] = 0
        spine[spine!=0] = 1
        unique, counts = np.unique(spine*lbl,return_counts=True)

        if 1 in unique:
            convert_2to1 = True

        if convert_2to1 and 3 not in unique:
            spine = spine
        elif 3 in unique:
            spine = spine * 3
        else:
            spine = spine*lbl
        result = result + spine
    result = result + lbl_sacrum

    return result

def clahe(img, adaptive_hist_range=False):
    """
    input 1 numpy shape image (H x W x (D) x C)
    """
    temp = np.zeros_like(img)
    for idx in range(temp.shape[-1]):
        temp[...,idx] = mclahe.mclahe(img[...,idx], n_bins=128, clip_limit=0.04, adaptive_hist_range=adaptive_hist_range)
    return temp

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        
        albu.OneOf(
        [
        albu.RandomBrightnessContrast(brightness_limit=(-0.3, 0.0), contrast_limit=(-0.1, 0.1), brightness_by_max=True, p=.5),
        albu.RandomGamma(gamma_limit=(90,110), p=.5),
        albu.RandomToneCurve(scale=0.1, p=.5), 
        albu.HueSaturationValue(hue_shift_limit=(-10,5), sat_shift_limit=(-10,5), val_shift_limit=(-10,5), p=.5),
        albu.InvertImg(p=.5),
        ],p=0.5,
        ),
        
        albu.OneOf(
        [
        albu.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, src_radius=100, p=0.5,),
        albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=.1, p=0.5), 
        ],p=0.5,
        ),
        
        albu.OneOf(
        [    
        albu.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
        albu.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
        albu.ISONoise(color_shift=(0.01, 0.02),intensity=(0.1, 0.3),p=0.5),
        ], p=0.5,
        ),        
        
        albu.OneOf([
        albu.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, alpha=1, sigma=50, alpha_affine=50, p=0.5),
        albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, distort_limit=0.3, num_steps=5, p=0.5),
        albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, distort_limit=.05, shift_limit=0.05, p=0.5),
        ],p=0.5),

        albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=90, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, p=.5),
    ]
    return albu.Compose(train_transform, additional_targets={'mask1': 'mask'})


def get_validation_augmentation(): 
    test_transform = [
    ]
    return albu.Compose(test_transform, additional_targets={'mask1': 'mask'})

class Dataset(BaseDataset):
    """    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    def __init__(
            self, 
            images_list , 
            fracture_list, 
            position_list, 
            augmentation=None, 
    ):
        self.images_list = images_list
        self.fracture_list = fracture_list
        self.position_list = position_list
        self.augmentation = augmentation
        print('images {} {} {}'.format(len(images_list),len(fracture_list),len(position_list)))
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, i):
        # read data
        fname = self.images_list[i]
        image = cv2.imread(self.images_list[i]) 
        fracture = cv2.imread(self.fracture_list[i], cv2.IMREAD_GRAYSCALE)
        ''' back / spine / fracutre / sacrum
            0->0 , 1,2->1 , 3->2 , 4->3      '''
        fracture = np.where(fracture==2,1,fracture)
        fracture = np.where(fracture==3,2,fracture)
        fracture = np.where(fracture==4,3,fracture)
        fracture = np.expand_dims(fracture,-1) # (H,W,1). uint8

        position = cv2.imread(self.position_list[i], cv2.IMREAD_GRAYSCALE)
        position = refine_TLlabel(position)
        position = np.expand_dims(position,-1) # (H,W,1), uint8
        
        # apply augmentations, need to be numpy shape (H,W,C)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=fracture, mask1= position)
            image, fracture, position = sample['image'], sample['mask'], sample['mask1']
        
        # change to tensor shape (C,H,W) and float32
        image = image.astype(np.float32)
        image = clahe(image,True)
        image = np.moveaxis(image,-1,0)
        image = torch.tensor(image)
        
        fracture = np.moveaxis(fracture,-1,0)
        position = np.moveaxis(position,-1,0)
        unique = np.unique(fracture)
        
        return {'x' : image, 'y_fracture' : fracture, 'y_position' : position, 'fname': fname}