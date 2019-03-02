import cv2
import glob
from keras.utils import Sequence
import math
import numpy as np
import os
from skimage.io import imread

from utils import *

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Transpose,
    RandomRotate90,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    RandomCrop,
    ShiftScaleRotate,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    HueSaturationValue,
    Normalize
)

def train_aug(p=1.):
    return Compose([
        RandomCrop(256,256,p=1.),
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.1,rotate_limit=45,p=0.5,border_mode=cv2.BORDER_REPLICATE),
        Transpose(),
        OneOf(
            [IAAAdditiveGaussianNoise(),
            GaussNoise(),],
            p=0.2
        ),
        OneOf([
            RandomGamma(p=1.),
            RandomBrightnessContrast(p=1.),
            ],p=0.5
        ),
        HueSaturationValue(p=0.3),
        Normalize(p=1.),
        ],p=p
    )
    
def valid_aug(p=1.):
    return Normalize(p=1.)

aug_train = train_aug()
aug_valid = valid_aug()

class SequenceTrainData(Sequence):
    def __init__(self, df, batch_size, train_image_dir):
        self.df = df
        self.train_image_dir = train_image_dir
        self.batch_size = batch_size
        self.all_batches = list(self.df.groupby('ImageId'))

    def __len__(self):
        num_imgs = len(self.df)
        return math.ceil(num_imgs / self.batch_size)

    def __getitem__(self, idx):
        out_rgb = []
        out_mask = []
        for c_img_id, c_masks in self.all_batches[idx * self.batch_size: (idx + 1) * self.batch_size]:
            rgb_path = os.path.join(self.train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            augment = aug_train(image=c_img, mask=c_mask)
            c_img = augment['image']
            c_mask = augment['mask']
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= self.batch_size:
                return np.stack(out_rgb, 0), np.stack(out_mask, 0)

class SequenceValidData(Sequence):
    def __init__(self, df, batch_size, train_image_dir):
        self.df = df
        self.train_image_dir = train_image_dir
        self.batch_size = batch_size
        self.all_batches = list(self.df.groupby('ImageId'))

    def __len__(self):
        num_imgs = len(self.df)
        return math.ceil(num_imgs / self.batch_size)

    def __getitem__(self, idx):
        out_rgb = []
        out_mask = []
        for c_img_id, c_masks in self.all_batches[idx * self.batch_size: (idx + 1) * self.batch_size]:
            rgb_path = os.path.join(self.train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            augment = aug_valid(image=c_img, mask=c_mask)
            c_img = augment['image']
            c_mask = augment['mask']
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= self.batch_size:
                return np.stack(out_rgb, 0), np.stack(out_mask, 0)