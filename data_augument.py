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
import cv2

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
        # Normalize(p=1.),
        ],p=p
    )
