import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import CONFIG

data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        A.CoarseDropout(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ], p=1.),
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(),
        ToTensorV2()
    ], p=1.)
}
