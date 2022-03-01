import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";
LEARNING_RATE = 1e-5
BATCH_SIZE = 6
NUM_WORKERS = 2
IMAGE_SIZE = 256
NUM_EPOCHS = 40
EPSILON = 1e-5
EARLY_STOPPING_PATIENCE = 5;

 #Initialize transforms for training and validation
train_transforms = A.Compose(
[
    #A.PadIfNeeded(min_height = 512, min_width = 512),
    #A.RandomCrop(Config.IMAGE_SIZE, Config.IMAGE_SIZE, always_apply = False, p = 0.5),
    #A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
    #A.HorizontalFlip(p=0.5),
    #A.RandomBrightnessContrast(p=0.5),
    #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)

valid_transforms = A.Compose(
    [
    #A.PadIfNeeded(min_height = 512, min_width = 512),
    #A.RandomCrop(Config.IMAGE_SIZE, Config.IMAGE_SIZE, always_apply = True),
    #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ]
)
