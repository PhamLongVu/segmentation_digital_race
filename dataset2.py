from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import config

class SegmentationData(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background', 'boundary','road' ]
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            transform=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id[:-4] + '_mask.png') for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [0, 122, 255]
        
        self.transform = transform
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        print(len(masks))
        mask = np.stack(masks, axis=-1).astype('float')
        print(mask[:,:,2])
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
def get_dataloaders():
    dir = "data2"
    
    # Define Albumentations transformations
    transform_train = A.Compose([
        A.RandomBrightnessContrast(p=0.3),
        # Apply image blurring
        A.OneOf([
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.RandomRain(),
            A.RandomSnow(),
        ], p=0.3),

        ToTensorV2(),
        # You can add more Albumentations transformations here
    ])
    transform_val = A.Compose([

        ToTensorV2(),
        # You can add more Albumentations transformations here
    ])

    trn_ds = SegmentationData(os.path.join(dir, 'train_image'), os.path.join(dir, 'train_mask'), transform=transform_train)
    val_ds = SegmentationData(os.path.join(dir, 'test_image'), os.path.join(dir, 'test_mask'), transform=transform_val)
    trn_dl = DataLoader(trn_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

    return trn_dl, val_dl


if __name__ == "__main__":
    trn_dl, val_dl = get_dataloaders()
    for bx, data in enumerate(trn_dl):
        image, mask = data
        print(mask.shape)
        break

    

