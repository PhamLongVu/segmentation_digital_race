from engine import engine
from loss import SegmentationLoss
from config import config
from tqdm import tqdm
from model import make_model
from dataset2 import get_dataloaders
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

lookup_table = np.array([[0, 0, 0],   # Value 0 maps to (0, 0, 0) (Black)
                         [255, 0, 0], # Value 1 maps to (255, 0, 0) (Red)
                         [0, 255, 0]])

def predict(model, image):

    transform = A.Compose([


        ToTensorV2(),
        # You can add more Albumentations transformations here
    ])
    print(image.shape)
    augmented = transform(image=image)
    image = augmented['image']
    print(image.shape)
    image = image.unsqueeze(0)
    print(image.shape)
    image = image.to(config.DEVICE).float()
    _masks = model(image)
    print(_masks.shape)
    _masks = _masks.squeeze(0)
    print(_masks.shape)
    _masks = _masks.detach().cpu().numpy()
    print(_masks.shape)
    _masks = np.transpose(_masks, (1, 2, 0))
    _masks = np.argmax(_masks, axis=-1).astype(np.uint8)
    # _masks = lookup_table[_masks]
    return _masks#np.expand_dims(_masks, axis=-1)


model, criterion, optimizer, scheduler = make_model()
model.load_state_dict(torch.load('best_loss.pth'))
model.eval()  # Set the model to evaluation mode
image = cv2.imread("/home/vupl/Documents/unet/data2/train_image/data_10.jpg")
mask =predict(model, image)
print(mask.shape)
# save image
# cv2.imwrite("mask.png", mask)

import matplotlib.pyplot as plt
plt.imshow(mask, cmap='gray')
plt.show()

# cv2.imshow("mask",mask)
# cv2.waitKey(0)