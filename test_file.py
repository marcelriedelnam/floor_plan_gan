import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy, os
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
from tqdm import tqdm

DATA_AUG = False
IMG_PROC = True
NO_CAT = False
CAT = True
DIM = (512,512)


# === TEST IMAGE PROCESSING ===
if IMG_PROC == True:
    floor = cv2.imread('Testdaten/testdata_marcel/02_sl/ZB_0035_02_sl.png', cv2.IMREAD_GRAYSCALE)
    support = cv2.imread('Testdaten/testdata_marcel/03_co/ZB_0035_03_co.png', cv2.IMREAD_GRAYSCALE)
    floor = (floor < 255).astype(float)
    support = (support < 255).astype(float)
    floor = cv2.erode(floor, np.full((3, 3), 1))
    floor = cv2.dilate(floor, np.full((3, 3), 1))
    support = cv2.erode(support, np.full((3, 3), 1))
    support = cv2.dilate(support, np.full((3, 3), 1))
    kernel = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])
    floor = scipy.ndimage.convolve(floor, kernel) >= 2
    support = scipy.ndimage.convolve(support, kernel) >= 2
    floor = floor.astype(np.uint8) * 255
    support = support.astype(np.uint8) * 255
    floor = cv2.resize(floor, (512,512), interpolation=cv2.INTER_AREA)
    support = cv2.resize(support, (512,512), interpolation=cv2.INTER_AREA)
    support = 255 - support
    t,support = cv2.threshold(support, 254, 255, cv2.THRESH_BINARY)
    t,floor = cv2.threshold(floor, 0, 255, cv2.THRESH_BINARY)
    support = floor * support
    plt.imshow(support, 'gray')
    plt.show()
# ====================

# === TEST DATA AUGMENTATION ===
def process_images():
    im_as_np_array = np.zeros((3,2) + DIM, dtype=np.uint8)

    floor = cv2.imread('Testdaten/testdata_processed/floors/floor006.png', cv2.IMREAD_GRAYSCALE)
    support = cv2.imread('Testdaten/testdata_processed/supports/support006.png', cv2.IMREAD_GRAYSCALE)
    im_as_np_array[0] = np.append(floor, support).reshape((2,) + DIM)

    floor = cv2.imread('Testdaten/testdata_processed/floors/floor016.png', cv2.IMREAD_GRAYSCALE)
    support = cv2.imread('Testdaten/testdata_processed/supports/support016.png', cv2.IMREAD_GRAYSCALE)
    im_as_np_array[1] = np.append(floor, support).reshape((2,) + DIM)

    floor = cv2.imread('Testdaten/testdata_processed/floors/floor024.png', cv2.IMREAD_GRAYSCALE)
    support = cv2.imread('Testdaten/testdata_processed/supports/support024.png', cv2.IMREAD_GRAYSCALE)
    im_as_np_array[2] = np.append(floor, support).reshape((2,) + DIM)

    im_as_np_array = torch.FloatTensor(im_as_np_array) / 255
    return im_as_np_array

class CustomImageDataset(Dataset):
    def __init__(self, transform=None):
        self.img_arr = process_images()
        self.transform = transform

    def __len__(self):
        return self.img_arr.shape[0]

    def __getitem__(self, index):
        floor = self.img_arr[index][0]
        support = self.img_arr[index][1]
        if self.transform:
            if NO_CAT == True:
                floor = self.transform(floor)
                support = self.transform(support)
            if CAT == True:
                floor = floor.reshape((1,)+DIM)
                support = support.reshape((1,)+DIM)
                to_trnsfm = torch.cat((floor, support))
                trnsfmd = self.transform(to_trnsfm)
                floor = trnsfmd[0]
                support = trnsfmd[1]
        return floor, support

if DATA_AUG == True:
    transform = transforms.RandomChoice([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((90,90)),
        transforms.RandomRotation((180,180)),
        transforms.RandomRotation((270,270)),
        transforms.RandomVerticalFlip()
    ])
    dataset = CustomImageDataset(transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    for (floor, support) in dataloader:
        fig, ((ax1, ax2)) = plt.subplots(1,2)
        floor = floor.reshape(DIM)
        support = support.reshape(DIM)
        ax1.imshow(floor, 'gray')
        ax2.imshow(support, 'gray')
        plt.show()
# ===========================