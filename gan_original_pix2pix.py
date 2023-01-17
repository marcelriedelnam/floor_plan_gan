import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from datetime import datetime
import os, os.path, math, sys, shutil, time, scipy, cv2
from tqdm import tqdm


# === REQUIREMENTS ===
# Corresponding floor and support filenames have to be the same
# Floor and support directories have to be in the same directory
# ============

# === GLOBAL VARIABLES ===
DIM = (256, 256)
SAMPLES = 0 # rasterized image output dimension
EPOCHS = 10000 # number of training iterations
REGION = "Gemeinde Schwerte" # change region to load a different data set
DIR = 'Testdaten/testdata_marcel' # the directory where the floor and support directories are saved
LEARNING_RATE_G = 0.001 # learning rate generator
LEARNING_RATE_D = 0.001 # learning rate discriminator
BETA_1 = 0.5 # beta_1 and beta_2 are "coefficients used for computing running averages of gradient and its square" - Adam wiki
BETA_2 = 0.999
MINI_BATCH = 32
TRAIN_TEST_SPLIT = 0.8 # Percentage of trainig to test data (the number is the percentage of training samples)
EXPERIMENT_NAME = "Original GAN" # naming variable to distinguish between experiments
# ===============

# === FUNCTION TO RASTERIZE FLOOR PLATE ===
def raster_images():
    # shapes is of the shape: np.array(list(np.array(np.array)))
    # to get a tuple use shapes[i][0][j][k], i=shape_nr, j=tuple_nr, k=tuple_idx
    shapes = np.load(f"Testdaten/trainingdata/extracteddata/{REGION}.npy", allow_pickle=True)

    # choose SAMPLES many random samples
    buildings = np.random.choice(shapes, SAMPLES)
    shapes = None

    # todo: sind haeuser ueberhaupt richtig? sind ja keine richtigen gebaeude wie in den daten
    # todo: ich male immer nur alles in einem polygon an, k.p. was mit den innenhoefen passiert

    # initialize 3D block for layers of 2D images
    im_as_np_array = np.zeros((len(buildings),) + DIM)
    for i in range(len(buildings)):
        max_val = buildings[i][0].max(axis=0)
        min_val = buildings[i][0].min(axis=0)

        # scale images to size DIM
        buildings[i][0] = (buildings[i][0] - min_val) / (max_val - min_val) * 256

        im = Image.new("L", DIM, 0)
        draw = ImageDraw.Draw(im)
        # https://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
        draw.polygon(tuple(map(tuple, buildings[i][0])), fill=(255))

        # save image in 3D block
        im_as_np_array[i] = np.asarray(im)

    im_as_np_array /= np.amax(im_as_np_array)
    # print(f"raster images min: {np.amin(im_as_np_array)} max: {np.amax(im_as_np_array)}")
    return im_as_np_array
# ====================

# === FUNCTION TO PROCESS DATA ===
def process_images():
    im_as_np_array = np.zeros((len(os.listdir(DIR + "/01_fl")) - 2,2) + DIM, dtype=np.uint8) #-2 because 2 floorplans are blank
    counter = 0
    with open('Testdaten/testdata_processed/Dimensions.txt', 'r') as file:
        data = file.read().rstrip()
    if (data != str(DIM)):
        print("Processing images...")
        file = open(f"Testdaten/testdata_processed/Dimensions.txt", "w")
        file.write(f'{DIM}')
        file.close()
        # initialize 4D block for layers of 2D images (image_nr, floor_or_supprt, pixel_x, pixel_y)
        for files in tqdm(os.listdir(DIR + "/02_sl")):
            # ZB_0048_01_fl.png, ZB_0133_01_fl.png are empty images (no need to generate support structures then)
            if (files == 'ZB_0048_02_sl.png') or (files == 'ZB_0133_02_sl.png'):
                continue
            floor = cv2.imread(os.path.join(DIR + "/02_sl", files), cv2.IMREAD_GRAYSCALE)
            supp_files = files[:len(files)-9] + '03_co.png'
            support = cv2.imread(os.path.join(DIR + "/03_co", supp_files), cv2.IMREAD_GRAYSCALE)
            # turn grayscale image into bw image
            floor = (floor < 255).astype(float)
            support = (support < 255).astype(float)
            floor = cv2.erode(floor, np.full((3, 3), 1)) # TODO is this meaningful?
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
            # crop 1 pixel non-white border (necessary for later calculations)
            h, w = floor.shape
            floor = floor[1:h-1, 1:w-1]
            # get the corners of the floor plans
            whiteY, whiteX = np.where(floor==255)
            top, bottom = np.min(whiteY), np.max(whiteY)
            left, right = np.min(whiteX), np.max(whiteX)
            # calculate the size of the dimensions
            delta_x = right - left
            delta_y = bottom - top
            delta_xy = abs(delta_x - delta_y)
            # crop and center images
            if delta_x < delta_y:
                floor = floor[top:bottom, left:right]
                support = support[top:bottom, left:right]
                if (delta_xy / 2 % 2 == 0):
                    mat = [[0 for col in range(delta_xy // 2)] for row in range(delta_y)]
                    floor = np.concatenate([np.concatenate([mat, floor], axis=1), mat], axis=1)
                    support = np.concatenate([np.concatenate([mat, support], axis=1), mat], axis=1)
                else:
                    mat1 = [[0 for col in range(math.ceil(delta_xy / 2))] for row in range(delta_y)]
                    mat2 = [[0 for col in range(delta_xy // 2)] for row in range(delta_y)]
                    if (len(mat2) == 0):
                        floor = np.concatenate([mat1, floor], axis=1)
                        support = np.concatenate([mat1, support], axis=1)
                    else:
                        floor = np.concatenate([np.concatenate([mat1, floor], axis=1), mat2], axis=1)
                        support = np.concatenate([np.concatenate([mat1, support], axis=1), mat2], axis=1)
            elif (delta_x > delta_y):
                floor = floor[top:bottom, left:right]
                support = support[top:bottom, left:right]
                if (delta_xy / 2 % 2 == 0):
                    mat = [[0 for col in range(delta_x)] for row in range(delta_xy // 2)]
                    floor = np.concatenate([np.concatenate([mat, floor], axis=0), mat], axis=0)
                    support = np.concatenate([np.concatenate([mat, support], axis=0), mat], axis=0)
                else:
                    mat1 = [[0 for col in range(delta_x)] for row in range(math.ceil(delta_xy / 2))]
                    mat2 = [[0 for col in range(delta_x)] for row in range(delta_xy // 2)]
                    if (len(mat2) == 0):
                        floor = np.concatenate([mat1, floor], axis=0)
                        support = np.concatenate([mat1, support], axis=0)
                    else:
                        floor = np.concatenate([np.concatenate([mat1, floor], axis=0), mat2], axis=0)
                        support = np.concatenate([np.concatenate([mat1, support], axis=0), mat2], axis=0)
            else:
                floor = floor[top:bottom, left:right]
                support = support[top:bottom, left:right]
            # draw support structures onto floor plate (instead of only support structures)
            support = (255-support)
            blackY, blackX = np.where(support < 255)
            temp = floor.copy()
            for i in range(len(blackX)):
                temp[blackY[i], blackX[i]] = support[blackY[i], blackX[i]]
            support = temp
            # resize images
            floor = floor.astype(float)
            support = support.astype(float)
            floor = cv2.resize(floor, DIM)
            support = cv2.resize(support, DIM)
            # floor as distance map (distance transform)
            # floor = floor.astype(np.uint8)
            # floor = cv2.distanceTransform(floor, cv2.DIST_L2, 3)
            # floor.astype(float)
            cv2.imwrite(os.path.join('Testdaten/testdata_processed/floors', 'floor%03d' %counter + '.png'), floor)
            cv2.imwrite(os.path.join('Testdaten/testdata_processed/supports', 'support%03d' %counter + '.png'), support)
            im_as_np_array[counter] = np.append(floor, support).reshape((2,) + DIM)
            counter += 1
    # read the images instead of re-calculating them since they are the same
    else:
        print('Loading processed images...`')
        for files in tqdm(os.listdir('Testdaten/testdata_processed/floors')):
            floor = cv2.imread('Testdaten/testdata_processed/floors/' + files,  cv2.IMREAD_GRAYSCALE)
            supp_files = 'support' + files[5:]
            support = cv2.imread('Testdaten/testdata_processed/supports/' + supp_files,  cv2.IMREAD_GRAYSCALE)
            im_as_np_array[counter] = np.append(floor, support).reshape((2,) + DIM)
            counter += 1
    im_as_np_array = torch.FloatTensor(im_as_np_array)
    return im_as_np_array
# ======================

# === CONVOLUTIONAL LAYER STRUCTURE ===
class ConvLayerGen(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x, 0.2)
        return x
# ==============

# === DECONVOLUTIONAL LAYER STRUCTURE ===
class DeconvLayerGen(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.deconv(x)
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        x = F.relu(x)
        return x
# =============

# === RESIDUAL BLOCK STRUCTURE ===
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.relu(x)
        return x
# =========

# === DISCRIMINATOR MODEL ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=2, stride=1),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss().cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))

    def forward(self, input):
        return self.model(input)
# ==============

# === GENERATOR MODEL ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Convulational Layers
            ConvLayerGen(1, 2),
            ConvLayerGen(2, 4),
            ConvLayerGen(4, 8),
            ConvLayerGen(8, 16),
            ConvLayerGen(16, 32),
            # ResNet
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            # ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            # Deconvolutional Layers
            DeconvLayerGen(32, 16),
            DeconvLayerGen(16, 8),
            DeconvLayerGen(8, 4),
            DeconvLayerGen(4, 2),
            DeconvLayerGen(2,1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))

    def forward(self, inputs):
        return self.model(inputs)
# =============

# === CUSTOM DATASET ===
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
            floor = self.transform(floor)
            support = self.transform(support)
        return floor, support
# ===========

# === TRAINING GAN ===
def main():
    # start time
    time_start = datetime.now()

    print("Uploading models to GPU...")
    D = Discriminator()
    D.cuda()
    G = Generator()
    G.cuda()
    print("Done.")

    # save this file, rasterizer.py and some parameters for later reference
    res_dir = time_start.strftime('%Y-%m-%d %H:%M:%S ') + EXPERIMENT_NAME
    os.mkdir(os.path.join("Ergebnisse/", res_dir))
    shutil.copyfile(sys.argv[0], os.path.join(os.path.join("Ergebnisse/", res_dir), "GAN.py"))
    file = open(f"Ergebnisse/{res_dir}/Specs.txt", "w")
    file.write(f"Dimension of the pictures = {DIM}\n")
    file.write(f"Number of epochs = {EPOCHS}\n")
    file.write(f"Number of training samples = {SAMPLES}\n")
    file.write(f"Batch size = {MINI_BATCH}\n")
    file.write(f"Learning rate of the generator = {LEARNING_RATE_G}\n")
    file.write(f"Learning rate of the discriminator = {LEARNING_RATE_D}\n")
    file.write(f"Beta 1 = {BETA_1}\n")
    file.write(f"Beta 2 = {BETA_2}\n")
    file.write(f"Dataset from which traning samples are taken = {REGION}\n")
    file.write(f"Directory where the floor/support directories are for testing = {DIR}\n")
    file.write(f"\n {G} \n\n {D}\n")
    file.close()

    # create new dataset with data augmentation (transform)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0,360)),
        transforms.RandomVerticalFlip()
    ])
    dataset = CustomImageDataset()
    # create a training set and a test set according to TRAIN_TEST_SPLIT
    train_set_size = math.ceil(len(dataset) * TRAIN_TEST_SPLIT)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_set_size, test_set_size])
    # create dataloaders
    train_dataloader = DataLoader(train_set, batch_size=MINI_BATCH, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_set, batch_size=MINI_BATCH, shuffle=True, drop_last=True)

    time_now = datetime.now() # used to time how long test/training samples, network are uploaded and Specs.txt is created

    # start training
    print("Starting training")

    # https://stackoverflow.com/questions/51520143/update-matplotlib-image-in-a-function
    # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    fig, (ax1,ax2) = plt.subplots(1,2)
    im = ax1.imshow(np.zeros(DIM+(3,)))
    # im2 = ax3.imshow(np.zeros(DIM+(3,)))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    progress_loss_G = []
    progress_loss_D = []

    for epoch in range(EPOCHS):
        for (floors, supports) in train_dataloader:
            floors, supports = floors.cuda(), supports.cuda()
            
            # train discriminator D
            D.zero_grad()
            floors = floors.reshape((MINI_BATCH,1)+DIM)
            supports = supports.reshape((MINI_BATCH,1)+DIM)
            train_img_batch = torch.cat((floors, supports), 1)

            D_res = D.forward(train_img_batch)
    
            D_real_loss = D.loss_function(D_res, torch.ones_like(D_res, requires_grad=True).cuda())

            G_res = G.forward(floors)
            generated_imgs = torch.cat((floors, G_res), 1)

            D_res = D.forward(generated_imgs)
            D_fake_loss = D.loss_function(D_res, torch.zeros_like(D_res, requires_grad=True).cuda()).cuda()

            D_train_loss = (D_fake_loss + D_real_loss) * 0.5
            D.optimizer.step()

            # train generator G
            G.zero_grad()
            D.zero_grad()

            G_res = G.forward(floors)
            D_res = D.forward(torch.cat((floors, G_res), 1))

            G_train_loss = D.loss_function(D_res, torch.ones_like(D_res, requires_grad=True).cuda()).cuda()
            G_train_loss.backward()
            G.optimizer.step()

            progress_loss_G.append(G_train_loss.item())
            progress_loss_D.append(D_train_loss.item())

            if epoch % 1 == 0:
                image = np.zeros(DIM+(3,))
                image[:,:,0] = floors[0, 0].cpu().detach().numpy()
                image[:,:,1] = G_res[0, 0].cpu().detach().numpy()
                im.set_data(image)

                # input_img = train_imgs[torch.randint(0, train_imgs.shape[0], (1,))]
                # image[:,:,0] = input_img.cpu().detach().numpy()
                # image[:,:,1] = G.forward(input_img.reshape((1,1)+DIM).cuda())[0][0].cpu().detach().numpy()
                # im2.set_data(image)

                ax2.clear()
                ax2.plot(range(len(progress_loss_D)), progress_loss_D, label="D")
                ax2.plot(range(len(progress_loss_G)), progress_loss_G, label="G")
                ax2.legend()
                fig.canvas.draw_idle()
                plt.savefig(f'Ergebnisse/{res_dir}/{epoch}.png')
                fig.canvas.flush_events()

                
                


    time_end = datetime.now()
    file = open(f"Ergebnisse/{res_dir}/Specs.txt", "a")
    file.write(f"\n Execution start = {time_start}\n")
    file.write(f"Exectuion stop = {time_end}\n")
    file.write(f"Total execution time = {time_end - time_start}\n")
    file.write(f"Time used to upload training samples, test samples and the network to the GPU = {time_start - time_now}\n")
    file.close()
# ==============

if __name__ == "__main__":
    main()