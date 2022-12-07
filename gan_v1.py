import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as Variable
from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os, os.path, rasterizer

# === REQUIREMENTS ===
# Corresponding floor and support filenames have to be the same
# Floor and support directories have to be named "floor" and "support" and have to be in the same directory
# ============

# === GLOBAL VARIABLES ===
DIM = (256, 256)
SAMPLES = 10000 # rasterized image output dimensions
EPOCHS = 10000 # number of training iterations
REGION = "Gemeinde Schwerte" # change region to load a different data set
DIR = 'Testdaten/testdatav1' # the directory where the floor and support directories are saved
# NUM_TRAIN_DATA = len(os.listdir(DIR + "/floor")) # NUM_TRAIN_DATA = get the number of test samples (files) from the directory DIR
NUM_TRAIN_DATA = 69 # NUM_TRAIN_DATA = get the number of test samples (files) from the directory DIR
LEARNING_RATE_G = 0.02 # learning rate generator
LEARNING_RATE_D = 0.002 # learning rate discriminator
BETA_1 = 0.5 # beta_1 and beta_2 are "coefficients used for computing running averages of gradient and its square" - Adam wiki
BETA_2 = 0.999
MINI_BATCH = NUM_TRAIN_DATA
# ===============

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
        x = F.leaky_relu(x)
        return x
# ==============

# === DECONVOLUTIONAL LAYER STRUCTURE ===
class DeconvLayerGen(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.deconv(x)
        x = F.leaky_relu(x)
        x = F.pad(x, (0, -1, 0, -1), "constant", 0)
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
            nn.Conv2d(2, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.loss_function = nn.MSELoss().cuda()
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
            ConvLayerGen(1, 8),
            ConvLayerGen(8, 32),
            ConvLayerGen(32, 128),
            ConvLayerGen(128, 512),
            # ResNet
            # ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            # ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            # Deconvolutional Layers
            DeconvLayerGen(512, 128),
            DeconvLayerGen(128, 32),
            DeconvLayerGen(32, 8),
            DeconvLayerGen(8, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))

    def forward(self, inputs):
        return self.model(inputs)
# =============

# === TRAINING GAN ===
def main():
    # load the training data
    print("Uploading training data samples to GPU...")
    # DIM = dimension of the pictures, SAMPLES = how many samples, DATA = 
    train_imgs = torch.FloatTensor(rasterizer.raster_images(DIM, SAMPLES, REGION)).cuda()
    # print(f"train_imgs.shape={train_imgs.shape}")
    print("Done.")

    # load test data as images, crop and resize
    print("Uploading test data samples to GPU...")
    # initialize 4D block for layers of 2D images (image_nr, floor_or_supprt, pixel_x, pixel_y)
    im_as_np_array = np.zeros((NUM_TRAIN_DATA,2) + DIM, dtype=np.float32)
    counter = 0
    for files in os.listdir(DIR + "/floor")[:NUM_TRAIN_DATA]:
        floor = Image.open(os.path.join(DIR + "/floor", files)).convert('L')
        support = Image.open(os.path.join(DIR + "/supports", files)).convert('L')
        img_arr = np.array(floor)
        # get the corners of the floor plans
        whiteY, whiteX = np.where(img_arr==255)
        top, bottom = np.min(whiteY), np.max(whiteY)
        left,right = np.min(whiteX), np.max(whiteX)
        # calculate the size of the dimension
        delta_x = right - left
        delta_y = bottom - top
        delta_xy = abs(delta_x - delta_y)
        # crop and resize the images
        if delta_x < delta_y:
            floor = floor.crop((left - delta_xy / 2, top, right + delta_xy / 2, bottom)).resize(DIM)
            support = support.crop((left - delta_xy / 2, top, right + delta_xy / 2, bottom)).resize(DIM)
        else:
            floor = floor.crop((left, top - delta_xy / 2, right, bottom + delta_xy / 2)).resize(DIM)
            support = support.crop((left, top - delta_xy / 2, right, bottom + delta_xy / 2)).resize(DIM)
        # save image in 4D block
        im_as_np_array[counter] = np.append(np.array(floor), np.array(support)).reshape((2,) + DIM)
        counter += 1
    test_imgs = torch.FloatTensor(im_as_np_array).cuda()
    # print(f"test_imgs.shape={test_imgs.shape}")
    im_as_np_array = None
    print("Done.")

    print("Uploading models to GPU...")
    D = Discriminator()
    D.cuda()
    G = Generator()
    G.cuda()
    print("Done.")

    # start training
    print("Starting training")

    # https://stackoverflow.com/questions/51520143/update-matplotlib-image-in-a-function
    fig, (ax1,ax2) = plt.subplots(1,2)
    im = ax1.imshow(np.zeros(DIM+(3,)))

    progress_loss_G = []
    progress_loss_D = []

    for epoch in range(EPOCHS):
        # print every 5% of training process
        if epoch % (EPOCHS * 0.05) == 0:
            print(f"{epoch / EPOCHS * 100}%")
        
        # train discriminator D
        D.zero_grad()

        D_res = D.forward(test_imgs)
        # print(f"D_res.shape={D_res.shape}")
        # print(f"D_res={D_res}")
        # print("ones:", torch.ones_like(D_res, requires_grad=True).cuda())

        D_real_loss = D.loss_function(D_res, torch.ones_like(D_res, requires_grad=True).cuda())
        # print(f"D_real_loss.shape={D_real_loss.shape}")

        indices = torch.randint(0, SAMPLES, (MINI_BATCH,)).cuda()
        # print(f"indices.shape={indices.shape}")
        train_imgs_batch = torch.index_select(train_imgs.reshape((SAMPLES, 1)+DIM), 0, indices)
        # print(f"train_imgs_batch.shape={train_imgs_batch.shape}")

        G_res = G.forward(train_imgs_batch)
        # print(f"G_res.shape={G_res.shape}")
        generated_imgs = torch.cat((train_imgs_batch, G_res), 1)
        # print(f"generated_imgs.shape={generated_imgs.shape}")

        D_res = D.forward(generated_imgs)
        # print(f"D_res.shape={D_res.shape}")
        D_fake_loss = D.loss_function(D_res, torch.zeros_like(D_res, requires_grad=True).cuda()).cuda()

        D_train_loss = (D_fake_loss + D_real_loss) * 0.5
        D_train_loss.backward()
        D.optimizer.step()

        # train generator G
        G.zero_grad()
        D.zero_grad()

        G_res = G.forward(train_imgs_batch)
        D_res = D.forward(torch.cat((train_imgs_batch, G_res), 1))

        # G_train_loss = D.loss_function(D_res, torch.ones_like(D_res, requires_grad=True).cuda()).cuda()
        G_train_loss = D.loss_function(D_res, torch.ones_like(D_res, requires_grad=True).cuda()).cuda()
        G_train_loss.backward()
        G.optimizer.step()

        progress_loss_G.append(G_train_loss.item())
        progress_loss_D.append(D_train_loss.item())

        if epoch % 1 == 0:
            image = np.zeros(DIM+(3,))
            image[:,:,0] = train_imgs_batch[0,0].cpu().detach().numpy()
            image[:,:,1] = G_res[0,0].cpu().detach().numpy()
            
            im.set_data(image)
            ax2.clear()
            ax2.plot(range(len(progress_loss_D)), progress_loss_D, label="D")
            ax2.plot(range(len(progress_loss_G)), progress_loss_G, label="G")
            ax2.legend()
            fig.canvas.draw_idle()
            plt.pause(0.01)

    # plt.plot(np.arange(len(D.progress)) / len(D.progress) , D.progress, label="D")
    # plt.plot(np.arange(len(G.progress)) / len(G.progress), G.progress, label="G")
    # plt.legend()
    # plt.show()
    # plt.imshow(G.forward(torch.cuda.FloatTensor(10).normal_()).detach().cpu().numpy().reshape(imgs[0].shape))
    # plt.show()
# ==============

if __name__ == "__main__":
    main()