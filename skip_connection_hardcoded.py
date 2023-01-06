import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as Variable
from PIL import Image, ImageDraw, ImageOps
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from datetime import datetime
import os, os.path, rasterizer, sys, shutil, time

# === REQUIREMENTS ===
# Corresponding floor and support filenames have to be the same
# Floor and support directories have to be named "floor" and "support" and have to be in the same directory
# ============

# === GLOBAL VARIABLES ===
DIM = (256, 256)
SAMPLES = 5000 # rasterized image output dimensions
EPOCHS = 2000 # number of training iterations
REGION = "Gemeinde Schwerte" # change region to load a different data set
DIR = 'Testdaten/testdatav1' # the directory where the floor and support directories are saved
NUM_TRAIN_DATA = len(os.listdir(DIR + "/floor")) # NUM_TRAIN_DATA = get the number of test samples (files) from the directory DIR
LEARNING_RATE_G = 0.02 # learning rate generator
LEARNING_RATE_D = 0.02 # learning rate discriminator
BETA_1 = 0.9 # beta_1 and beta_2 are "coefficients used for computing running averages of gradient and its square" - Adam wiki
BETA_2 = 0.999
MINI_BATCH = 64
EXPERIMENT_NAME = "Skip Connection" # naming variable to distinguish between experiments
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
        x = F.leaky_relu(x, 0.2)
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
            nn.Sigmoid(),
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
        self.conv = ConvLayerGen(1, 8)
        self.model = nn.Sequential(
            ConvLayerGen(8, 32),
            ConvLayerGen(32, 128),
            ConvLayerGen(128, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            DeconvLayerGen(512, 128),
            DeconvLayerGen(128, 32),
            DeconvLayerGen(32, 8)
        )
        self.deconv = DeconvLayerGen(8, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.model(x1)
        x3 = self.deconv(x2)
        x4 = F.sigmoid(x3)
        x5 = x4 * inputs # input black ===> output black
        return x5

# =============

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
    shutil.copyfile(f"Code/rasterizer.py", os.path.join(os.path.join("Ergebnisse/", res_dir), "rasterizer.py"))
    file = open(f"Ergebnisse/{res_dir}/Specs.txt", "w")
    file.write(f"Dimension of the pictures = {DIM}\n")
    file.write(f"Number of epochs = {EPOCHS}\n")
    file.write(f"Number of training samples = {SAMPLES}\n")
    file.write(f"Number of test samples = {NUM_TRAIN_DATA}\n")
    file.write(f"Batch size = {MINI_BATCH}\n")
    file.write(f"Learning rate of the generator = {LEARNING_RATE_G}\n")
    file.write(f"Learning rate of the discriminator = {LEARNING_RATE_D}\n")
    file.write(f"Beta 1 = {BETA_1}\n")
    file.write(f"Beta 2 = {BETA_2}\n")
    file.write(f"Dataset from which traning samples are taken = {REGION}\n")
    file.write(f"Directory where the floor/support directories are for testing = {DIR}\n")
    file.write(f"\n {G} \n\n {D}\n")
    file.close()

    # load the training data
    print("Uploading training data samples to GPU...")
    # DIM = dimension of the pictures, SAMPLES = how many samples, DATA = 
    train_imgs = torch.FloatTensor(rasterizer.raster_images(DIM, SAMPLES, REGION)).cuda()
    # print(f"train_imgs.shape={train_imgs.shape} dtpe={train_imgs.dtype}")
    print("Done.")

    # load test data as images, crop and resize
    print("Uploading test data samples to GPU...")
    # initialize 4D block for layers of 2D images (image_nr, floor_or_supprt, pixel_x, pixel_y)
    im_as_np_array = np.zeros((NUM_TRAIN_DATA*8,2) + DIM, dtype=np.float32) # NUM_TRAIN_DATA*8 because the pictures are flipped (horizontally/ vertically) and rotated (90,180,270 deg)
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
        # Data Augmentation
        # rotate by 90, 180 and 270 degrees
        floor_r90 = floor.rotate(90)
        floor_r180 = floor.rotate(180)
        floor_r270 = floor.rotate(270)
        support_r90 = support.rotate(90)
        support_r180 = support.rotate(180)
        support_r270 = support.rotate(270)
        im_as_np_array[counter+1] = np.append(np.array(floor_r90), np.array(support_r90)).reshape((2,) + DIM)
        im_as_np_array[counter+2] = np.append(np.array(floor_r180), np.array(support_r180)).reshape((2,) + DIM)
        im_as_np_array[counter+3] = np.append(np.array(floor_r270), np.array(support_r270)).reshape((2,) + DIM)
        # flip images horizontally (mirror()) and vertically (flip())
        im_as_np_array[counter+4] = np.append(np.array(ImageOps.flip(floor)), np.array(ImageOps.flip(support))).reshape((2,) + DIM)
        im_as_np_array[counter+5] = np.append(np.array(ImageOps.mirror(floor_r90)), np.array(ImageOps.mirror(support_r90))).reshape((2,) + DIM)
        im_as_np_array[counter+6] = np.append(np.array(ImageOps.mirror(floor_r180)), np.array(ImageOps.mirror(support_r180))).reshape((2,) + DIM)
        im_as_np_array[counter+7] = np.append(np.array(ImageOps.mirror(floor_r270)), np.array(ImageOps.mirror(support_r270))).reshape((2,) + DIM)
        counter += 8
    im_as_np_array /= np.amax(im_as_np_array)
    print(f"Test data images min: {np.amin(im_as_np_array)} max: {np.amax(im_as_np_array)}")
    test_imgs = torch.FloatTensor(im_as_np_array).cuda()
    # print(f"test_imgs.shape={test_imgs.shape} dtype={test_imgs.dtype}")
    im_as_np_array = None
    print("Done.")

    time_now = datetime.now() # used to time how long test/training samples, network are uploaded and Specs.txt is created

    # start training
    print("Starting training")

    # https://stackoverflow.com/questions/51520143/update-matplotlib-image-in-a-function
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    im = ax1.imshow(np.zeros(DIM+(3,)))
    im2 = ax3.imshow(np.zeros(DIM+(3,)))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    progress_loss_G = []
    progress_loss_D = []

    for epoch in range(EPOCHS):      
        # train discriminator D
        D.zero_grad()

        indices = torch.randint(0, NUM_TRAIN_DATA * 8, (MINI_BATCH,)).cuda()
        # print(f"indices.shape={indices.shape}")
        test_imgs_batch = torch.index_select(test_imgs, 0, indices)
        floor_input = test_imgs_batch[:,0,:,:].reshape((MINI_BATCH,1)+DIM)
        # train_imgs_batch = test_imgs[:,0:1,:,:]
        # print(f"test_imgs_batch.shape={test_imgs_batch.shape}")
        # print(f"floor plans input: {floor_input.shape}")

        D_res = D.forward(test_imgs_batch)
        # print(f"D_res.shape={D_res.shape}")
        # print(f"D_res={D_res}")
        # print("ones:", torch.ones_like(D_res, requires_grad=True).cuda())

        D_real_loss = D.loss_function(D_res, torch.ones_like(D_res, requires_grad=True).cuda())
        # print(f"D_real_loss.shape={D_real_loss.shape}")

        G_res = G.forward(floor_input)
        # print(f"G_res.shape={G_res.shape}")
        generated_imgs = torch.cat((floor_input, G_res), 1)
        # print(f"generated_imgs.shape={generated_imgs.shape}")

        D_res = D.forward(generated_imgs)
        # print(f"D_res.shape={D_res.shape}")
        D_fake_loss = D.loss_function(D_res, torch.zeros_like(D_res, requires_grad=True).cuda()).cuda()

        D_train_loss = (D_fake_loss + D_real_loss) * 0.5
        D_train_loss.backward()
        #if epoch < 20 or True:
        D.optimizer.step()

        # train generator G
        G.zero_grad()
        D.zero_grad()

        G_res = G.forward(floor_input)
        D_res = D.forward(torch.cat((floor_input, G_res), 1))

        G_train_loss = D.loss_function(D_res, torch.ones_like(D_res, requires_grad=True).cuda()).cuda()
        G_train_loss.backward()
        G.optimizer.step()

        progress_loss_G.append(G_train_loss.item())
        progress_loss_D.append(D_train_loss.item())

        if epoch % 1 == 0:
            image = np.zeros(DIM+(3,))
            image[:,:,0] = floor_input[0, 0].cpu().detach().numpy()
            image[:,:,1] = G_res[0, 0].cpu().detach().numpy()
            im.set_data(image)

            input_img = train_imgs[torch.randint(0, train_imgs.shape[0], (1,))]
            image[:,:,0] = input_img.cpu().detach().numpy()
            image[:,:,1] = G.forward(input_img.reshape((1,1)+DIM).cuda())[0][0].cpu().detach().numpy()
            im2.set_data(image)
            
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