import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dts

# img = mpimg.imread("Testdaten/testdata/original/1.png")
# img = img[:,:,0]
# print(f"image size: {img.shape}")
# print(f"image parameter count: {img.shape[0] * img.shape[1]}")

print("upload training data to gpu.... ")
images = dts.FashionMNIST("Testdaten/fashionmnist", download=True)
imgs = torch.FloatTensor(np.zeros((len(images),images[0][0].height,images[0][0].width), dtype=float)).to('cuda')
for i in range(len(images)):
    imgs[i] = torch.cuda.FloatTensor(np.array(images[i][0]).astype(float) / 256.0)
print("done.")

def main():
    D = Discriminator()
    D.to('cuda')
    G = Generator()
    G.to('cuda')

    # https://stackoverflow.com/questions/51520143/update-matplotlib-image-in-a-function
    fig, (ax1,ax2) = plt.subplots(1,2)
    im = ax1.imshow(imgs[0].cpu().numpy(), cmap='Greys')

    num_iter = 20000
    for i in range(num_iter):
        if i % 1000 == 0:
            print(f"{i / num_iter * 100}%")
        # img = imgs[np.random.randint(len(images))]
        img = imgs[0]
        D.train(img, torch.cuda.FloatTensor([1.0]))
        D.train(G.forward(torch.cuda.FloatTensor(10).normal_()).detach(), torch.cuda.FloatTensor([0.0]))
        G.train(D, torch.cuda.FloatTensor(10).normal_(), torch.cuda.FloatTensor([1.0]))
        G.train(D, torch.cuda.FloatTensor(10).normal_(), torch.cuda.FloatTensor([1.0]))

        if i % 100 == 0:
            im.set_data(G.forward(torch.cuda.FloatTensor(10).normal_()).detach().cpu().numpy().reshape(imgs[0].shape))
            ax2.clear()
            ax2.plot(np.arange(len(D.progress)) / len(D.progress) * i , D.progress, label="D")
            ax2.plot(np.arange(len(G.progress)) / len(G.progress) * i, G.progress, label="G")
            ax2.legend()
            fig.canvas.draw_idle()
            plt.pause(0.01)

    plt.plot(np.arange(len(D.progress)) / len(D.progress) , D.progress, label="D")
    plt.plot(np.arange(len(G.progress)) / len(G.progress), G.progress, label="G")
    plt.legend()
    plt.show()
    plt.imshow(G.forward(torch.cuda.FloatTensor(10).normal_()).detach().cpu().numpy().reshape(imgs[0].shape))
    plt.show()
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(imgs.shape[1] * imgs.shape[2], 3*10*10),
            nn.LeakyReLU(),
            nn.Linear(3*10*10, 64),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        ).cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs.flatten())
    
    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 30),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(30, 150),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(150, 375),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(375,imgs.shape[1] * imgs.shape[2]),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs.flatten())

    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)
        d_output = D.forward(g_output)
        loss = D.loss_function(d_output, targets)
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

main()