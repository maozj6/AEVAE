
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
​
​
# class Decoder(nn.Module):
#     """ VAE decoder """
#     def __init__(self,img_channels, latent_size ):
#         super(Decoder, self).__init__()
#         self.latent_size = latent_size
#         self.img_channels = img_channels
#         # self.img_size = img_size
#
#         self.fc1 = nn.Linear(latent_size, 128)
#         self.fc2 = nn.Linear(128, 8192)  # Adjust the output size based on your image size
#         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(32, img_channels, kernel_size=3, padding=1)
#
#     def forward(self, z):
#         x = F.relu(self.fc1(z))
#         x = F.relu(self.fc2(x))
#         x = x.view(x.size(0), 128, 8, 8)  # Reshape to match the last convolutional layer's output
#         x = F.relu(self.deconv1(x))
#         x = F.interpolate(x, scale_factor=2, mode='nearest')  # Upsample
#         x = F.relu(self.deconv2(x))
#         x = F.interpolate(x, scale_factor=2, mode='nearest')  # Upsample
#         x = torch.sigmoid(self.deconv3(x))  # Sigmoid activation for pixel values between 0 and 1
#
#         return x
​
class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self,img_channels, latent_size ):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
​
        self.fc1 = nn.Linear(latent_size, 8192)  # Adjust the output size based on your image size
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1)
​
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(x.size(0), 512, 4, 4)  # Reshape to match the last convolutional layer's output
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Sigmoid activation for pixel values between 0 and 1
​
        return x
​
class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels
​
        self.conv1 = nn.Conv2d(img_channels, 32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, 256, 4, stride=1)
        self.fcmu1 = nn.Linear(8192, 128)  # Adjust the input size based on your image size
        self.fclog1 = nn.Linear(8192, 128)  # Adjust the input size based on your image size
        self.fc_mu2 = nn.Linear(128, latent_size)
        self.fc_logsigma = nn.Linear(128, latent_size)
​
​
    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
​
        mu = self.fc_mu2(self.fcmu1(x))
        logsigma = self.fc_logsigma(self.fclog1(x))
​
        return mu, logsigma
​
class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)
​
        # self.conv1 = nn.Conv2d(1, 6, 3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(2704, 1024)
        # self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(256,2)
    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
​
        recon_x = self.decoder(z)
​
        # input_size = recon_x.size(0)
        #
        # x = self.pool(F.relu(self.conv1(recon_x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(input_size,-1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return recon_x, mu, logsigma
