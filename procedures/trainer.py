from __future__ import print_function, division
from config import *  # user configuration in config.py
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from utils.dataloader import DataLoader


class Trainer:
    def __init__(self, isInjector=True):
        # cli_args = os.environ.get("CLI_ARGS", "").split()
        self.isInjector = isInjector
        # Input shape
        cube_shape = config['cube_shape']
        self.img_rows = config['cube_shape'][1]
        self.img_cols = config['cube_shape'][2]
        self.img_depth = config['cube_shape'][0]
        self.channels = 1
        self.num_classes = 5
        self.img_shape = (self.channels, self.img_depth, self.img_rows, self.img_cols)  # PyTorch uses channels first

        # Configure data loader
        if self.isInjector:
            self.dataset_path = config['unhealthy_samples']
            self.modelpath = config['modelpath_inject']
        else:
            self.dataset_path = config['healthy_samples']
            self.modelpath = config['modelpath_remove']

        self.dataloader = DataLoader(dataset_path=self.dataset_path, normdata_path=self.modelpath,
                                    img_res=(self.img_rows, self.img_cols, self.img_depth))

        # Number of filters in the first layer of G and D
        self.gf = 100
        self.df = 100

        # Set device
        self.device = torch.device("cuda:{}".format(config['gpus']) if torch.cuda.is_available() else "cpu")
        print("Using Device: ",self.device)
        # self.device='cpu'
        # Build generator and discriminator
        self.generator = self.build_generator().to(self.device)
        self.discriminator = self.build_discriminator().to(self.device)
        print(self.generator)
        print(self.discriminator)

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


        # Loss functions
        # self.criterion_GAN = nn.HuberLoss(delta=1.0)
        self.criterion_GAN = nn.MSELoss()  # Mean Squared Error Loss

        self.criterion_pixelwise = nn.L1Loss()

        # Loss weights
        self.lambda_pixel = 100

        # Get discriminator output shape for patch calculation
        dummy_A = torch.zeros((1, *self.img_shape)).to(self.device)
        dummy_B = torch.zeros((1, *self.img_shape)).to(self.device)
        with torch.no_grad():
            dummy_output = self.discriminator(dummy_A, dummy_B)
            self.disc_patch = dummy_output.shape[2:]  # Channels, D, H, W
            print("Value: ",self.disc_patch)
            print(f"Discriminator output shape: {dummy_output.shape}")

    def build_generator(self):
        """U-Net Generator"""
        return UNetGenerator(self.img_shape, self.gf)

    def build_discriminator(self):
        """PatchGAN Discriminator"""
        return PatchGANDiscriminator(self.img_shape, self.df)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            # save model
            if epoch%10 == 0:
                print("Saving Models...")
                torch.save(self.generator.state_dict(), os.path.join(self.modelpath, "G_model.pth"))
                torch.save(self.discriminator.state_dict(), os.path.join(self.modelpath, "D_model.pth"))

            for batch_i, (imgs_A, imgs_B) in enumerate(self.dataloader.load_batch(batch_size)):
                # Convert numpy arrays to PyTorch tensors
                real_A = torch.from_numpy(imgs_A).float().to(self.device)
                real_B = torch.from_numpy(imgs_B).float().to(self.device)
                
                # Reshape for PyTorch (N, C, D, H, W)
                real_A = real_A.permute(0, 4, 3, 1, 2)
                real_B = real_B.permute(0, 4, 3, 1, 2)

                # Adversarial ground truths (Corrected)
                valid = torch.zeros((batch_size, 1, *self.disc_patch), requires_grad=False).to(self.device)  # Real images
                fake = torch.ones((batch_size, 1, *self.disc_patch), requires_grad=False).to(self.device)  # Fake images

                # ---------------------
                #  Train Generator
                # ---------------------
                self.optimizer_G.zero_grad()

                # Generate fake images
                fake_A = self.generator(real_B)

                # Adversarial loss
                pred_fake = self.discriminator(fake_A, real_B)
                loss_GAN = self.criterion_GAN(pred_fake, valid)
                
                # Pixel-wise loss
                loss_pixel = self.criterion_pixelwise(fake_A, real_A)

                # Total generator loss
                loss_G = loss_GAN + self.lambda_pixel * loss_pixel
                loss_G.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                # Real loss
                pred_real = self.discriminator(real_A, real_B)
                loss_real = self.criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = self.discriminator(fake_A.detach(), real_B)
                loss_fake = self.criterion_GAN(pred_fake, fake)

                # Total discriminator loss
                loss_D = 0.5 * (loss_real + loss_fake)
                loss_D.backward()
                self.optimizer_D.step()

                # Calculate discriminator accuracy - ensure shapes match first
                pred_real_flat = pred_real.ge(0.5).cpu().detach().numpy().reshape(-1)
                valid_flat = valid.cpu().detach().numpy().reshape(-1)
                min_len = min(len(pred_real_flat), len(valid_flat))
                acc = np.mean(np.equal(pred_real_flat[:min_len], valid_flat[:min_len])) * 100

                # Update progress
                elapsed_time = datetime.datetime.now() - start_time
                
                if batch_i%20 == 0:
                    print(f"\033[1;32m[Epoch {epoch}/{epochs}]\033[0m "  # Green for epoch
                        f"\033[1;34m[Batch {batch_i}/{self.dataloader.n_batches}]\033[0m "  # Blue for batch
                        f"\033[1;31m[D loss: {loss_D.item():.6f}]\033[0m "  # Red for discriminator loss
                        f"\033[1;33m[Acc: {acc:.2f}%]\033[0m "  # Yellow for accuracy
                        f"\033[1;35m[G loss: {loss_G.item():.6f}]\033[0m "  # Magenta for generator loss
                        f"\033[1;36m[Time: {elapsed_time}]\033[0m")  # Cyan for elapsed time

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.show_progress(epoch, batch_i)

    def show_progress(self, epoch, batch_i):
        filename = f"{epoch}_{batch_i}.png"
        if self.isInjector:
            savepath = os.path.join(config['progress'], "injector")
        else:
            savepath = os.path.join(config['progress'], "remover")
        os.makedirs(savepath, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.dataloader.load_data(batch_size=3, is_testing=True)
        
        # Convert to PyTorch tensors
        real_A = torch.from_numpy(imgs_A).float().to(self.device)
        real_B = torch.from_numpy(imgs_B).float().to(self.device)
        
        # Reshape for PyTorch (N, C, D, H, W)
        real_A = real_A.permute(0, 4, 3, 1, 2)
        real_B = real_B.permute(0, 4, 3, 1, 2)
        
        # Generate fake images
        with torch.no_grad():
            fake_A = self.generator(real_B)

        # Convert back to numpy for visualization
        real_A = real_A.cpu().numpy().transpose(0, 3, 4, 2, 1)  # (N, H, W, D, C)
        real_B = real_B.cpu().numpy().transpose(0, 3, 4, 2, 1)
        fake_A = fake_A.cpu().numpy().transpose(0, 3, 4, 2, 1)

        gen_imgs = np.concatenate([real_B, fake_A, real_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                # Reshape for visualization
                img_slice = gen_imgs[cnt].reshape((self.img_rows, self.img_cols, self.img_depth, self.channels))
                axs[i, j].imshow(img_slice[:, :, int(self.img_depth/2), 0])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(savepath, filename))
        plt.close()


# Define the U-Net Generator
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels)
        )
    
    def forward(self, x):
        return x + self.conv(x)  # Residual connection

class UNetGenerator(nn.Module):
    def __init__(self, img_shape, gf=100):
        super(UNetGenerator, self).__init__()
        channels, d, h, w = img_shape
        self.channels = channels
        
        cli_args = os.environ.get("CLI_ARGS", "").split()
        self.flag = 'b'
        if cli_args[0] == 'l':
            self.flag = 'l'
        else:
            self.flag='b'
        
        if self.flag=='b':
        # Encoder (downsampling)
            self.down1 = nn.Sequential(
                nn.Conv3d(channels, gf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(gf)  # Add residual block
            )
            self.down2 = nn.Sequential(
                nn.Conv3d(gf, gf * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 2, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(gf * 2)  # Add residual block
            )
            self.down3 = nn.Sequential(
                nn.Conv3d(gf * 2, gf * 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 4, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(gf * 4)  # Add residual block
            )
            self.down4 = nn.Sequential(
                nn.Conv3d(gf * 4, gf * 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 8, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(gf * 8)  # Add residual block
            )
            self.down5 = nn.Sequential(
                nn.Conv3d(gf * 8, gf * 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 8, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(gf * 8)  # Add residual block
            )
            
            # Decoder (upsampling)
            self.up1 = nn.Sequential(
                nn.ConvTranspose3d(gf * 8, gf * 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 8, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                ResidualBlock(gf * 8)  # Add residual block
            )
            self.up2 = nn.Sequential(
                nn.ConvTranspose3d(gf * 16, gf * 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 4, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                ResidualBlock(gf * 4)  # Add residual block
            )
            self.up3 = nn.Sequential(
                nn.ConvTranspose3d(gf * 8, gf * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 2, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                ResidualBlock(gf * 2)  # Add residual block
            )
            self.up4 = nn.Sequential(
                nn.ConvTranspose3d(gf * 4, gf, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(gf)  # Add residual block
            )
        else:
            self.down1 = nn.Sequential(
                nn.Conv3d(channels, gf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # ResidualBlock(gf)  # Add residual block
            )
            self.down2 = nn.Sequential(
                nn.Conv3d(gf, gf * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 2, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                # ResidualBlock(gf * 2)  # Add residual block
            )
            self.down3 = nn.Sequential(
                nn.Conv3d(gf * 2, gf * 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 4, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                # ResidualBlock(gf * 4)  # Add residual block
            )
            self.down4 = nn.Sequential(
                nn.Conv3d(gf * 4, gf * 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 8, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                # ResidualBlock(gf * 8)  # Add residual block
            )
            self.down5 = nn.Sequential(
                nn.Conv3d(gf * 8, gf * 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 8, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                # ResidualBlock(gf * 8)  # Add residual block
            )
            
            # Decoder (upsampling)
            self.up1 = nn.Sequential(
                nn.ConvTranspose3d(gf * 8, gf * 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 8, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                # ResidualBlock(gf * 8)  # Add residual block
            )
            self.up2 = nn.Sequential(
                nn.ConvTranspose3d(gf * 16, gf * 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 4, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                # ResidualBlock(gf * 4)  # Add residual block
            )
            self.up3 = nn.Sequential(
                nn.ConvTranspose3d(gf * 8, gf * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf * 2, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
                # ResidualBlock(gf * 2)  # Add residual block
            )
            self.up4 = nn.Sequential(
                nn.ConvTranspose3d(gf * 4, gf, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(gf, momentum=0.8),
                nn.LeakyReLU(0.2, inplace=True),
                # ResidualBlock(gf)  # Add residual block
            )
        self.final = nn.Sequential(
            nn.ConvTranspose3d(gf * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        # Decoder with skip connections
        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        
        return self.final(u4)

# Define the PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, img_shape, df):
        super(PatchGANDiscriminator, self).__init__()
        channels, d, h, w = img_shape
        cli_args = os.environ.get("CLI_ARGS", "").split()
        self.flag = 'b'
        if cli_args[0] == 'l':
            self.flag = 'l'
        else:
            self.flag='b'
        
        # Calculate the expected output shape
        self.patch_d = d // 2**4
        self.patch_h = h // 2**4
        self.patch_w = w // 2**4
        
        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bn=True):
            block=None
            if self.flag=='b':
                block = [spectral_norm(nn.Conv3d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding)),
                        nn.LeakyReLU(0.2, inplace=True)]
            else:
                block = [nn.Conv3d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2, inplace=True)]
                
            if bn:
                block.append(nn.BatchNorm3d(out_filters, momentum=0.8))
            return block
        
        if self.flag == 'b':
            self.model = nn.Sequential(
                *discriminator_block(channels * 2, df, bn=False),
                *discriminator_block(df, df * 2),
                *discriminator_block(df * 2, df * 4),
                *discriminator_block(df * 4, df * 8),
                spectral_norm(nn.Conv3d(df * 8, 1, kernel_size=4, stride=1, padding=1))

            )
        else:
            self.model = nn.Sequential(
                *discriminator_block(channels * 2, df, bn=False),
                *discriminator_block(df, df * 2),
                *discriminator_block(df * 2, df * 4),
                *discriminator_block(df * 4, df * 8),
                nn.Conv3d(df * 8, 1, kernel_size=4, stride=1, padding=1))

            
            
        
    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), dim=1)
        output = self.model(img_input)
        
        # Ensure output has expected shape
        if output.shape[2:] != (self.patch_d, self.patch_h, self.patch_w):
            output = F.interpolate(output, size=(self.patch_d, self.patch_h, self.patch_w), 
                                   mode='trilinear', align_corners=False)
            
        return output