# MIT License
# 
# Copyright (c) 2019 Yisroel Mirsky
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from config import *  # user configurations
import torch
import torch.nn as nn
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
from utils.equalizer import *
import pickle
import numpy as np
import time
import scipy.ndimage
from utils.dicom_utils import *
from utils.utils import *

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
    
def load_model(model_path):
    """Helper function to load PyTorch model"""
    try:
        # Define input shape
        channels = 1
        img_depth = config['cube_shape'][0]
        img_rows = config['cube_shape'][1]
        img_cols = config['cube_shape'][2]
        img_shape = (channels, img_depth, img_rows, img_cols)
        
        # Create model instance
        model = UNetGenerator(img_shape)
        
        # Load state dict
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # If the state dict was saved from DataParallel wrapper
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error in load_model: {e}")
        raise e


class scan_manipulator:
    def __init__(self):
        print("===Init Tamperer===")
        self.scan = None
        self.load_path = None
        # self.m_zlims = config['mask_zlims']
        # self.m_ylims = config['mask_ylims']
        # self.m_xlims = config['mask_xlims']
        self.m_zlims,self.m_ylims,self.m_xlims = generate_random_mask(config['cube_shape'], config['mask_size'])
    
        # Load model and parameters
        self.model_inj_path = config['modelpath_inject']
        self.model_rem_path = config['modelpath_remove']

        # Load models
        print("Loading models")
        try:
            # print(os.path.join(self.model_inj_path, "G_model.pth"))
            if os.path.exists(os.path.join(self.model_inj_path, "G_model.pth")):
                self.generator_inj = load_model(os.path.join(self.model_inj_path, "G_model.pth"))
                self.eq_inj = histEq([], path=os.path.join(self.model_inj_path, 'equalization.pkl'))
                try:
                    self.norm_inj = np.load(os.path.join(self.model_inj_path, 'normalization.npy'))
                    if self.norm_inj.size == 0:
                        raise ValueError("Loaded normalization.npy is empty.")
                except Exception as e:
                    print(f"Error loading normalization.npy: {e}")
                    self.norm_inj = np.array([1])  # Fallback default
                print("Loaded Injector Model")
            else:
                self.generator_inj = None
                import traceback
                traceback.print_exc()
                print("Failed to Load Injector Model")

            if os.path.exists(os.path.join(self.model_rem_path, "G_model.pth")):
                self.generator_rem = load_model(os.path.join(self.model_rem_path, "G_model.pth"))
                self.norm_rem = np.load(os.path.join(self.model_rem_path, 'normalization.npy'))
                self.eq_rem = histEq([], path=os.path.join(self.model_rem_path, 'equalization.pkl'))
                print("Loaded Remover Model")
            else:
                self.generator_rem = None
                print("Failed to Load Remover Model")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.generator_inj = None
            self.generator_rem = None

    def load_target_scan(self, load_path):
        try:
            self.load_path = load_path
            print('Loading scan')
            self.scan, self.scan_spacing, self.scan_orientation, self.scan_origin, self.scan_raw_slices = load_scan(load_path)
            self.scan = self.scan.astype(float)
        except Exception as e:
            print(f"Error loading target scan: {e}")
            self.scan = None

    def save_tampered_scan(self, save_dir, output_type='dicom'):
        if self.scan is None:
            print('Cannot save: load a target scan first.')
            return

        print('Saving scan')
        try:
            if output_type == 'dicom':
                if self.load_path.split('.')[-1] == "mhd":
                    toDicom(save_dir=save_dir, img_array=self.scan, pixel_spacing=self.scan_spacing, orientation=self.scan_orientation)
                else:  # input was dicom
                    save_dicom(self.scan, original_raw_slices=self.scan_raw_slices, dst_directory=save_dir)
                    print("Saved in ",save_dir)
            else:  # save as numpy
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, 'tampered_scan.np'), self.scan)
            print('Done.')
        except Exception as e:
            print(f"Error saving tampered scan: {e}")

    def tamper(self, coord, action="inject", isVox=True):
        if self.scan is None:
            print('Cannot tamper: load a target scan first.')
            return
        if (action == 'inject') and (self.generator_inj is None):
            print('Cannot inject: no injection model loaded.')
            return
        if (action == 'remove') and (self.generator_rem is None):
            print('Cannot inject: no removal model loaded.')
            return

        if action == 'inject':
            print('===Injecting Evidence===')
        else:
            print('===Removing Evidence===')
        try:
            if not isVox:
                print("Converting Vox wooofff")
                coord = world2vox(coord, self.scan_spacing, self.scan_orientation, self.scan_origin)
            # print("Hurray no need to convert to Vox !..already in vox")
            ### Cut Location
            print("Cutting out target region")
            cube_shape = get_scaled_shape(config["cube_shape"], 1 / self.scan_spacing)
            clean_cube_unscaled = cutCube(self.scan, coord, cube_shape)
            clean_cube, resize_factor = scale_scan(clean_cube_unscaled, self.scan_spacing)
            sdim = int(np.max(cube_shape) * 1.3)
            clean_cube_unscaled2 = cutCube(self.scan, coord, np.array([sdim, sdim, sdim]))  # for noise touch ups later

            ### Normalize/Equalize Location
            print("Normalizing sample")
            if action == 'inject':
                clean_cube_eq = self.eq_inj.equalize(clean_cube)
                clean_cube_norm = (clean_cube_eq - self.norm_inj[0]) / ((self.norm_inj[2] - self.norm_inj[1]))
            else:
                clean_cube_eq = self.eq_rem.equalize(clean_cube)
                clean_cube_norm = (clean_cube_eq - self.norm_rem[0]) / ((self.norm_rem[2] - self.norm_rem[1]))

            ### Inject/Remove evidence
            if action == 'inject':
                print("Injecting evidence")
            else:
                print("Removing evidence")

            x = np.copy(clean_cube_norm)
            x[self.m_zlims[0]:self.m_zlims[1], self.m_xlims[0]:self.m_xlims[1], self.m_ylims[0]:self.m_ylims[1]] = 0
            
            # print("Shape of clean_cube_unscaled:", clean_cube_unscaled.shape)
            # print("Shape of clean_cube after scaling:", clean_cube.shape)
            # print("Shape of clean_cube_norm:", clean_cube_norm.shape)
            # print(f"Shape of clean_cube_norm: {clean_cube_norm.shape}")
            # print(f"Coord: {coord}, Scan shape: {self.scan.shape}")
            # print(f"cube shape {cube_shape}")
            # print(x.shape)


            # Reshape to match PyTorch's expected format: (batch_size, channels, depth, height, width)
            x = x.reshape((1, *config['cube_shape'], 1))  # First reshape to include batch and channel dims
            x = np.transpose(x, (0, 4, 1, 2, 3))  # Transpose to get channels after batch dim
            
            # Convert to PyTorch tensor
            x_tensor = torch.FloatTensor(x)
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()

            # Run prediction
            with torch.no_grad():
                if action == 'inject':
                    x_mal = self.generator_inj(x_tensor)
                else:
                    x_mal = self.generator_rem(x_tensor)
                
                # Convert back to numpy and reshape to original format
                x_mal = x_mal.cpu().numpy()
                x_mal = np.transpose(x_mal, (0, 2, 3, 4, 1))  # Move channels to last dimension
                x_mal = x_mal.reshape(config['cube_shape'])  # Remove batch and channel dimensions

            ### De-Norm/De-equalize
            print("De-normalizing sample")
            x_mal[x_mal > .5] = .5  # fix boundary overflow
            x_mal[x_mal < -.5] = -.5
            if action == 'inject':
                mal_cube_eq = x_mal * ((self.norm_inj[2] - self.norm_inj[1])) + self.norm_inj[0]
                mal_cube = self.eq_inj.dequalize(mal_cube_eq)
            else:
                mal_cube_eq = x_mal * ((self.norm_rem[2] - self.norm_rem[1])) + self.norm_rem[0]
                mal_cube = self.eq_rem.dequalize(mal_cube_eq)

            # Rest of the function remains the same...
            # Correct for pixel norm error
            bad = np.where(mal_cube > 2000)
            for i in range(len(bad[0])):
                neiborhood = cutCube(mal_cube, np.array([bad[0][i], bad[1][i], bad[2][i]]), (np.ones(3) * 5).astype(int), -1000)
                mal_cube[bad[0][i], bad[1][i], bad[2][i]] = np.mean(neiborhood)
            mal_cube[mal_cube < -1000] = -1000

            ### Paste Location
            print("Pasting sample into scan")
            mal_cube_scaled, resize_factor = scale_scan(mal_cube, 1 / self.scan_spacing)
            self.scan = pasteCube(self.scan, mal_cube_scaled, coord)

            ### Noise Touch-ups
            print("Adding noise touch-ups...")
        
            noise_map_dim = clean_cube_unscaled2.shape
            ben_cube_ext = clean_cube_unscaled2
            mal_cube_ext = cutCube(self.scan, coord, noise_map_dim)
            local_sample = clean_cube_unscaled

            # Init Touch-ups
            if action == 'inject':  # inject type
                if len(local_sample[local_sample < -600]) > 0:
                    noisemap = np.random.randn(150, 200, 300) * np.nanstd(local_sample[local_sample < -600]) * .6
                else:
                    noisemap = np.random.randn(150, 200, 300) * 100  # Fallback value
                
                kernel_size = 3
                factors = sigmoid((mal_cube_ext + 700) / 70)
                k = kern01(mal_cube_ext.shape[0], kernel_size)
                for i in range(factors.shape[0]):
                    factors[i, :, :] = factors[i, :, :] * k
            else:  # remove type
                noisemap = np.random.randn(150, 200, 200) * 30
                kernel_size = .1
                k = kern01(mal_cube_ext.shape[0], kernel_size)
                factors = None

            # Perform touch-ups
            if config['copynoise']:
                benm = cutCube(self.scan, np.array([int(self.scan.shape[0] / 2), int(self.scan.shape[1] * .43), int(self.scan.shape[2] * .27)]), noise_map_dim)
                x = np.copy(benm)
                if np.any(x < -800):  
                    x[x > -800] = np.nanmean(x[x < -800])  
                else:  
                    print("Warning: No values satisfy x < -800. Defaulting to 0.")  
                    x[x > -800] = 0  # Or any other appropriate default value  

                noise = x - np.nanmean(x)
            else:
                rf = np.ones((3,)) * (60 / np.std(local_sample[local_sample < -600])) * 1.3
                np.random.seed(np.int64(time.time()))
                noisemap_s = scipy.ndimage.interpolation.zoom(noisemap, rf, mode='nearest')
                noise = noisemap_s[:noise_map_dim, :noise_map_dim, :noise_map_dim]
            mal_cube_ext += noise

            if action == 'inject':  # Injection
                final_cube_s = np.maximum((mal_cube_ext * factors + ben_cube_ext * (1 - factors)), ben_cube_ext)
            else:  # Removal
                minv = np.min((np.min(mal_cube_ext), np.min(ben_cube_ext)))
                final_cube_s = (mal_cube_ext + minv) * k + (ben_cube_ext + minv) * (1 - k) - minv

            self.scan = pasteCube(self.scan, final_cube_s, coord)
            print('touch-ups complete')
        except Exception as e:
            print(f"Error during tampering: {e}")
            import traceback
            traceback.print_exc()
            exit(0)
