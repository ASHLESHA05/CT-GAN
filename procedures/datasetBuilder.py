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

from config import *
import pandas as pd
import multiprocessing
from scipy.ndimage.interpolation import rotate
from joblib import Parallel, delayed
import itertools
from utils.equalizer import *
from utils.dicom_utils import *
from utils.utils import *

class Extractor:
    #is_healthy_dataset: indicates if the datset are of healthy scans or of unhealthy scans
    #src_dir: Path to directory containing all of the scans (folders of dicom series, or mhd/raw files)
    #dst_path: Path to file to save the dataset in np serialized format. e.g., data/healthy_samples.npy
    #norm_save_dir: the directory where the normalization parameters should be saved. e.g, data/models/INJ/
    #coords_csv: path to csv of the candidate locations with the header:   filename, z, x, y  (if vox, slice should be 0-indexed)
    #   if filename is a directory or has a *.dcm extension, then dicom format is assumed (each scan should have its own directory contianting all of its dcm slices)
    #   if filename has the *.mhd extension, then mhd/raw is assumed (all mdh/raw files should be in same directory)
    #parallelize: inidates whether the processign should be run over multiple CPU cores
    #coordSystem: if the coords are the matrix indexes, then choose 'vox'. If the coords are realworld locations, then choose 'world'
    def __init__(self, is_healthy_dataset, src_dir=None, coords_csv_path=None, dst_path=None, norm_save_dir=None, parallelize=False, coordSystem=None):
        self.parallelize = parallelize
        if coordSystem is None:
            self.coordSystem = config['traindata_coordSystem']
        else:
            self.coordSystem = coordSystem
        if is_healthy_dataset:
            self.src_dir = src_dir if src_dir is not None else config['healthy_scans_raw']
            self.dst_path = dst_path if dst_path is not None else config['healthy_samples']
            self.norm_save_dir = norm_save_dir if norm_save_dir is not None else config['modelpath_remove']
            self.coords = pd.read_csv(coords_csv_path) if coords_csv_path is not None else pd.read_csv(config['healthy_coords'])
        else:
            self.src_dir = src_dir if src_dir is not None else config['unhealthy_scans_raw']
            self.dst_path = dst_path if dst_path is not None else config['unhealthy_samples']
            self.norm_save_dir = norm_save_dir if norm_save_dir is not None else config['modelpath_inject']
            self.coords = pd.read_csv(coords_csv_path) if coords_csv_path is not None else pd.read_csv(config['unhealthy_coords'])

    def extract(self,plot=True):
        # Prep jobs (one per coordinate)
        print("preparing jobs...")
        J = [] #jobs
        for i, sample in self.coords.iterrows():
            coord = np.array([sample.z, sample.y, sample.x])
            if not pd.isnull(sample.z):
                #job: (path to scan, coordinate, instance shape, coord system 'vox' or 'world')
                J.append([os.path.join(self.src_dir,sample.filename), coord, config['cube_shape'], self.coordSystem])

        print("extracting and augmenting samples...")
        try:
            if self.parallelize:
                num_cores = int(np.ceil(min(np.ceil(multiprocessing.cpu_count() * 0.75), len(J))))
                X = Parallel(n_jobs=num_cores)(delayed(self._processJob)(j) for j in J)
            else:
                X = []
                for job in J:
                    try:
                        X.append(self._processJob(job))
                    except Exception as e:
                        print("Failed to process sample: ",str(e))
                        exit(0)
            instances = np.array(list(itertools.chain.from_iterable(X))) #each job creates a batch of augmented instances: so collect hem
        except Exception as e:
            print("error: ",e)
            import traceback
            traceback.print_exc()
        # Histogram Equalization:
        print("equalizing the data...")
        eq = histEq(instances)
        instances = eq.equalize(instances)
        os.makedirs(self.norm_save_dir,exist_ok=True)
        eq.save(path=os.path.join(self.norm_save_dir,'equalization.pkl'))

        # -1 1 Normalization
        print("normalizing the data...")
        min_v = np.min(instances)
        max_v = np.max(instances)
        mean_v = np.mean(instances)
        norm_data = np.array([mean_v, min_v, max_v])
        instances = (instances - mean_v) / (max_v - min_v)
        np.save(os.path.join(self.norm_save_dir,'normalization.npy'),norm_data)

        if plot:
            self.plot_sample(instances)

        print("saving the dataset")
        np.save(self.dst_path,instances)

    def _processJob(self,args):
        # print("Working on job: " + args[0] + "   "+args[3]+" coord (zyx): ", args[1], "cube shape",args[2])
        # eg args[0]:  D:/CapstoneProject/DataSet/LungsCT-BigData/manifest-1600709154662/LIDC-IDRI\LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192
        instances = self._get_instances_from_scan(scan_path=args[0], coord=args[1], cube_shape=args[2], coordSystem=args[3])
        return instances

    def _get_instances_from_scan(self, scan_path, coord, cube_shape, coordSystem):
        # load scan data
        try:
            scan, spacing, orientation, origin, raw_slices = load_scan(scan_path)
            # scale the image
            #scan_resized, resize_factor = scale_scan(scan, spacing)
            # compute sample coords as vox
            if coordSystem == 'world': #convert from world to vox
                coord = world2vox(coord,spacing,orientation,origin)
            elif coordSystem != 'vox':
                raise Exception("Coordinate conversion error: you can only select world or vox")
            #coordn = scale_vox_coord(coord, spacing)  # ccord relative to scaled scan

            # extract instances
            # print("Extracting Instance")
            X = []
            init_cube_shape = get_scaled_shape(cube_shape + 8, 1/spacing)
            
            #print("CutCubing")
            
            #print(f"Scan Shape{scan.shape} :: coord: {coord} :: init_Cube_shape_scaled: {init_cube_shape}")
            clean_cube_unscaled = cutCube(scan, coord, init_cube_shape, padd=-1000)
            #print("Scaling",clean_cube_unscaled)
            # print("Scaling shape",clean_cube_unscaled.shape)
            x, resize_factor = scale_scan(clean_cube_unscaled,spacing)
            # perform data augmentations to generate more instances
            Xaug = self._augmentInstance(x)
            # trim the borders to get the actual desired shape
            for xa in Xaug:
                center = np.array(x.shape)//2
                X.append(cutCube(xa, center, cube_shape, padd=-1000))  # cut out  augmented cancer without extra boundry
            return X
        except Exception as e:
            print("Error occuoured ",str(e))
            import traceback
            traceback.print_exc()
            raise TypeError
    
    # Augmentation details
    # def add_random_noise(x, noise_factor=0.1):
    #     noise = np.random.normal(0, noise_factor, x.shape)
    #     return x + noise
    # def random_scale(x, scale_range=(0.8, 1.2)):
    #     scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    #     return scipy.ndimage.zoom(x, zoom=(1, scale_factor, scale_factor), mode='reflect')
    # def random_crop(x, crop_size=(24, 24)):
    #     z, y, x = x.shape
    #     y_start = np.random.randint(0, y - crop_size[0])
    #     x_start = np.random.randint(0, x - crop_size[1])
    #     return x[:, y_start:y_start + crop_size[0], x_start:x_start + crop_size[1]]
    # def adjust_brightness_contrast(x, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
    #     brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    #     contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    #     x = x * brightness
    #     x = (x - np.mean(x)) * contrast + np.mean(x)
    #     return x
    # def elastic_deformation(x, alpha=10, sigma=5):
    #     shape = x.shape
    #     dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #     dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #     dz = np.zeros_like(dx)
    #     indices = np.reshape(np.arange(shape[0]), (-1, 1, 1)) + dz, np.reshape(np.arange(shape[1]), (1, -1, 1)) + dy, np.reshape(np.arange(shape[2]), (1, 1, -1)) + dx
    #     return scipy.ndimage.map_coordinates(x, indices, order=1, mode='reflect')
    # def flip_z(x):
    #     return np.flip(x, 0)
    # def gamma_correction(x, gamma_range=(0.8, 1.2)):
    #     gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    #     return x ** gamma
    
    # def _augmentInstance(self, x0):
    #     # Original augmentations
    #     # xy flip
    #     xf_x = np.flip(x0, 1)
    #     xf_y = np.flip(x0, 2)
    #     xf_xy = np.flip(xf_x, 2)
    #     # xy shift
    #     xs1 = scipy.ndimage.shift(x0, (0, 4, 4), mode='constant')
    #     xs2 = scipy.ndimage.shift(x0, (0, -4, 4), mode='constant')
    #     xs3 = scipy.ndimage.shift(x0, (0, 4, -4), mode='constant')
    #     xs4 = scipy.ndimage.shift(x0, (0, -4, -4), mode='constant')
    #     # small rotations
    #     R = []
    #     for ang in range(6, 360, 6):
    #         R.append(rotate(x0, ang, axes=(1, 2), mode='reflect', reshape=False))
    #     X = [x0, xf_x, xf_y, xf_xy, xs1, xs2, xs3, xs4] + R

    #     # New augmentations
    #     x_noise = self.add_random_noise(x0)
    #     x_scale = self.random_scale(x0)
    #     x_crop = self.random_crop(x0)
    #     x_brightness = self.adjust_brightness_contrast(x0)
    #     x_elastic = self.elastic_deformation(x0)
    #     x_flip_z = self.flip_z(x0)
    #     x_gamma = self.gamma_correction(x0)

    #     # Add new augmentations to the list
    #     X += [x_noise, x_scale, x_crop, x_brightness, x_elastic, x_flip_z, x_gamma]

    #     # Remove instances which are cropped out of bounds of scan
    #     Res = []
    #     for x in X:
    #         if (x.shape[0] != 0) and (x.shape[1] != 0) and (x.shape[2] != 0):
    #             Res.append(x)
    #     return Res
    def _augmentInstance(self, x0):
        # xy flip
        xf_x = np.flip(x0, 1)
        xf_y = np.flip(x0, 2)
        xf_xy = np.flip(xf_x, 2)
        # xy shift
        xs1 = scipy.ndimage.shift(x0, (0, 4, 4), mode='constant')
        xs2 = scipy.ndimage.shift(x0, (0, -4, 4), mode='constant')
        xs3 = scipy.ndimage.shift(x0, (0, 4, -4), mode='constant')
        xs4 = scipy.ndimage.shift(x0, (0, -4, -4), mode='constant')

        # small rotations
        R = []
        for ang in range(6, 360, 6):
            R.append(rotate(x0, ang, axes=(1, 2), mode='reflect', reshape=False))
        X = [x0, xf_x, xf_y, xf_xy, xs1, xs2, xs3, xs4] + R

        # remove instances which are cropped out of bounds of scan
        Res = []
        for x in X:
            if (x.shape[0] != 0) and (x.shape[1] != 0) and (x.shape[2] != 0):
                Res.append(x)
        return Res

    def plot_sample(self,X):
        import matplotlib.pyplot as plt
        r, c = 3, 10
        batch = X[np.random.permutation(len(X))[:30]]
        fig, axs = plt.subplots(r, c, figsize=np.array([30, 10]) * .5)
        fig.suptitle('Random sample of extracted instances: middle slice shown\nIf target samples are incorrect, consider swapping input target x and y coords.')
        cnt = 0
        for i in range(r):
            for j in range(c):
                if cnt < len(batch):
                    axs[i, j].imshow(batch[cnt][16, :, :],cmap='bone')
                    axs[i, j].axis('off')
                cnt += 1
        plt.show()