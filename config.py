import os
import numpy as np
import torch


def generate_random_mask(cube_shape, mask_size):
    """
    Generate a random mask within the cube.
    Args:
        cube_shape: Shape of the input cube (z, y, x).
        mask_size: Size of the mask (z, y, x).
    Returns:
        mask_lims: Limits of the mask in the format [zlims, ylims, xlims].
    """
    z, y, x = cube_shape
    mask_z, mask_y, mask_x = mask_size
    
    # Randomly choose the starting point for the mask
    z_start = np.random.randint(0, z - mask_z)
    y_start = np.random.randint(0, y - mask_y)
    x_start = np.random.randint(0, x - mask_x)
    
    # Define the mask limits
    mask_lims = np.array([
        [z_start, z_start + mask_z],
        [y_start, y_start + mask_y],
        [x_start, x_start + mask_x]
    ])
    
    return mask_lims


flag =None
cli_args = os.environ.get("CLI_ARGS", "").split()
if cli_args[0] == 'b':
    flag ='b'
else:
    flag ='l'

print("Values in cli_Arg",cli_args)
if flag=='l':
    config = {}

    # Data Location
    config['healthy_scans_raw'] = "D:\\CapstoneProject\\DataSet\\Normal Cases"
    config['healthy_coords'] = "D:\\CapstoneProject\\DataSet\\healthy_lung.csv"
    config['healthy_samples'] = "output/healthy_samples.npy"

    config['unhealthy_scans_raw'] = r"D:\CAPSTONE\BraTS2021\BraTS2021_Training_Data"
    config['unhealthy_coords'] = r"D:\CAPSTONE\BraTS2021\BraTS2021_tumor_centroids.csv"
    # config['unhealthy_coords'] = r"D:\CapstoneProject\Attack_prep\updated_roi_centroids.csv"
    config['unhealthy_samples'] = "output/unhealthy_samples.npy"

    config['traindata_coordSystem'] = "vox"  # the coord system used to note the locations of the evidence ('world' or 'vox')
    BASE_PATH = r'D:\CAPSTONE\CT-GAN'
    # Model & Progress Location
    config['modelpath_inject'] = os.path.join(BASE_PATH,"data", "models", "INJ")
    config['modelpath_remove'] = os.path.join(BASE_PATH,"data", "models", "REM")
    config['progress'] = "images"

    # Device Configuration
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        config['device'] = 'cpu'
        print("No GPU available, using CPU")
    config['gpus'] = "0" if torch.cuda.is_available() else ""

    # CT-GAN Configuration
    config['cube_shape'] = np.array([32, 32, 32])  # z,y,x
    config['mask_xlims'] = np.array([6, 26])
    config['mask_ylims'] = np.array([6, 26])
    config['mask_zlims'] = np.array([6, 26])
    config['copynoise'] = True

    # Validation checks
    if config['mask_zlims'][1] > config['cube_shape'][0]:
        raise Exception('Out of bounds: cube mask is larger then cube on dimension z.')
    if config['mask_ylims'][1] > config['cube_shape'][1]:
        raise Exception('Out of bounds: cube mask is larger then cube on dimension y.')
    if config['mask_xlims'][1] > config['cube_shape'][2]:
        raise Exception('Out of bounds: cube mask is larger then cube on dimension x.')

    # Make save directories
    os.makedirs(config['modelpath_inject'], exist_ok=True)
    os.makedirs(config['modelpath_remove'], exist_ok=True)
    os.makedirs(config['progress'], exist_ok=True)
else:
    '''
    BELOW ONE IS FOR BRAIN TUMOR

    '''
    import os
    import numpy as np
    import torch

    config = {}

    # # Data Location
    # config['healthy_scans_raw'] = r"D:\CapstoneProject\DataSet\HealthyBrain"
    # config['healthy_coords'] = r"d:\CapstoneProject\Attack_prep\Healthy_brain_fake_tumor_loc.csv"
    # config['healthy_samples'] = "output2/healthy_samples.npy"

    # config['unhealthy_scans_raw'] = r"D:\CapstoneProject\DataSet\BraTS2021\BraTS2021_Training_Data"
    # config['unhealthy_coords'] = r"D:\CapstoneProject\Attack_prep\BraTS2021_tumor_centroids.csv"
    # # config['unhealthy_coords'] = r"D:\CapstoneProject\Attack_prep\updated_roi_centroids.csv"
    # config['unhealthy_samples'] = "output2/unhealthy_samples.npy"
    
    config['healthy_scans_raw'] = "/kaggle/input/healthybrain-at"
    config['healthy_coords'] = '/kaggle/input/healty-braincsv-at/Healthy_brain_fake_tumor_loc.csv'
    config['healthy_samples'] = "/kaggle/working/output2/healthy_samples.npy"

    config['unhealthy_scans_raw'] = "/kaggle/input/brats21"
    config['unhealthy_coords'] = "/kaggle/input/brain-tumor-csv/BraTS2021_tumor_centroids.csv"
    # config['unhealthy_coords'] = r"D:\CapstoneProject\Attack_prep\updated_roi_centroids.csv"
    config['unhealthy_samples'] = "/kaggle/working/output2/unhealthy_samples.npy"

    config['traindata_coordSystem'] = "vox"  # the coord system used to note the locations of the evidence ('world' or 'vox')
    # BASE_PATH = r'D:\CAPSTONE\CT-GAN'
    BASE_PATH = '/kaggle/working'
    # Model & Progress Location
    config['modelpath_inject'] = os.path.join(BASE_PATH,"data2", "models", "INJ")
    config['modelpath_remove'] = os.path.join(BASE_PATH,"data2", "models", "REM")
    config['progress'] = "images"

    # Device Configuration
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        config['device'] = 'cpu'
        print("No GPU available, using CPU")
    config['gpus'] = "0" if torch.cuda.is_available() else ""

    # CT-GAN Configuration
    config['cube_shape'] = np.array([32, 32, 32])  # z,y,x
    config['mask_xlims'] = np.array([6, 26])
    config['mask_ylims'] = np.array([6, 26])
    config['mask_zlims'] = np.array([6, 26])
    config['copynoise'] = True

    # Validation checks
    if config['mask_zlims'][1] > config['cube_shape'][0]:
        raise Exception('Out of bounds: cube mask is larger then cube on dimension z.')
    if config['mask_ylims'][1] > config['cube_shape'][1]:
        raise Exception('Out of bounds: cube mask is larger then cube on dimension y.')
    if config['mask_xlims'][1] > config['cube_shape'][2]:
        raise Exception('Out of bounds: cube mask is larger then cube on dimension x.')

    # Make save directories
    os.makedirs(config['modelpath_inject'], exist_ok=True)
    os.makedirs(config['modelpath_remove'], exist_ok=True)
    os.makedirs(config['progress'], exist_ok=True)
    