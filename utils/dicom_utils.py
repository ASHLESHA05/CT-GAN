import pydicom #for loading dicom
import SimpleITK as sitk #for loading mhd/raw
import nibabel as nib #for loading nifti
import os
import numpy as np
import scipy.ndimage
from pydicom.encaps import encapsulate
from pydicom.uid import UID

#DICOM: send path to any *.dcm file where containing dir has the other slices (dcm files), or path to the dir itself
#MHD/RAW: send path to the *.mhd file where containing die has the coorisponding *.raw file
def load_scan(path2scan):
    # ext = path2scan.split('.')[-1].lower()
    # if ext in ['mhd', 'raw']:
    #     return load_mhd(path2scan)
    # elif ext in ['nii', 'gz']:
    #     return load_nifti(path2scan)
    # elif ext == 'dcm':
    #     return load_dicom(os.path.split(path2scan)[0])
    # elif any(file.endswith('.dcm') for file in os.listdir(path2scan)):
    #     return load_dicom(path2scan)
    # else:
    #     raise Exception('No valid scan [series] found in given file/directory')
    print(path2scan)
    if (path2scan.split('.')[-1] == 'mhd') or (path2scan.split('.')[-1] == 'raw'):
        return load_mhd(path2scan)
    elif path2scan.split('.')[-1] == 'dcm':
        # print("Here:",path2scan)
        return load_dicom(os.path.split(path2scan)[0]) #pass containing directory
    elif any(file.endswith('.dcm') for file in os.listdir(path2scan)):
        # print("Here:",path2scan)
        return load_dicom(path2scan)
    elif any(file.endswith('t1ce.nii.gz') for file in os.listdir(path2scan)):
        return load_nifti(path2scan)
    elif any(path2scan.endswith('t1ce.nii')):
        return load_nifti(path2scan)
    else:
        raise Exception('No valid scan [series] found in given file/directory')


def load_mhd(path2scan):
    itkimage = sitk.ReadImage(path2scan)
    scan = sitk.GetArrayFromImage(itkimage)
    spacing = np.flip(np.array(itkimage.GetSpacing()),axis=0)
    orientation = np.transpose(np.array(itkimage.GetDirection()).reshape((3, 3)))
    origin = np.flip(np.array(itkimage.GetOrigin()),axis=0) #origionally in yxz format (read xy in viewers but sanved as yx)
    return scan, spacing, orientation, origin, None #output in zyx format

def load_nifti(path2scan):
    try:
        for file in os.listdir(path2scan):
            if file.endswith('.nii'):
                full_path = os.path.join(path2scan, file)

                if os.path.isfile(full_path):
                    nifti_file = full_path
                elif file.endswith('t1ce.nii.gz') or file.endswith('t1ce.nii'):
                    # If it's a directory, get the first file inside it
                    first_file = os.listdir(full_path)[0]
                    nifti_file = os.path.join(full_path, first_file)
            
        # print(path2scan,"--|--",nifti_file)
        nifti_img = nib.load(nifti_file)
        scan = np.array(nifti_img.get_fdata(), dtype=np.float32)
        spacing = np.array(nifti_img.header.get_zooms())
        affine = nifti_img.affine
        orientation = affine[:3, :3]
        origin = affine[:3, 3]
        return scan, spacing, orientation, origin, None
    except Exception as e:
        print("Error in loading nifti: ",str(e))

def load_dicom(path2scan_dir):
    dicom_folder = path2scan_dir
# Assuming dicom_folder is the path to your folder
    dcms = [file for file in os.listdir(dicom_folder) if file.endswith('.dcm') and os.path.isfile(os.path.join(dicom_folder, file))]
    # print(dcms)
    # print(path2scan_dir)
    first_slice_data = pydicom.dcmread(path2scan_dir + '\\' + dcms[0])
    # print(first_slice_data)
    first_slice = first_slice_data.pixel_array
    orientation = np.transpose(first_slice_data.ImageOrientationPatient) #zyx format
    # print("orientation shape in load dicom ,",orientation.shape)
    spacing_xy = np.array(first_slice_data.PixelSpacing, dtype=float)
    spacing_z = np.float64(first_slice_data.SliceThickness)
    spacing = np.array([spacing_z, spacing_xy[1], spacing_xy[0]]) #zyx format

    scan = np.zeros((len(dcms),first_slice.shape[0],first_slice.shape[1]))
    raw_slices=[]
    indexes = []
    for dcm in dcms:
        slice_data = pydicom.dcmread(dicom_folder + '\\' + dcm)
        slice_data.filename = dcm
        raw_slices.append(slice_data)
        indexes.append(float(slice_data.ImagePositionPatient[2]))
    indexes = np.array(indexes,dtype=float)

    raw_slices = [x for _, x in sorted(zip(indexes, raw_slices), key=lambda x: x[0])]
    origin = np.array(raw_slices[0][0x00200032].value) #origin is assumed to be the image location of the first slice
    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.array([origin[2],origin[1],origin[0]]) #change from x,y,z to z,y,x

    for i, slice in enumerate(raw_slices):
        scan[i, :, :] = slice.pixel_array
    # for i, index in enumerate(indexes):
    #     for slice in raw_slices:
    #         if int(slice.InstanceNumber) == index:
    #             scan[i,:,:] = slice._pixel_data_numpy()
    return scan, spacing, orientation, origin, raw_slices


#point to directory of folders conting dicom scans only (subdirs only), runs aon all folders..
# ref directory is used to copy the m


def save_dicom(modified_scan, original_raw_slices, dst_directory):
    os.makedirs(dst_directory, exist_ok=True)
    
    for i, slice_dcm in enumerate(original_raw_slices):
        # Get the modified pixel data for this slice
        modified_slice = np.nan_to_num(modified_scan[i], nan=0)

        # Clip values to the valid range of int16
        modified_slice = np.clip(modified_slice, -32768, 32767)

        # Convert to int16
        modified_slice = modified_slice.astype(np.int16)
        
        # Check if the original uses a compressed transfer syntax
        is_compressed = slice_dcm.file_meta.TransferSyntaxUID.is_compressed
        
        # Update pixel data
        if is_compressed:
            # Compressed syntax requires encapsulation
            slice_dcm.PixelData = encapsulate([modified_slice.tobytes()])
        else:
            # Uncompressed syntax
            slice_dcm.PixelData = modified_slice.tobytes()
        
        # Update metadata for consistency
        slice_dcm.Rows, slice_dcm.Columns = modified_slice.shape
        slice_dcm.PlanarConfiguration = 0  # Required for RGB
        
        # Save the modified slice
        slice_dcm.save_as(os.path.join(dst_directory, f"slice_{i}.dcm"))
        
#img_array: 3d numpy matrix, z,y,x
def toDicom(save_dir, img_array,  pixel_spacing, orientation):
    ref_scan = pydicom.dcmread('utils/ref_scan.dcm') #slice from soem real scan so we can copy the meta data

    #write dcm file for each slice in scan
    for i, slice in enumerate(img_array):
        ref_scan.pixel_array.flat = img_array[i,:,:].flat
        ref_scan.PixelData = ref_scan.pixel_array.tobytes()
        ref_scan.RefdSOPInstanceUID = str(i)
        ref_scan.SOPInstanceUID = str(i)
        ref_scan.InstanceNumber = str(i)
        ref_scan.SliceLocation = str(i)
        ref_scan.ImagePositionPatient[2] = str(i*pixel_spacing[0])
        ref_scan.RescaleIntercept = 0
        ref_scan.Rows = img_array.shape[1]
        ref_scan.Columns = img_array.shape[2]
        ref_scan.PixelSpacing = [str(pixel_spacing[2]),str(pixel_spacing[1])]
        ref_scan.SliceThickness = pixel_spacing[0]
        #Pixel Spacing                       DS: ['0.681641', '0.681641']
        #Image Position (Patient)            DS: ['-175.500000', '-174.500000', '49']
        #Image Orientation (Patient)         DS: ['1.000000', '0.000000', '0.000000', '0.000000', '1.000000', '0.000000']
        #Rows                                US: 512
        #Columns                             US: 512
        os.makedirs(save_dir,exist_ok=True)
        ref_scan.save_as(os.path.join(save_dir,str(i)+'.dcm'))

def scale_scan(scan,spacing,factor=1):
    try:
        resize_factor = factor * spacing
        new_real_shape = scan.shape * resize_factor
        new_shape = np.round(new_real_shape)
        if 0 in scan.shape or 0 in new_shape:
            print("Warning: Zero dimension encountered in scan.shape or new_shape, skipping resize.")
            return scan, 0  # Skip resizing, return the original scan
        real_resize_factor = new_shape / scan.shape
        new_spacing = spacing / real_resize_factor
        scan_resized = scipy.ndimage.interpolation.zoom(scan, real_resize_factor, mode='nearest')
        return scan_resized, resize_factor
    except Exception as e:
        print("Error in something ",str(e))
        import traceback
        traceback.print_exc()
        exit(0)

# get the shape of an orthotope in 1:1:1 ratio AFTER scaling to the given spacing
def get_scaled_shape(shape, spacing):
    new_real_shape = shape * spacing
    return np.round(new_real_shape).astype(int)

def scale_vox_coord(coord, spacing, factor=1):
    resize_factor = factor * spacing
    return (coord*resize_factor).astype(int)

def world2vox(world_coord, spacing, orientation, origin):
    try:

        if len(orientation) == 6:  # If orientation is given as 6 values, reshape to (3, 3)
            orientation = np.array(orientation).reshape(3, 2)
        orientation = np.pad(orientation, ((0, 0), (0, 1)), mode='constant')

        world_coord = np.dot(np.linalg.inv(np.dot(orientation, np.diag(spacing))), world_coord - origin)
        if orientation[0, 0] < 0:
            vox_coord = (np.array([world_coord[0], world_coord[2], world_coord[1]])).astype(int)
        else:
            vox_coord = (np.array([world_coord[0], world_coord[1], world_coord[2]])).astype(int)
        return vox_coord
    except Exception as e:
        print("Error in world2vox" ,str(e))
        import traceback
        traceback.print_exc()