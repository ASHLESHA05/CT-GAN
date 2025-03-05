import os
import sys

os.environ["CLI_ARGS"] = " ".join(arg.lower() for arg in sys.argv[1:]) if len(sys.argv) > 1 else "b"

cli_args = os.environ.get("CLI_ARGS", "").split()
from procedures.attack_pipeline import *
if cli_args[0] == 'b':
    flag ='b'
else:
    flag ='l'
# Init pipeline
remover = scan_manipulator()

if flag=='l':
    print("Lung")
    # Load target scan (provide path to dcm or mhd file)
    remover.load_target_scan(r'D:\CapstoneProject\DataSet\LungsCT-BigData\manifest-1600709154662\LIDC-SMALL\LIDC-IDRI-0002\01-01-2000-NA-NA-98329\3000522.000000-NA-04919')
    # remover.load_target_scan(r'D:\CapstoneProject\DataSet\BraTS2021\TestData\BraTS2021_00000\BraTS2021_00000_t1ce_DICOM')

    # Inject at two locations (this version does not implement auto candidate location selection)
    vox_coord1 = np.array([53,213,400]) #z, y , x (x-y should be flipped if the coordinates were obtained from an image viewer such as RadiAnt)
    vox_coord2 = np.array([33,313,200])
    remover.tamper(vox_coord1, action='remove', isVox=True) #can supply realworld coord too
    remover.tamper(vox_coord2, action='remove', isVox=True)

    # Save scan
    remover.save_tampered_scan('remove_evidence_lung',output_type='dicom') #output can be dicom or numpy
else:
    print("Brain")
    # Load target scan (provide path to dcm or mhd file)
    # remover.load_target_scan(r'D:\CapstoneProject\DataSet\LungsCT-BigData\manifest-1600709154662\LIDC-SMALL\LIDC-IDRI-0002\01-01-2000-NA-NA-98329\3000522.000000-NA-04919')
    remover.load_target_scan(r'D:\CapstoneProject\DataSet\BraTS2021\TestData\BraTS2021_00048\BraTS2021_00048_t1ce_DICOM')

    # Inject at two locations (this version does not implement auto candidate location selection)
    vox_coord1 = np.array([76,157,177]) #z, y , x (x-y should be flipped if the coordinates were obtained from an image viewer such as RadiAnt)
    vox_coord2 = np.array([76,148,195])
    remover.tamper(vox_coord1, action='remove', isVox=True) #can supply realworld coord too
    remover.tamper(vox_coord2, action='remove', isVox=True)

    # Save scan
    remover.save_tampered_scan('remove_evidence_brain',output_type='dicom') #output can be dicom or numpy
