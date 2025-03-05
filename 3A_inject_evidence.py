
import os
import sys
import time
os.environ["CLI_ARGS"] = " ".join(arg.lower() for arg in sys.argv[1:]) if len(sys.argv) > 1 else "b"

time.sleep(0.5)

cli_args = os.environ.get("CLI_ARGS", "").split()
from procedures.attack_pipeline import *
print("Cmdl: ",cli_args)
if cli_args[0] == 'b':
    flag ='b'
else:
    flag ='l'
print('Injecting Evidence...')

# Init pipeline
injector = scan_manipulator()
if flag=='l':
    # Load target scan (provide path to dcm or mhd file)
    PATH = r"D:\CapstoneProject\DataSet\LungsCT-BigData\manifest-1600709154662\LIDC-SMALL\LIDC-IDRI-0002\01-01-2000-NA-NA-98329\3000522.000000-NA-04919"
    # PATH = r"D:\CapstoneProject\DataSet\HealthyBrainTest\healthyTest\0_DICOM"
    injector.load_target_scan(PATH)

    # Inject at two locations (this version does not implement auto candidate location selection)
    vox_coord1 = np.array([53,213,400]) #z, y , x (x-y should be flipped if the coordinates were obtained from an image viewer such as RadiAnt)
    vox_coord2 = np.array([33,313,200])
    injector.tamper(vox_coord1, action='inject', isVox=True) #can supply realworld coord too
    injector.tamper(vox_coord2, action='inject', isVox=True)

    # Save scan
    injector.save_tampered_scan('Lung_inject-out',output_type='dicom') #output can be dicom or numpy
    
    
    
else:
    PATH = r"D:\CapstoneProject\DataSet\HealthyBrainTest\healthyTest\19_DICOM"
    injector.load_target_scan(PATH)

    # Inject at two locations (this version does not implement auto candidate location selection)
    vox_coord1 = np.array([211,109,128]) #z, y , x (x-y should be flipped if the coordinates were obtained from an image viewer such as RadiAnt)
    vox_coord2 = np.array([216,83,96])
    injector.tamper(vox_coord1, action='inject', isVox=True) #can supply realworld coord too
    injector.tamper(vox_coord2, action='inject', isVox=True)

    # Save scan
    injector.save_tampered_scan('Brain_inject-out',output_type='dicom') #output can be dicom or numpy
