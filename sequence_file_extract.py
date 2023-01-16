import numpy as np
import os
import pydicom
import argparse
import glob
import re
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-s", "--sequence", required=False, help="name of required sequence")
args = vars(ap.parse_args())

dir_paths = sorted(glob.glob(os.path.join(args["directory"], "*")))

for dir_path in dir_paths:
    id = os.path.basename(dir_path)
    dir = os.mkdir(id)
    dcm_paths = sorted(glob.glob(os.path.join(dir_path, "*", "DICOM", "*.dcm")))

    for dcm_path in dcm_paths:
        file = pydicom.dcmread(dcm_path)
        if re.match(args["sequence"], str(file.SeriesDescription), flags=re.I):
            shutil.move(dcm_path, f"{id}")
