import numpy as np
import pandas as pd
import os
import pydicom
import argparse
from os import walk
import glob
import re
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-s", "--sequence", required=False, help="name of required sequence")
ap.add_argument("-o", "--output", required=False, help="output directory")
args = vars(ap.parse_args())

paths = sorted(glob.glob(os.path.join(args["directory"], "*","*","*", "*.dcm")))

main_dir = []

for path in paths:
    file = pydicom.dcmread(path)
    try:
        file.SeriesDescription
    except NameError:
        file.SeriesDescription = None
    if re.match(args["sequence"], str(file.SeriesDescription), flags=re.I):
        main_dir.append(file)
        output_dir = args["output"]
        fname = os.path.basename(path)
        file_name = os.path.join(output_dir, fname)
        file1 = open(fname, "w")
        file1.write('file')
        file1.close()
