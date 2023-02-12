import numpy as np
import pydicom
import os
import re
import utils
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# ### Main ###

LABELS = '.../peakLV_labels.csv'
label_file = pd.read_csv(LABELS)
print(label_file.head())
IMG_SIZE = 256

perf_videos = []
for root, dirs, files in os.walk(args["directory"], topdown=True):
    for dir in dirs:
        perf = utils.load_perfusion_data(os.path.join(root,dir))
        perf_videos.append(perf)
