import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import glob
import argparse
import utils

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load perfusion images
videos, indices = utils.load_perfusion_data(args["directory"])

# Histogram of pixel intensity
for video in videos:
    for i in range(len(video[0])):
        pix_hist = []
        pix_sum = np.sum(video[0:i-1,:])
        pix_hist.append(pix_sum)
