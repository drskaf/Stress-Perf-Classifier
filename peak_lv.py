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
    pix_bins = []
    indices = []
    for i in range(len(video[:,])):
        pix_sum = np.sum(video[i])
        pix_mean = np.mean(video[i])
        pix_std = np.std(video[i])
        pix_norm = pix_sum - pix_mean // pix_std
        pix_bins.append(pix_norm)
        indices.append(i)
    plt.bar(indices, pix_bins, color='blue')
    plt.show()
