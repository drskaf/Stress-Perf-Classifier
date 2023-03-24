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

""" EXTRACT PEAK LV""" 
# Loop over cases and their study tags
for video, i in zip(videos, indices):
    # Initiate dictionaries for 3 groups of apical, mid and basal LV level
    # both for pixel sum values and pixel peak values for each frame
    a_tot = {}
    m_tot = {}
    b_tot = {}
    a_peak = {}
    m_peak = {}
    b_peak = {}
    f = len(video[:,]) // 3
    # Generate keys for dictionaries from the indices of the perfusion video frames
    keys = range(len(video[:,]))
    # Loop over the keys and calculate the sum and peak pixel values for each frame
    for k in keys:
        sum = np.sum(video[k])
        max = np.max(video[k])
        # Collect frames from the first group of slices
        if k <= f:
            a_tot[int(k)] = sum
            a_peak[int(k)] = max
        # Collect frames from the second group of slices
        elif k > f and k <= 2*f:
            m_tot[k] = sum
            m_peak[k] = max
        # Collect frames from the third group of slices
        else:
            b_tot[k] = sum
            b_peak[k] = max
    
    # Generate a list of peak pixels values then find the key of the frame with the max value,
    # this will be followed by 4-5 frames to get the myocardial contrast frame
    # This will be done on all 3 groups of slices
    a_max_value = list(a_peak.values())
    a_max_key = [key for key, value in a_peak.items() if value == np.max(a_max_value)]
    l = a_max_key.pop()
    plt.imshow(video[l + 4], cmap='gray')
    plt.show()

    m_max_value = list(m_peak.values())
    m_max_key = [key for key, value in m_peak.items() if value == np.max(m_max_value)]
    l = m_max_key.pop()
    plt.imshow(video[l + 4], cmap='gray')
    plt.show()

    b_max_value = list(b_peak.values())
    b_max_key = [key for key, value in b_peak.items() if value == np.max(b_max_value)]
    l = b_max_key.pop()
    plt.imshow(video[l + 4], cmap='gray')
    plt.show()

