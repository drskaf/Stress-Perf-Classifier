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
    video = videoraw.pixel_array
    aif_tot = {}
    a_tot = {}
    m_tot = {}
    b_tot = {}
    aif_peak = {}
    a_peak = {}
    m_peak = {}
    b_peak = {}
    
    
    '''The following part will detect peak lv frames from series with or without aif'''
    
    f = len(video[:,]) // 3
    aif_f = len(video[:,]) // 4
    # Generate keys for dictionaries from the indices of the perfusion video frames
    keys = range(len(video[:,]))
    # Loop over the keys and calculate the sum and peak pixel values for each frame
    for k in keys:
        img = utils.centre_crop(video[k])
        sum = np.sum(img)
        max = np.max(img)
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
    # First, identify sequence which performs better with sum pixel rather than peak
    if videoraw.PulseSequenceName == 'B-TFE':
        a_5max = {}
        m_5max = {}
        b_5max = {}
        
        for k, v in a_tot.items():
            if k >= 3 and k< (len(a_tot.items()) - 3):
                value = a_tot[k - 2] + a_tot[k-1] + a_tot[k] + a_tot[k+1] + a_tot[k + 2]   # augment the pixel values
                a_5max[k] = value
            else:
                a_5max[k] = v
        a_max_value = list(a_5max.values())
        a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
        a_k = a_max_key.pop()

        for k, v in m_tot.items():
            if k >= 3 and k < (len(m_tot.items()) - 3):
                value = m_tot[k - 2] + m_tot[k - 1] + m_tot[k] + m_tot[k + 1] + m_tot[k - 2]
                m_5max[k] = value
            else:
                m_5max[k] = v
        m_max_value = list(m_5max.values())
        m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
        m_k = m_max_key.pop()

        for k, v in b_tot.items():
            if k >= 3 and k < (len(b_tot.items()) - 3):
                value = b_tot[k - 2] + b_tot[k - 1] + b_tot[k] + b_tot[k + 1] + b_tot[k + 2]
                b_5max[k] = value
            else:
                b_5max[k] = v
        b_max_value = list(b_5max.values())
        b_max_key = [key for key, value in b_5max.items() if value == np.max(b_5max_value)]
        b_k = b_max_key.pop()


