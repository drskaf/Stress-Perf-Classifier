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
# Define series with AIF frames
for videoraw, i in zip(videos, indices):
    test = {}
    video = videoraw.pixel_array
    keys = range(len(video[:, ]))
    for k in keys:
        # Calculate image sharpness
        img = utils.centre_crop(video[k])
        gy, gx = np.gradient(img)
        gnorm = np.sqrt(gx**2, gy**2)
        sharp = np.average(gnorm)
        test[k] = sharp
    plt.bar(test.keys(), test.values(), color='purple')
    plt.show()
    aif_f = len(video[:,]) // 4
    aif_sharp = np.sum(list(test.values())[:aif_f])
    nonaif_sharp = np.sum(list(test.values())[aif_f+1:2*aif_f])
    if aif_sharp < nonaif_sharp // 2:
        # Initiate dictionaries for 4 groups of AIF, apical, mid and basal LV level
        # both for pixel sum values and pixel peak values for each frame
        aif_tot = {}
        a_tot = {}
        m_tot = {}
        b_tot = {}
        aif_peak = {}
        a_peak = {}
        m_peak = {}
        b_peak = {}
        '''Work on series with AIF frames'''
        f = len(video[:, ]) // 4
        # Loop over the keys and calculate the sum and peak pixel values for each frame
        for k in keys:
            img = utils.centre_crop(video[k])
            sum = np.sum(img)
            max = np.max(img)
            # Collect frames from the first group of slices
            if k <= f:
                aif_tot[k] = sum
                aif_peak[k] = max
            # Collect frames from the second group of slices
            elif k > f and k <= 2 * f:
                a_tot[k] = sum
                a_peak[k] = max
            # Collect frames from the third group of slices
            elif k > 2 * f and k <= 3 * f:
                m_tot[k] = sum
                m_peak[k] = max
            # Collect frames from the fourth group of slices
            else:
                b_tot[k] = sum
                b_peak[k] = max
        # Initiate dictionaries
        aif_5max = {}
        a_5max = {}
        m_5max = {}
        b_5max = {}
        # Augment pixel sum and trip off the first and last few frames
        for k, v in aif_tot.items():
            if k >= 5 and k < (f - 3):
                value = aif_tot[k - 2] + aif_tot[k - 1] + aif_tot[k] + aif_tot[k + 1] + aif_tot[k + 2]
                aif_5max[k] = value
            else:
                aif_5max[k] = 0
        aif_max_value = list(aif_5max.values())
        aif_max_key = [key for key, value in aif_5max.items() if value == np.max(aif_max_value)]
        aif_l = aif_max_key.pop()
        for img in video[aif_l:aif_l + 5]:
            plt.imshow(img, cmap='gray')
            plt.show()

        for k, v in a_tot.items():
            if k >= 5 + f and k < (2 * f - 3):
                value = a_tot[k - 2] + a_tot[k - 1] + a_tot[k] + a_tot[k + 1] + a_tot[k + 2]
                a_5max[k] = value
            else:
                a_5max[k] = 0
        a_max_value = list(a_5max.values())
        a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
        a_l = a_max_key.pop()
        for img in video[a_l:a_l + 5]:
            plt.imshow(img, cmap='gray')
            plt.show()

        for k, v in m_tot.items():
            if k >= 5 + 2 * f and k < (3 * f - 3):
                value = m_tot[k - 2] + m_tot[k - 1] + m_tot[k] + m_tot[k + 1] + m_tot[k - 2]
                m_5max[k] = value
            else:
                m_5max[k] = 0
        m_max_value = list(m_5max.values())
        m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
        m_l = m_max_key.pop()
        for img in video[m_l:m_l + 5]:
            plt.imshow(img, cmap='gray')
            plt.show()

        for k, v in b_tot.items():
            if k >= 5 + 3 * f and k < (4 * f - 3):
                value = b_tot[k - 2] + b_tot[k - 1] + b_tot[k] + b_tot[k + 1] + b_tot[k + 2]
                b_5max[k] = value
            else:
                b_5max[k] = 0
        b_max_value = list(b_5max.values())
        b_max_key = [key for key, value in b_5max.items() if value == np.max(b_max_value)]
        b_l = b_max_key.pop()
        for img in video[b_l:b_l + 5]:
            plt.imshow(img, cmap='gray')
            plt.show()
     
    else:
        # Initiate dictionaries
        a_tot = {}
        m_tot = {}
        b_tot = {}
        a_peak = {}
        m_peak = {}
        b_peak = {}
        '''Work on series without AIF frames'''
        f = len(video[:, ]) // 3
        for k in keys:
            img = utils.centre_crop(video[k])
            sum = np.sum(img)
            max = np.max(img)
            # Collect frames from the first group of slices
            if k <= f:
                a_tot[k] = sum
                a_peak[k] = max
            # Collect frames from the second group of slices
            elif k > f and k <= 2 * f:
                m_tot[k] = sum
                m_peak[k] = max
            # Collect frames from the third group of slices
            else:
                b_tot[k] = sum
                b_peak[k] = max

        # Generate a list of peak pixels values then find the key of the frame with the max value,
        # This will be done on all 3 groups of slices
        # First, identify sequence which performs better with sum pixel rather than peak
        if videoraw.PulseSequenceName == 'B-TFE':
            a_5max = {}
            m_5max = {}
            b_5max = {}
            # Working on 1st group
            for k, v in a_tot.items():
                if k >= 5 and k < (f - 3):
                    value = a_tot[k - 2] + a_tot[k - 1] + a_tot[k] + a_tot[k + 1] + a_tot[k + 2]
                    a_5max[k] = value
                else:
                    a_5max[k] = 0
            a_max_value = list(a_5max.values())
            a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
            a_l = a_max_key.pop()
            for img in video[a_l:a_l + 5]:
                plt.imshow(img, cmap='gray')
                plt.show()
            # Working on 2nd group
            for k, v in m_tot.items():
                if k >= 3 + f and k < (2 * f - 3):
                    value = m_tot[k - 2] + m_tot[k - 1] + m_tot[k] + m_tot[k + 1] + m_tot[k + 2]
                    m_5max[k] = value
                else:
                    m_5max[k] = 0
            m_max_value = list(m_5max.values())
            m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
            m_l = m_max_key.pop()
            for img in video[m_l:m_l + 5]:
                plt.imshow(img, cmap='gray')
                plt.show()
            # Working on 3rd group
            for k, v in b_tot.items():
                if k >= 5 + 2 * f and k < (3 * f - 3):
                    value = b_tot[k - 2] + b_tot[k - 1] + b_tot[k] + b_tot[k + 1] + b_tot[k + 2]
                    b_5max[k] = value
                else:
                    b_5max[k] = 0
            b_max_value = list(b_5max.values())
            b_max_key = [key for key, value in b_5max.items() if value == np.max(b_max_value)]
            b_l = b_max_key.pop()
            for img in video[b_l:b_l + 5]:
                plt.imshow(img, cmap='gray')
                plt.show()

        # Working on the rest of sequences
        else:
            a_5max = {}
            m_5max = {}
            b_5max = {}
            # Working on 1st group
            for k, v in a_peak.items():
                if k >= 5 and k < (f - 3):
                    value = a_peak[k - 2] + a_peak[k - 1] + a_peak[k] + a_peak[k + 1] + a_peak[k + 2]
                    a_5max[k] = value
                else:
                    a_5max[k] = 0
            a_max_value = list(a_5max.values())
            a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
            a_l = a_max_key.pop()
            for img in video[a_l:a_l + 5]:
                plt.imshow(img, cmap='gray')
                plt.show()
            # Working on 2nd group
            for k, v in m_peak.items():
                if k >= 5 + f and k < (2 * f - 3):
                    value = m_peak[k - 2] + m_peak[k - 1] + m_peak[k] + m_peak[k + 1] + m_peak[k + 2]
                    m_5max[k] = value
                else:
                    m_5max[k] = 0
            m_max_value = list(m_5max.values())
            m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
            m_l = m_max_key.pop()
            for img in video[m_l:m_l + 5]:
                plt.imshow(img, cmap='gray')
                plt.show()
            # Working on 3rd group
            for k, v in b_peak.items():
                if k >= 5 + 2 * f and k < (3 * f - 3):
                    value = b_peak[k - 2] + b_peak[k - 1] + b_peak[k] + b_peak[k + 1] + b_peak[k + 2]
                    b_5max[k] = value
                else:
                    b_5max[k] = 0
            b_max_value = list(b_5max.values())
            b_max_key = [key for key, value in b_5max.items() if value == np.max(b_max_value)]
            b_l = b_max_key.pop()
            for img in video[b_l:b_l + 5]:
                plt.imshow(img, cmap='gray')
                plt.show()
