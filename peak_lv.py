import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import glob
import argparse
import utils
from skimage import color
import pydicom
from sortedcontainers import SortedDict

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load perfusion images and their indices
videoSingles, indicesSingles, videoStacks, indicesStacks = utils.load_perfusion_data(args["directory"])

""" EXTRACT PEAK LV frames"""
for videoraw, video, i in zip(videoraws, videos, indices):
    test = {}
    keys = range(len(video[:, ]))
    # Define series with AIF frames
    for k in keys:
        # Calculate image sharpness
        img = utils.centre_crop(video[k])
        gy, gx = np.gradient(img)
        gnorm = np.sqrt(gx ** 2, gy ** 2)
        sharp = np.average(gnorm)
        test[k] = sharp
    aif_f = len(video[:, ]) // 4
    aif_sharp = np.sum(list(test.values())[:aif_f])
    nonaif_sharp = np.sum(list(test.values())[aif_f + 1:2 * aif_f])
    if aif_sharp < nonaif_sharp // 2:
        '''Work on series with AIF frames'''
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
            elif k > 2 * f and k <= 3 * f:
                m_tot[k] = sum
                m_peak[k] = max
            # Collect frames from the third group of slices
            else:
                b_tot[k] = sum
                b_peak[k] = max

        aif_5max = {}
        a_5max = {}
        m_5max = {}
        b_5max = {}
        for k, v in aif_tot.items():
            if k >= 5 and k < (f - 3):
                value = aif_tot[k - 2] + aif_tot[k - 1] + aif_tot[k] + aif_tot[k + 1] + aif_tot[k + 2]
                aif_5max[k] = value
            else:
                aif_5max[k] = 0
        aif_max_value = list(aif_5max.values())
        aif_max_key = [key for key, value in aif_5max.items() if value == np.max(aif_max_value)]
        aif_l = aif_max_key.pop()
        for k, v in a_tot.items():
            if k >= 5 + f and k < (2 * f - 3):
                value = a_tot[k - 2] + a_tot[k - 1] + a_tot[k] + a_tot[k + 1] + a_tot[k + 2]
                a_5max[k] = value
            else:
                a_5max[k] = 0
        a_max_value = list(a_5max.values())
        a_max_key = [key for key, value in a_5max.items() if value == np.max(a_max_value)]
        a_l = a_max_key.pop()
        for k, v in m_tot.items():
            if k >= 5 + 2 * f and k < (3 * f - 3):
                value = m_tot[k - 2] + m_tot[k - 1] + m_tot[k] + m_tot[k + 1] + m_tot[k - 2]
                m_5max[k] = value
            else:
                m_5max[k] = 0
        m_max_value = list(m_5max.values())
        m_max_key = [key for key, value in m_5max.items() if value == np.max(m_max_value)]
        m_l = m_max_key.pop()
        for k, v in b_tot.items():
            if k >= 5 + 3 * f and k < (4 * f - 3):
                value = b_tot[k - 2] + b_tot[k - 1] + b_tot[k] + b_tot[k + 1] + b_tot[k + 2]
                b_5max[k] = value
            else:
                b_5max[k] = 0
        b_max_value = list(b_5max.values())
        b_max_key = [key for key, value in b_5max.items() if value == np.max(b_max_value)]
        b_l = b_max_key.pop()

        # Saving images
        if 'SliceLocation' in videoraw:
            slice_dic = {videoraw[a_l].SliceLocation: video[a_l + 3],
                         videoraw[m_l].SliceLocation: video[m_l + 3],
                         videoraw[b_l].SliceLocation: video[b_l + 3]}
            sd = SortedDict(slice_dic)
            img1 = utils.centre_crop(slice_dic[sd.iloc[0]])
            img2 = utils.centre_crop(slice_dic[sd.iloc[1]])
            img3 = utils.centre_crop(slice_dic[sd.iloc[2]])
            dir = f"{i}"
            path = os.path.join('peak_LV_images', dir)
            os.makedirs(path, exist_ok=True)
            plt.imshow(img1, cmap='gray')
            plt.savefig(f"peak_LV_images/{dir}/a.png")
            plt.imshow(img2, cmap='gray')
            plt.savefig(f"peak_LV_images/{dir}/m.png")
            plt.imshow(img3, cmap='gray')
            plt.savefig(f"peak_LV_images/{dir}/b.png")

        else:
            img1 = utils.centre_crop(video[a_l + 3])
            img2 = utils.centre_crop(video[m_l + 3])
            img3 = utils.centre_crop(video[b_l + 3])
            dir = f"{i}"
            path = os.path.join('peak_LV_images', dir)
            os.makedirs(path, exist_ok=True)
            plt.imshow(img1, cmap='gray')
            plt.savefig(f"peak_LV_images/{dir}/a.png")
            plt.imshow(img2, cmap='gray')
            plt.savefig(f"peak_LV_images/{dir}/m.png")
            plt.imshow(img3, cmap='gray')
            plt.savefig(f"peak_LV_images/{dir}/b.png")

    else:
        '''Work on series without AIF frames'''
        # Initiate dictionaries
        a_tot = {}
        m_tot = {}
        b_tot = {}
        a_peak = {}
        m_peak = {}
        b_peak = {}
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
        # this will be followed by 4-5 frames to get the myocardial contrast frame
        # This will be done on all 3 groups of slices
        # First, identify sequence which performs better with sum pixel rather than peak
        if 'PulseSequenceName' in videoraw and videoraw.PulseSequenceName == 'B-TFE':
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

            # Saving images
            if 'SliceLocation' in videoraw:
                slice_dic = {videoraw[a_l].SliceLocation: video[a_l + 3],
                             videoraw[m_l].SliceLocation: video[m_l + 3],
                             videoraw[b_l].SliceLocation: video[b_l + 3]}
                sd = SortedDict(slice_dic)
                img1 = utils.centre_crop(slice_dic[sd.iloc[0]])
                img2 = utils.centre_crop(slice_dic[sd.iloc[1]])
                img3 = utils.centre_crop(slice_dic[sd.iloc[2]])
                dir = f"{i}"
                path = os.path.join('peak_LV_images', dir)
                os.makedirs(path, exist_ok=True)
                plt.imshow(img1, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/a.png")
                plt.imshow(img2, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/m.png")
                plt.imshow(img3, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/b.png")

            else:
                img1 = utils.centre_crop(video[a_l + 3])
                img2 = utils.centre_crop(video[m_l + 3])
                img3 = utils.centre_crop(video[b_l + 3])
                dir = f"{i}"
                path = os.path.join('peak_LV_images', dir)
                os.makedirs(path, exist_ok=True)
                plt.imshow(img1, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/a.png")
                plt.imshow(img2, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/m.png")
                plt.imshow(img3, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/b.png")

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

            # Saving images
            if 'SliceLocation' in videoraw:
                slice_dic = {videoraw[a_l].SliceLocation: video[a_l + 3],
                             videoraw[m_l].SliceLocation: video[m_l + 3],
                             videoraw[b_l].SliceLocation: video[b_l + 3]}
                sd = SortedDict(slice_dic)
                img1 = utils.centre_crop(slice_dic[sd.iloc[0]])
                img2 = utils.centre_crop(slice_dic[sd.iloc[1]])
                img3 = utils.centre_crop(slice_dic[sd.iloc[2]])
                dir = f"{i}"
                path = os.path.join('peak_LV_images', dir)
                os.makedirs(path, exist_ok=True)
                plt.imshow(img1, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/a.png")
                plt.imshow(img2, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/m.png")
                plt.imshow(img3, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/b.png")

            else:
                img1 = utils.centre_crop(video[a_l + 3])
                img2 = utils.centre_crop(video[m_l + 3])
                img3 = utils.centre_crop(video[b_l + 3])
                dir = f"{i}"
                path = os.path.join('peak_LV_images', dir)
                os.makedirs(path, exist_ok=True)
                plt.imshow(img1, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/a.png")
                plt.imshow(img2, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/m.png")
                plt.imshow(img3, cmap='gray')
                plt.savefig(f"peak_LV_images/{dir}/b.png")














