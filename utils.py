import numpy as np
import pydicom
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
import glob
from collections import Counter
import tensorflow as tf
import functools
import dicom2nifti
from pathlib import Path
import nibabel as nib
import dicom2nifti.settings as settings


def load_multislice(directory, df, im_size):
    """
    Read through .png images in sub-folders, read through label .csv file and
    annotate
    Args:
     directory: path to the data directory
     df_info: .csv file containing the label information
     im_size: target image size
    Return:
        resized images with their labels
    """
    # Initiate lists of images and labels
    images = []
    labels = []
    total = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            info_df = df[df['ID'] == int(folder_strip)]
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    the_class = np.array(info_df[['p_basal anterior','p_basal anteroseptum']]) #,'p_basal inferoseptum','p_basal inferior'
                        #,'p_basal inferolateral', 'p_basal anterolateral']])
                    the_class = np.squeeze(the_class)
                    tot = _sum(the_class)
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    out = cv2.merge([gray, gray, gray])
                    # out = gray[..., np.newaxis]
                    images.append(out)
                    labels.append(the_class)
                    total.append(tot)


                elif file_name == 'm':
                    the_class = np.array(info_df[['p_mid anterior','p_mid anteroseptum']]) #,'p_mid inferoseptum','p_mid inferior',
                                       #'p_mid inferolateral','p_mid anterolateral']])
                    the_class = np.squeeze(the_class)
                    tot = _sum(the_class)
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    out = cv2.merge([gray, gray, gray])
                    # out = gray[..., np.newaxis]
                    images.append(out)
                    labels.append(the_class)
                    total.append(tot)


                elif file_name == 'a':
                    the_class = np.array(info_df[['p_apical anterior', 'p_apical septum']]) #,'p_apical inferior','p_apical lateral']])
                    the_class = np.squeeze(the_class)
                    #the_class = np.pad(the_class, (0,2), 'constant', constant_values=0)
                    tot = _sum(the_class)
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    out = cv2.merge([gray, gray, gray])
                    # out = gray[..., np.newaxis]
                    images.append(out)
                    labels.append(the_class)
                    total.append(tot)

    return (np.array(images), labels, total)


def patient_dataset_splitter(df, patient_key='patient_TrustNumber'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id
    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
    '''

    df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    train_size = round(total_values * 0.8)
    train = df[df[patient_key].isin(unique_values[:train_size])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[train_size:])].reset_index(drop=True)

    return train, validation


def compose_perfusion_video(lstFilesDCM):
    """
    Args:
        lstFilesDCM (list of dirs): This is a list of the original DICOMs,
        where the ArrayDicom will be generated from.
    Return:
        3D arrays with one dimension for frames number
    """

    RefDs = pydicom.read_file(lstFilesDCM[0])
    ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))

    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for i in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(i)

        for j in range(len(lstFilesDCM)):
            ArrayDicom[j, :, :] = ds.pixel_array

    return ArrayDicom


def load_perfusion_data(directory):
    """
    Args:
     directory: the path to the folder where dicom images are stored
    Return:
        combined 3D files with 1st dimension as frames depth
    """

    videoStackList = []
    indicesStackList = []
    videoSingleList = []
    indicesSingleList = []
    videorawList = []

    dir_paths = sorted(glob.glob(os.path.join(directory, "*")))
    for dir_path in dir_paths:
        file_paths = sorted(glob.glob(os.path.join(dir_path, "*.dcm")))

        if len(file_paths) > 10:
            folder = os.path.split(dir_path)[1]
            print("\nWorking on ", folder)
            vlist = []
            vrlist = []
            for file_path in file_paths:
                imgraw = pydicom.read_file(file_path)
                vrlist.append(imgraw)
                img = imgraw.pixel_array
                vlist.append(img)
            videorawList.append(vrlist)
            videoSingleList.append(vlist)
            indicesSingleList.append(folder)

        else:
            folder = os.path.split(dir_path)[1]
            print("\nWorking on ", folder)
            for i in file_paths[0:]:
                # Read stacked dicom and add to list
                videoraw = pydicom.read_file(os.path.join(dir_path, i), force=True)
                videoStackList.append(videoraw)
                indicesStackList.append(folder)

    return videorawList, videoSingleList, indicesSingleList, videoStackList, indicesStackList



def centre_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = width//2

    if new_height is None:
        new_height = height//2

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        centre_cropped_img = img[top:bottom, left:right]
    else:
        centre_cropped_img = img[top:bottom, left:right, ...]

    return centre_cropped_img


def load_basal_slice(directory, df, im_size, name):
    """
    Read through .png images in sub-folders, read through label .csv file and
    annotate
    Args:
     directory: path to the data directory
     df_info: .csv file containing the label information
     im_size: target image size
     name: name of the AHA segment
    Return:
        resized images with their labels
    """
    # Initiate lists of images and labels
    images = []
    labels = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            info_df = df[df['ID'] == int(folder_strip)]
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    the_class = np.array(info_df[name])
                    #the_class = np.squeeze(the_class)
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    out = cv2.merge([gray, gray, gray])

                    images.append(out)
                    labels.append(the_class)
                else:
                    continue

    return (np.array(images), np.array(labels))


def load_mid_slice(directory, df, im_size, name):
    """
    Read through .png images in sub-folders, read through label .csv file and
    annotate
    Args:
     directory: path to the data directory
     df_info: .csv file containing the label information
     im_size: target image size
     name: name of the AHA segment
    Return:
        resized images with their labels
    """
    # Initiate lists of images and labels
    images = []
    labels = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            info_df = df[df['ID'] == int(folder_strip)]
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'm':
                    the_class = np.array(info_df[name])
                    #the_class = np.squeeze(the_class)
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    out = cv2.merge([gray, gray, gray])

                    images.append(out)
                    labels.append(the_class)
                else:
                    continue

    return (np.array(images), np.array(labels))


def load_apical_slice(directory, df, im_size, name):
    """
    Read through .png images in sub-folders, read through label .csv file and
    annotate
    Args:
     directory: path to the data directory
     df_info: .csv file containing the label information
     im_size: target image size
     name: name of the AHA segment
    Return:
        resized images with their labels
    """
    # Initiate lists of images and labels
    images = []
    labels = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            folder_strip = folder.rstrip('_')
            info_df = df[df['ID'] == int(folder_strip)]
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                dir_path = os.path.join(directory, folder)
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'a':
                    the_class = np.array(info_df[name])
                    #the_class = np.squeeze(the_class)
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    out = cv2.merge([gray, gray, gray])

                    images.append(out)
                    labels.append(the_class)
                else:
                    continue

    return (np.array(images), np.array(labels))



