import numpy as np
import pydicom
import os
import re
from skimage.transform import resize

def compose_perfusion_video(lstFilesDCM):
    """
    Args:
        lstFilesDCM (list of dirs): This is a list of the original DICOMs,
        where the ArrayDicom will be generated from.
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
    :param directory: the path to the folder where dicom images are stored
    :return: video files
    """

    for root, dirs, files in os.walk(directory, topdown=True):

        if len(files) > 10:
            subfolder = os.path.split(root)[0]
            folder = os.path.split(subfolder)[1]
            out_name = os.path.split(folder)[1] + '_' + os.path.split(root)[1]
            print("\nWorking on ", out_name)
            lstFilesDCM = []
            for filename in files:
                if ('dicomdir' not in filename.lower() and
                        'dirfile' not in filename.lower() and
                        filename[0] != '.' and
                        'npy' not in filename.lower() and
                        'png' not in filename.lower()):
                    lstFilesDCM.append(os.path.join(root, filename))

            print("Loading the data: {} files".format(len(lstFilesDCM)))
            video = \
                compose_perfusion_video(lstFilesDCM)
            
            return video

        else:
            for i in files[0:]:
                print("Loading the data: {} files".format(len(files)))
                video = pydicom.read_file(os.path.join(root, i))
                video = video.pixel_array
                
                return video
 
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]
