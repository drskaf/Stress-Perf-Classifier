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
