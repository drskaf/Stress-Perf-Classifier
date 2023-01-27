import numpy as np
import pydicom
import os
import re
from skimage.transform import resize

def read_perfusion_slices(lstFilesDCM):
    """By Cian and Nathan: Read the image slices into arrays/tensors from the DICOM. This function
    makes use of the pydicom modules and further uses position information from
    metadata to order slices. The slices are returned separately.
    Code is mostly taken from:
    https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/
    Args:
        lstFilesDCM (list of dirs): This is a list of the original DICOMs,
        where the ArrayDicom will be read from generated from.
    Returns:
        ArrayDicom (array): This is image series.
        slicePosition (array): The corresponding slice positions for the image
        series.
        sliceOrientation (array): The corresponding slice orientations for the
        image series.
        indices (array): This is an array of the corresponding indices for the
        image series.
        meta (array): This is the corresponding meta data to each image in the
        series.
    """

    RefDs = pydicom.read_file(lstFilesDCM[0])
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    slicePosition = []
    sliceOrientation = []
    indices = []

    name_numbers = []

    # order DICOM list
    for filenameDCM in lstFilesDCM:
        tmpFilename = (filenameDCM.split(os.sep)[-1]).split('-')[-2]
        name_numbers.append([float(s) for s in re.findall(r'-?\d+\.?\d*',
                                                          tmpFilename)])

    name_numbers = np.abs(name_numbers)
    nums_to_order = []
    for i in range(len(name_numbers)):
        tmp = 0
        for j in range(len(name_numbers[i])):
            tmp += name_numbers[i][j]
        nums_to_order.append(tmp)

    true_order = (np.argsort(nums_to_order))

    allData = []

    # loop through all the DICOM files
    for i in range(len(true_order)):
        filenameDCM = lstFilesDCM[true_order[i]]
        # read the file
        ds = pydicom.read_file(filenameDCM)
        allData.append(ds)
        if ds.ImagePositionPatient not in slicePosition:
            slicePosition.append(ds.ImagePositionPatient)
            sliceOrientation.append(ds.ImageOrientationPatient)
            indices.append(slicePosition.index(ds.ImagePositionPatient))

            ArrayDicom[:, :, i] = ds.pixel_array

    return ArrayDicom, slicePosition, sliceOrientation, indices, allData

