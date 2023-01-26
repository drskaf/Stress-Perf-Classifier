import numpy as np
import pydicom
import os
import re

def read_perfusion_slices(lstFilesDCM):

    # reads dicom images from given path
    # code is mostly taken from
    # https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/
    # makes use of the pydicom module

    # it further uses position information from metadata to order slices
    # and the returns the 3 slices separately

    import pydicom
    import os
    import re

    RefDs = pydicom.read_file(lstFilesDCM[0])
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    ConstPixelSpacing = (
        float(RefDs.PixelSpacing[0]),
        float(RefDs.PixelSpacing[1]),
        float(RefDs.SliceThickness),
    )

    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    slicePosition = []
    sliceOrientation = []
    indices = []

    name_numbers = []

    # order DICOM list
    for filenameDCM in lstFilesDCM:
        tmpFilename = filenameDCM.split(os.sep)[-2]
        name_numbers.append([float(s) for s in re.findall(r"-?\d+\.?\d*", tmpFilename)])

    name_numbers = np.abs(name_numbers)
    nums_to_order = []
    for i in range(len(name_numbers)):
        tmp = 0
        for j in range(len(name_numbers[i])):
            tmp += name_numbers[i][j]
        nums_to_order.append(tmp)

    true_order = np.argsort(nums_to_order)

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
        # store the raw image data
        ArrayDicom[:, :, i] = ds.pixel_array

    return ArrayDicom, slicePosition, sliceOrientation, indices, allData
