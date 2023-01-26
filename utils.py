import numpy as np
import pydicom
import os
import re
from skimage.transform import resize

def read_perfusion_slices(lstFilesDCM):

    # By Cian and Nathan: reads dicom images from given path
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


def order_slices(ArrayDicom, slicePosition, sliceOrientation, indices, lst,
                 meta):
    """By Cian and Nathan: Order the slices based on the z-axis.
    Args:
        ArrayDicom (array): This is image series.
        slicePosition (array): The corresponding slice positions for the image
        series.
        sliceOrientation (array): The corresponding slice orientations for the
        image series.
        indices (array): This is an array of the corresponding indices for the
        image series.
        lst (list): This is a list of the original DICOMs, where the
        ArrayDicom was generated from.
        meta (array): This is the corresponding meta data to each image in the
        series.
    Returns:
        slices (dict of array): The slices,
        slices_lst (dict of array): The list of the DICOMs.
        slices_meta (dict of array): The corresponding meta data.
    """

    # orders in descending order of slice position
    order = np.zeros(len(slicePosition))

    for i in range(len(slicePosition)):
        order[i] = (np.dot(slicePosition[i],
                           np.cross(sliceOrientation[i][0:3],
                                    sliceOrientation[i][3:6])))
    indices = np.asarray(indices)
    difforder = np.abs(np.diff(order))
    for i in range(len(difforder)):
        if difforder[i] < 1E-5:
            k1 = len(np.where(indices == i)[0])
            k2 = len(np.where(indices == i+1)[0])
            if k2 > k1:
                indices[np.where(indices == i)[0]] = i+1
                order = np.delete(order, i)
            else:
                indices[np.where(indices == i+1)[0]] = i
                order = np.delete(order, i+1)

    idxOrder = np.argsort(order)[::-1]

    uniqueinds = np.unique(indices)
    for i in range(len(uniqueinds)):
        indices[np.where(indices == uniqueinds[i])[0]] = i

    slices = {}
    slices_meta = {}
    slices_lst = {}
    for i in range(len(idxOrder)):
        tmpindices = [idx for idx in range(len(indices))
                      if indices[idx] == idxOrder[i]]
        j = i+1
        slices['slices_%02d' % j] = ArrayDicom[:, :, tmpindices]
        slices_meta['slices_%02d' % j] = [meta[j] for j in tmpindices]
        slices_lst['slices_%02d' % j] = [lst[j] for j in tmpindices]

    for slice in slices:
        meta = slices_meta[slice]
        unorder_times = np.zeros(len(meta))
        for i in range(len(meta)):
            unorder_times[i] = float(meta[i].InstanceNumber)
            
        idxOrder = np.argsort(unorder_times)
        tmp_slice = np.zeros((ArrayDicom.shape[0],
                              ArrayDicom.shape[1],
                              len(unorder_times)))
        for i, j in enumerate(idxOrder):
            tmp_slice[..., i] = slices[slice][..., j]

        slices[slice] = tmp_slice
        slices_meta[slice] = [slices_meta[slice][j] for j in idxOrder]
        slices_lst[slice] = [slices_lst[slice][j] for j in idxOrder]

    return slices, slices_lst, slices_meta
