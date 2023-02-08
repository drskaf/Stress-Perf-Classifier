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
        combined 3D files with 1st dimension as frames number
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
            subfolder = os.path.split(root)[0]
            folder = os.path.split(subfolder)[1]
            out_name = os.path.split(folder)[1] + '_' + os.path.split(root)[1]
            print("\nWorking on ", out_name)
            for i in files[0:]:
                print("Loading the data: {} files".format(len(files)))
                video = pydicom.read_file(os.path.join(root, i), force=True)
                video.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                video = video.pixel_array
                
                return video
 

def load_label_png(directory, df_info, im_size):
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
    for root, dirs, files in os.walk(directory, topdown=True):

        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            dir_name = int(folder)
            dir_path = os.path.join(directory, folder)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                
                # Initiate lists of images and labels
                images = []
                labels = []

                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    img1 = mpimg.imread(os.path.join(dir_path, file))
                    img1 = resize(img1, (im_size, im_size))
                if file_name == 'm':
                    img2 = mpimg.imread(os.path.join(dir_path, file))
                    img2 = resize(img2, (im_size, im_size))
                if file_name == 'a':
                    img3 = mpimg.imread(os.path.join(dir_path, file))
                    img3 = resize(img3, (im_size, im_size))

                    out = cv2.vconcat([img1, img2, img3])

                    # Defining labels
                    patient_info = df_info[df_info["ID"].values == dir_name]
                    the_class = patient_info["Event"].astype(int)

                    images.append(out)
                    labels.append(the_class)

                    return images, labels
