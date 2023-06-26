import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
from keras.models import model_from_json, load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, average_precision_score
import pickle
from random import sample
from sklearn.model_selection import train_test_split
import scipy.stats
import utils
from sklearn.preprocessing import StandardScaler
from keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load dataframe and creating classes
patient_info = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

# Load images and label them
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


# Load trained models
# AHA1
(testX, survival_yhat1) = load_basal_slice(args["directory"], patient_info, 224, name='p_basal anterior')
json_file = open('aha10_VGG19.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("aha10_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds1 = model1.predict(testX)

# AHA2
(testX, survival_yhat2) = load_basal_slice(args["directory"], patient_info, 224, name='p_basal anteroseptum')
json_file = open('models/AHA2/aha2_VGG19.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights("models/AHA2/aha2_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds2 = model2.predict(testX)

# AHA3
(testX, survival_yhat3) = load_basal_slice(args["directory"], patient_info, 224, name='p_basal inferoseptum')
json_file = open('models/AHA3/aha3_VGG19.json','r')
model3_json = json_file.read()
json_file.close()
model3 = model_from_json(model3_json)
model3.load_weights("models/AHA3/aha3_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds3 = model3.predict(testX)

# AHA4
(testX, survival_yhat4) = load_basal_slice(args["directory"], patient_info, 224, name='p_basal inferior')
json_file = open('models/AHA4/aha4_VGG19.json','r')
model4_json = json_file.read()
json_file.close()
model4 = model_from_json(model4_json)
model4.load_weights("models/AHA4/aha4_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds4 = model4.predict(testX)

# AHA5
(testX, survival_yhat5) = load_basal_slice(args["directory"], patient_info, 224, name='p_basal inferolateral')
json_file = open('models/AHA5/aha5_VGG19.json','r')
model5_json = json_file.read()
json_file.close()
model5 = model_from_json(model5_json)
model5.load_weights("models/AHA5/aha5_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds5 = model5.predict(testX)

# AHA6
(testX, survival_yhat6) = load_basal_slice(args["directory"], patient_info, 224, name='p_basal anterolateral')
json_file = open('models/AHA6/aha6_VGG19.json','r')
model6_json = json_file.read()
json_file.close()
model6 = model_from_json(model6_json)
model6.load_weights("models/AHA6/aha6_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds6 = model6.predict(testX)

# AHA7
(testX, survival_yhat7) = load_mid_slice(args["directory"], patient_info, 224, name='p_mid anterior')
json_file = open('models/AHA7/aha7_VGG19.json','r')
model7_json = json_file.read()
json_file.close()
model7 = model_from_json(model7_json)
model7.load_weights("models/AHA7/aha7_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds7 = model7.predict(testX)

# AHA8
(testX, survival_yhat8) = load_mid_slice(args["directory"], patient_info, 224, name='p_mid anteroseptum')
json_file = open('models/AHA8/aha8_VGG19.json','r')
model8_json = json_file.read()
json_file.close()
model8 = model_from_json(model8_json)
model8.load_weights("models/AHA8/aha8_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds8 = model8.predict(testX)

# AHA9
(testX, survival_yhat9) = load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferoseptum')
json_file = open('models/AHA9/aha9_VGG19.json','r')
model9_json = json_file.read()
json_file.close()
model9 = model_from_json(model9_json)
model9.load_weights("models/AHA9/aha9_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds9 = model9.predict(testX)

# AHA10
(testX, survival_yhat10) = load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferior')
json_file = open('models/AHA10/aha10_VGG19.json','r')
model10_json = json_file.read()
json_file.close()
model10 = model_from_json(model10_json)
model10.load_weights("models/AHA10/aha10_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds10 = model10.predict(testX)

# AHA11
(testX, survival_yhat11) = load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferolateral')
json_file = open('models/AHA11/aha11_VGG19.json','r')
model11_json = json_file.read()
json_file.close()
model11 = model_from_json(model11_json)
model11.load_weights("models/AHA11/aha11_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds11 = model11.predict(testX)

# AHA12
(testX, survival_yhat12) = load_mid_slice(args["directory"], patient_info, 224, name='p_mid anterolateral')
json_file = open('models/AHA12/aha12_VGG19.json','r')
model12_json = json_file.read()
json_file.close()
model12 = model_from_json(model12_json)
model12.load_weights("models/AHA12/aha12_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds12 = model12.predict(testX)

# AHA13
(testX, survival_yhat13) = load_apical_slice(args["directory"], patient_info, 224, name='p_apical anterior')
json_file = open('models/AHA13/aha13_VGG19.json','r')
model13_json = json_file.read()
json_file.close()
model13 = model_from_json(model13_json)
model13.load_weights("models/AHA13/aha13_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds13 = model13.predict(testX)

# AHA14
(testX, survival_yhat14) = load_apical_slice(args["directory"], patient_info, 224, name='p_apical septum')
json_file = open('models/AHA14/aha14_VGG19.json','r')
model14_json = json_file.read()
json_file.close()
model14 = model_from_json(model14_json)
model14.load_weights("models/AHA14/aha14_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds14 = model14.predict(testX)

# AHA15
(testX, survival_yhat15) = load_apical_slice(args["directory"], patient_info, 224, name='p_apical inferior')
json_file = open('models/AHA15/aha15_VGG19.json','r')
model15_json = json_file.read()
json_file.close()
model15 = model_from_json(model15_json)
model15.load_weights("models/AHA15/aha15_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds15 = model15.predict(testX)

# AHA16
(testX, survival_yhat16) = load_apical_slice(args["directory"], patient_info, 224, name='p_apical lateral')
json_file = open('models/AHA16/aha16_VGG19.json','r')
model16_json = json_file.read()
json_file.close()
model16 = model_from_json(model16_json)
model16.load_weights("models/AHA16/aha16_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds16 = model16.predict(testX)

# Concatenate predictions and ground truth
predictions = np.concatenate((preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, preds13, preds14, preds15, preds16))
ground_truth = np.concatenate((survival_yhat1, survival_yhat2, survival_yhat3, survival_yhat4, survival_yhat5, survival_yhat6, survival_yhat7, survival_yhat8, survival_yhat9, survival_yhat10, survival_yhat11, survival_yhat12, survival_yhat13, survival_yhat14, survival_yhat15, survival_yhat16))

lad_pred = np.concatenate((preds1, preds2, preds7, preds8, preds13, preds14))
lad_truth = np.concatenate((survival_yhat1, survival_yhat2, survival_yhat7, survival_yhat8, survival_yhat13, survival_yhat14))

rca_pred = np.concatenate((preds3, preds4, preds9, preds10, preds15))
rca_truth = np.concatenate((survival_yhat3, survival_yhat4, survival_yhat9, survival_yhat10, survival_yhat15))

lcx_pred = np.concatenate((preds5, preds6, preds11, preds12, preds16))
lcx_truth = np.concatenate((survival_yhat5, survival_yhat6, survival_yhat11, survival_yhat12, survival_yhat16))

# Plot ROC
fpr, tpr, _ = roc_curve(ground_truth, predictions[:,0])
auc = round(roc_auc_score(ground_truth, predictions[:,0]), 2)
plt.plot(fpr, tpr, label="Average AHA Classifier AUC="+str(auc), color='navy')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Total AHA Classifiers')
plt.grid()
plt.show()

fpr, tpr, _ = roc_curve(lad_truth, lad_pred[:,0])
auc = round(roc_auc_score(lad_truth, lad_pred[:,0]), 2)
plt.plot(fpr, tpr, label="Average LAD Classifier AUC="+str(auc), color='navy')
fpr, tpr, _ = roc_curve(lcx_truth, lcx_pred[:,0])
auc = round(roc_auc_score(lcx_truth, lcx_pred[:,0]), 2)
plt.plot(fpr, tpr, label="Average LCx Classifier AUC="+str(auc), color='green')
fpr, tpr, _ = roc_curve(rca_truth, rca_pred[:,0])
auc = round(roc_auc_score(rca_truth, rca_pred[:,0]), 2)
plt.plot(fpr, tpr, label="Average RCA Classifier AUC="+str(auc), color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Coronary Territory Classifiers')
plt.grid()
plt.show()

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

print('DeLong test for non-linear and linear predictions:', delong_roc_test(y_test, preds1[:,1], preds2[:,1]))






