import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
from keras.models import model_from_json, load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, average_precision_score
import scipy.stats
import utils

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load dataframe and creating classes
patient_info = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

# Load trained models
# AHA1
(testX, survival_yhat1) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal anterior')
json_file = open('aha10_VGG19.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("aha10_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds1 = model1.predict(testX)

# AHA2
(testX, survival_yhat2) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal anteroseptum')
json_file = open('models/AHA2/aha2_VGG19.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights("models/AHA2/aha2_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds2 = model2.predict(testX)

# AHA3
(testX, survival_yhat3) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal inferoseptum')
json_file = open('models/AHA3/aha3_VGG19.json','r')
model3_json = json_file.read()
json_file.close()
model3 = model_from_json(model3_json)
model3.load_weights("models/AHA3/aha3_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds3 = model3.predict(testX)

# AHA4
(testX, survival_yhat4) = utilsload_basal_slice(args["directory"], patient_info, 224, name='p_basal inferior')
json_file = open('models/AHA4/aha4_VGG19.json','r')
model4_json = json_file.read()
json_file.close()
model4 = model_from_json(model4_json)
model4.load_weights("models/AHA4/aha4_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds4 = model4.predict(testX)

# AHA5
(testX, survival_yhat5) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal inferolateral')
json_file = open('models/AHA5/aha5_VGG19.json','r')
model5_json = json_file.read()
json_file.close()
model5 = model_from_json(model5_json)
model5.load_weights("models/AHA5/aha5_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds5 = model5.predict(testX)

# AHA6
(testX, survival_yhat6) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal anterolateral')
json_file = open('models/AHA6/aha6_VGG19.json','r')
model6_json = json_file.read()
json_file.close()
model6 = model_from_json(model6_json)
model6.load_weights("models/AHA6/aha6_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds6 = model6.predict(testX)

# AHA7
(testX, survival_yhat7) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid anterior')
json_file = open('models/AHA7/aha7_VGG19.json','r')
model7_json = json_file.read()
json_file.close()
model7 = model_from_json(model7_json)
model7.load_weights("models/AHA7/aha7_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds7 = model7.predict(testX)

# AHA8
(testX, survival_yhat8) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid anteroseptum')
json_file = open('models/AHA8/aha8_VGG19.json','r')
model8_json = json_file.read()
json_file.close()
model8 = model_from_json(model8_json)
model8.load_weights("models/AHA8/aha8_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds8 = model8.predict(testX)

# AHA9
(testX, survival_yhat9) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferoseptum')
json_file = open('models/AHA9/aha9_VGG19.json','r')
model9_json = json_file.read()
json_file.close()
model9 = model_from_json(model9_json)
model9.load_weights("models/AHA9/aha9_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds9 = model9.predict(testX)

# AHA10
(testX, survival_yhat10) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferior')
json_file = open('models/AHA10/aha10_VGG19.json','r')
model10_json = json_file.read()
json_file.close()
model10 = model_from_json(model10_json)
model10.load_weights("models/AHA10/aha10_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds10 = model10.predict(testX)

# AHA11
(testX, survival_yhat11) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferolateral')
json_file = open('models/AHA11/aha11_VGG19.json','r')
model11_json = json_file.read()
json_file.close()
model11 = model_from_json(model11_json)
model11.load_weights("models/AHA11/aha11_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds11 = model11.predict(testX)

# AHA12
(testX, survival_yhat12) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid anterolateral')
json_file = open('models/AHA12/aha12_VGG19.json','r')
model12_json = json_file.read()
json_file.close()
model12 = model_from_json(model12_json)
model12.load_weights("models/AHA12/aha12_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds12 = model12.predict(testX)

# AHA13
(testX, survival_yhat13) = utils.load_apical_slice(args["directory"], patient_info, 224, name='p_apical anterior')
json_file = open('models/AHA13/aha13_VGG19.json','r')
model13_json = json_file.read()
json_file.close()
model13 = model_from_json(model13_json)
model13.load_weights("models/AHA13/aha13_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds13 = model13.predict(testX)

# AHA14
(testX, survival_yhat14) = utils.load_apical_slice(args["directory"], patient_info, 224, name='p_apical septum')
json_file = open('models/AHA14/aha14_VGG19.json','r')
model14_json = json_file.read()
json_file.close()
model14 = model_from_json(model14_json)
model14.load_weights("models/AHA14/aha14_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds14 = model14.predict(testX)

# AHA15
(testX, survival_yhat15) = utils.load_apical_slice(args["directory"], patient_info, 224, name='p_apical inferior')
json_file = open('models/AHA15/aha15_VGG19.json','r')
model15_json = json_file.read()
json_file.close()
model15 = model_from_json(model15_json)
model15.load_weights("models/AHA15/aha15_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds15 = model15.predict(testX)

# AHA16
(testX, survival_yhat16) = utils.load_apical_slice(args["directory"], patient_info, 224, name='p_apical lateral')
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






