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
from keras.preprocessing.image import ImageDataGenerator
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

          ### Training apical AHA segments classification ###

# Load images and label them
(images, labels, total) = utils.load_multislice(args["directory"], patient_info, 224)
print(images.shape)

testX = images
survival_yhat = labels

# Load trained image model1
json_file = open('models/aha_ant_VGG19.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("models/aha_ant_VGG19_my_model.best.hdf5")

# Predict with model
preds1 = model1.predict(testX)
print(preds1[:20])
pred_test_cl1 = pred_test_cl1 = list(map(lambda x: 0 if x[0]<0.5 else 1, preds1))
pred_test_cl1 = list(map(lambda x: 0 if x<0.5 else 1, pred_test_cl1))
print(pred_test_cl1[:20])
print(survival_yhat[:20])

prob_outputs1 = {
    "pred": pred_test_cl1,
    "actual_value": survival_yhat
}
prob_output_df1 = pd.DataFrame(prob_outputs1)
print(prob_output_df1.head(20))

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl1))
print('Average AHA VGG19 ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl1))
print('Average AHA VGG19 Accuracy score:',accuracy_score(survival_yhat, pred_test_cl1))
print('Average AHA VGG19 score:',f1_score(survival_yhat, pred_test_cl1))

# Load trained model2
#json_file = open('models/AHA13/aha13_LeNet.json','r')
#model2_json = json_file.read()
#json_file.close()
#model2 = model_from_json(model2_json)
#model2.load_weights("models/AHA13/aha13_LeNet_my_model.best.hdf5")

# Predict with model
#preds2 = model1.predict(testX)
#pred_test_cl2 = []
#for p in preds2:
 #   pred = np.argmax(p, axis=0)
  #  pred_test_cl2.append(pred)
#print(pred_test_cl2[:5])

#prob_outputs2 = {
 #   "pred": pred_test_cl2,
  #  "actual_value": survival_yhat
#}
#prob_output_df2 = pd.DataFrame(prob_outputs2)
#print(prob_output_df2.head())

# Evaluate model
#print(classification_report(survival_yhat, pred_test_cl2))
#print('Image CNN LeNet ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl2))
#print('Image CNN LeNet Accuracy score:',accuracy_score(survival_yhat, pred_test_cl2))
#print('Image CNN LeNet score:',f1_score(survival_yhat, pred_test_cl2))

# Load model3
#json_file = open('models/AHA13/aha13_AlexNet.json','r')
#model3_json = json_file.read()
#json_file.close()
#model3 = model_from_json(model3_json)
#model3.load_weights("models/AHA13/aha13_AlexNet_my_model.best.hdf5")

# Predict with model
#preds3 = model3.predict(testX)
#pred_test_cl3 = []
#for p in preds3:
 #   pred = np.argmax(p, axis=0)
  #  pred_test_cl3.append(pred)
#print(pred_test_cl3[:5])

#prob_outputs3 = {
 #   "pred": pred_test_cl3,
  #  "actual_value": survival_yhat
#}
#prob_output_df3 = pd.DataFrame(prob_outputs3)
#print(prob_output_df3.head(20))

# Evaluate model
#print(classification_report(survival_yhat, pred_test_cl3))
#print('Image CNN LeNet ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl3))
#print('Image CNN LeNet Accuracy score:',accuracy_score(survival_yhat, pred_test_cl3))
#print('Image CNN LeNet score:',f1_score(survival_yhat, pred_test_cl3))

# Plot ROC
fpr, tpr, _ = roc_curve(survival_yhat, preds1[:,0])
auc = round(roc_auc_score(survival_yhat, preds1[:,0]), 2)
plt.plot(fpr, tpr, label="Average AHA Classifiers AUC="+str(auc))
#fpr, tpr, _ = roc_curve(survival_yhat, preds2[:,0])
#auc = round(roc_auc_score(survival_yhat, preds2[:,0]), 2)
#plt.plot(fpr, tpr, label="ShallowNet , AUC="+str(auc))
#fpr, tpr, _ = roc_curve(survival_yhat, preds3[:,0])
#auc = round(roc_auc_score(survival_yhat, preds3[:,0]), 2)
#plt.plot(fpr, tpr, label="MiniVGG , AUC="+str(auc))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
#plt.title('AUC Models Comparison')
plt.show()

# Plot PR
precision, recall, thresholds = precision_recall_curve(survival_yhat, preds1[:,0])
label='%s (F1 Score:%0.2f)' % ('Average AHA Classifiers', average_precision_score(survival_yhat, preds1[:,0]))
plt.plot(recall, precision, label=label)
#precision, recall, thresholds = precision_recall_curve(survival_yhat, preds2[:,0])
#label='%s (F1 Score:%0.2f)' % ('ShallowNet', average_precision_score(survival_yhat, preds2[:,0]))
#plt.plot(recall, precision, label=label)
#precision, recall, thresholds = precision_recall_curve(survival_yhat, preds3[:,0])
#label='%s (F1 Score:%0.2f)' % ('MiniVGG', average_precision_score(survival_yhat, preds3[:,0]))
#plt.plot(recall, precision, label=label)
plt.xlim(0.1, 1.2)
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.title('F1 Score Models Comparison')
plt.show()
