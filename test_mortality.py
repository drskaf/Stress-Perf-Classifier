import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, average_precision_score, precision_score, recall_score
import pickle
from random import sample
from sklearn.model_selection import train_test_split
import utils
from sklearn.preprocessing import StandardScaler
import tempfile
from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import mcnemar_table, mcnemar
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_regression

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
patient_df['Gender'] = patient_df['patient_GenderCode_x'].astype('category')
patient_df['Gender'] = patient_df['Gender'].cat.codes

# Load images
(df) = utils.load_perf_images('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/STRESS_test', patient_df, 224)
testX = np.array([x for x in df['Perf']])
print(testX.shape)
survival_yhat = np.array(df['Event_x'])
print(survival_yhat[:5])

# Load trained mixed model
# Define columns
categorical_col_listc = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender','Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_listc = ['Age_on_20.08.2021_x', 'LVEF_(%)']

def process_attributes(df):
    continuous = numerical_col_listc
    categorical = categorical_col_listc
    cs = MinMaxScaler()
    testContinuous = cs.fit_transform(df[continuous])

    # One-hot encode categorical data
    catBinarizer = LabelBinarizer().fit(df[categorical])
    testCategorical = catBinarizer.transform(df[categorical])

    # Construct our training and testing data points by concatenating
    # the categorical features with the continous features
    testX = np.hstack([testCategorical, testContinuous])

    return (testX)

testImageX = testX

testAttrX = process_attributes(df)
testAttrX = np.array(testAttrX)

# Load model
json_file = open('models/Mortality/mortality.json','r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('models/Mortality/mortality_my_model.best.hdf5')

# Predict with model
preds = model.predict([testAttrX, testImageX])
precision, recall, thresholds = precision_recall_curve(survival_yhat, preds[:,0])
pred_test_cl = np.array(list(map(lambda x: 0 if x<np.mean(thresholds) else 1, preds)))
print(pred_test_cl[:5])

prob_outputs = {
    "pred": pred_test_cl,
    "actual_value": survival_yhat
}

prob_output_df = pd.DataFrame(prob_outputs)
print(prob_output_df.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl))
print('HNN ROCAUC score:',roc_auc_score(survival_yhat, preds[:,0]))
print('HNN Accuracy score:',accuracy_score(survival_yhat, pred_test_cl))
print('HNN Precision:', np.mean(precision))
print('HNN recall:', np.mean(recall))
print('HNN F1 Score:',average_precision_score(survival_yhat, preds[:,0]))

# plot confusion matrix
cm = confusion_matrix(survival_yhat, pred_test_cl)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Define predictors

dir = os.listdir('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/STRESS_images')
dirOne = []
for d in dir:
    if '.DS_Store' in dir:
        dir.remove('.DS_Store')
    d = d.rstrip('_')
    dirOne.append(d)

df1 = pd.DataFrame(dirOne, columns=['index'])
df1['ID'] = df1['index'].astype(int)

# Create dataframe
data = patient_df.merge(df1, on=['ID'])
print(len(data))

trainx, testx = utils.patient_dataset_splitter(data, patient_key='patient_TrustNumber')
y_train = np.array(data['Event_x'])
y_test = np.array(df['Event_x'])
x_train = np.array(process_attributes(data))
x_test = np.array(process_attributes(df))

# fit Linear model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
print('LR Intercept:', lr_model.intercept_)
print('LR Coefficient:', lr_model.coef_)
lr_predict = lr_model.predict(x_test)
print(lr_predict[:5])
lr_preds = lr_model.predict_proba(x_test)[:,1]
print(lr_preds[:5])

# Information gain graph
continuous = data[numerical_col_listc]
categorical = data[categorical_col_listc]
Xtrain = pd.concat([continuous, categorical], axis=1)
Xtrain = Xtrain.rename(columns={"Chronic_kidney_disease_(disorder)": "CKD", "Essential_hypertension": "HTN",
              "Heart_failure_(disorder)": "Heart Failure", "Smoking_history": "Smoking",
              "Myocardial_infarction_(disorder)": "Myocardial Infarction", "Diabetes_mellitus_(disorder)": "DM",
              "Cerebrovascular_accident_(disorder)": "CVA", "Age_on_20.08.2021_x": "Age", "LVEF_(%)": "LV Ejection Fraction"} )
Ytrain = data['Event_x']
mutual_info = mutual_info_regression(Xtrain, Ytrain)
mutual_info = pd.Series(mutual_info)
mutual_info.index = Xtrain.columns
mi = mutual_info.sort_values(ascending=False)
print("Predictors importance:", mutual_info)
plt.barh(y=mutual_info.index, width=mi)
plt.show()

# Plot ROC
fpr, tpr, _ = roc_curve(survival_yhat, preds[:,0])
std_tpr = np.std(tpr)
tprs_upper = tpr + std_tpr
tprs_lower = tpr - std_tpr
auc = round(roc_auc_score(survival_yhat, preds[:,0]), 2)
plt.plot(fpr, tpr, label="HNN AUC="+str(auc), color='purple')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.fill_between(fpr, tprs_lower,tprs_upper, color='purple', alpha=.20)
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('AUC Models Comparison')
plt.grid()
plt.show()

fpr, tpr, _ = roc_curve(survival_yhat, preds[:,0])
std_tpr = np.std(tpr)
tprs_upper = tpr + std_tpr
tprs_lower = tpr - std_tpr
auc = round(roc_auc_score(survival_yhat, preds[:,0]), 2)
plt.plot(fpr, tpr, label="HNN AUC="+str(auc), color='purple')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()

