import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report
import pickle


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
ap.add_argument("-t", "--target", required=True, help="name of the target field")
args = vars(ap.parse_args())

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

# Loading images and labels
def load_test_data(directory, target, df, im_size):
    # initialize our images array
    images = []
    labels = []
    indices = []
    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            dir_path = os.path.join(directory, folder)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
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
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    gray = resize(gray, (224, 224))
                    out = gray[..., np.newaxis]

                    # Defining labels
                    patient_info = df[df["ID"].values == int(folder)]
                    the_class = patient_info[target].astype(int)

                    images.append(out)
                    labels.append(the_class)
                    indices.append(int(folder))

    return (np.array(images), np.array(labels), indices)

(testX, testy, indices) = load_test_data(args["directory"], args["target"], patient_df, 224)
le = LabelEncoder().fit(testy)
testY = to_categorical(le.transform(testy), 2)
df = pd.DataFrame(indices, columns=['ID'])
info_df = pd.merge(df, patient_df, on=['ID'])

# Load trained model 1
json_file = open('models/image_model_LeNet.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("models/#image_mortality_predictor_LeNet.best.hdf5")

# Predict with model
preds1 = model1.predict(testX)
pred_test_cl1 = []
for p in preds1:
    pred = np.argmax(p, axis=0)
    pred_test_cl1.append(pred)
print(pred_test_cl1[:5])
#print(len(pred_test_cl))
survival_yhat = list(info_df[args["target"]].values)
#print(len(survival_yhat))
print(survival_yhat[:5])

prob_outputs1 = {
    "pred": pred_test_cl1,
    "actual_value": survival_yhat
}
prob_output_df1 = pd.DataFrame(prob_outputs1)
print(prob_output_df1.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl1))
print('Image CNN LeNet ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl1))
print('Image CNN LeNet Accuracy score:',accuracy_score(survival_yhat, pred_test_cl1))
print('Image CNN LeNet score:',f1_score(survival_yhat, pred_test_cl1))

# Load trained model 2
json_file = open('models/image_model_LeNetreg.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights("models/#image_mortality_predictor_LeNetreg.best.hdf5")

# Predict with model
preds2 = model2.predict(testX)
pred_test_cl2 = []
for p in preds2:
    pred = np.argmax(p, axis=0)
    pred_test_cl2.append(pred)
print(pred_test_cl2[:5])

prob_outputs2 = {
    "pred": pred_test_cl2,
    "actual_value": survival_yhat
}
prob_output_df2 = pd.DataFrame(prob_outputs2)
print(prob_output_df2.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl1))
print('Image CNN LeNet with regularisation ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl2))
print('Image CNN LeNet with regularisation Accuracy score:',accuracy_score(survival_yhat, pred_test_cl2))
print('Image CNN LeNet with regularisation F1 score:',f1_score(survival_yhat, pred_test_cl2))

# Load trained model 3
json_file = open('models/image_model_ShallowNet.json','r')
model3_json = json_file.read()
json_file.close()
model3 = model_from_json(model3_json)
model3.load_weights("models/#image_mortality_predictor_ShallowNet.best.hdf5")

# Predict with model
preds3 = model3.predict(testX)
pred_test_cl3 = []
for p in preds3:
    pred = np.argmax(p, axis=0)
    pred_test_cl3.append(pred)
print(pred_test_cl3[:5])

prob_outputs3 = {
    "pred": pred_test_cl3,
    "actual_value": survival_yhat
}
prob_output_df3 = pd.DataFrame(prob_outputs3)
print(prob_output_df3.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl3))
print('Image CNN ShallowNet ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl3))
print('Image CNN ShallowNet Accuracy score:',accuracy_score(survival_yhat, pred_test_cl3))
print('Image CNN ShallowNet F1 score:',f1_score(survival_yhat, pred_test_cl3))

# Load trained model 4
def load_test_data(directory, target, df, im_size):
    # initialize our images array
    images = []
    labels = []
    indices = []
    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            dir_path = os.path.join(directory, folder)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
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
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    gray = resize(gray, (224, 224))
                    out = cv2.merge([gray, gray, gray])

                    # Defining labels
                    patient_info = df[df["ID"].values == int(folder)]
                    the_class = patient_info[target].astype(int)

                    images.append(out)
                    labels.append(the_class)
                    indices.append(int(folder))

    return (np.array(images), np.array(labels), indices)

(testX, testy, indices) = load_test_data(args["directory"], args["target"], patient_df, 224)
le = LabelEncoder().fit(testy)
testY = to_categorical(le.transform(testy), 2)
df = pd.DataFrame(indices, columns=['ID'])
info_df = pd.merge(df, patient_df, on=['ID'])

json_file = open('models/image_model_vgg19.json','r')
model4_json = json_file.read()
json_file.close()
model4 = model_from_json(model4_json)
model4.load_weights("models/#image_mortality_predictor_vgg19.best.hdf5")

# Predict with model
preds4 = model4.predict(testX)
pred_test_cl4 = []
for p in preds4:
    pred = np.argmax(p, axis=0)
    pred_test_cl4.append(pred)
print(pred_test_cl4[:5])

prob_outputs4 = {
    "pred": pred_test_cl4,
    "actual_value": survival_yhat
}
prob_output_df4 = pd.DataFrame(prob_outputs4)
print(prob_output_df4.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl4))
print('Image CNN VGG19Transfer ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl4))
print('Image CNN VGG19Transfer Accuracy score:',accuracy_score(survival_yhat, pred_test_cl4))
print('Image CNN VGG19Transfer F1 score:',f1_score(survival_yhat, pred_test_cl4))

# Load trained mixed model
# Define columns
categorical_col_list = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender', 'Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_list= ['Age_on_20.08.2021_x', 'LVEF_(%)']

def load_images(directory, im_size):
    # initialize our images array
    images = []
    indices = []
    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            dir_path = os.path.join(directory, folder)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
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
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    out = gray[..., np.newaxis]

                    indices.append(int(folder))
                    images.append(out)

    return (np.array(images), indices)

(testImageX, indices) = load_images(args["directory"], im_size=224)
testImageX = testImageX / 255.0
df = pd.DataFrame(indices, columns=['ID'])
info_df = pd.merge(df, patient_df, on=['ID'])
testy = info_df.pop(args["target"])
le = LabelEncoder().fit(testy)
testY = to_categorical(le.transform(testy), 2)

def process_attributes(df):
    continuous = numerical_col_list
    categorical = categorical_col_list
    cs = MinMaxScaler()
    testContinuous = cs.fit_transform(df[continuous])

    # One-hot encode categorical data
    catBinarizer = LabelBinarizer().fit(df[categorical])
    testCategorical = catBinarizer.transform(df[categorical])

    # Construct our training and testing data points by concatenating
    # the categorical features with the continous features
    testX = np.hstack([testCategorical, testContinuous])

    return (testX)

testAttrX = process_attributes(info_df)

# Load model
model5 = keras.models.load_model('models/#mixed_mortality_predictor_LeNet.best.hdf5')

# Predict with model
preds5 = model5.predict([testAttrX, testImageX])
pred_test_cl5 = []
for p in preds5:
    pred = np.argmax(p, axis=0)
    pred_test_cl5.append(pred)
print(pred_test_cl5[:5])

prob_outputs5 = {
    "pred": pred_test_cl5,
    "actual_value": survival_yhat
}
prob_output_df5 = pd.DataFrame(prob_outputs5)
print(prob_output_df5.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl5))
print('Mixed NN ROCAUC score:',roc_auc_score(survival_yhat, pred_test_cl5))
print('Mixed NN Accuracy score:',accuracy_score(survival_yhat, pred_test_cl5))
print('Mixed NN F1 score:',f1_score(survival_yhat, pred_test_cl5))

# Plot RUC
fpr, tpr, _ = roc_curve(survival_yhat, preds1[:,1])
auc = round(roc_auc_score(survival_yhat, preds1[:,1]), 2)
plt.plot(fpr, tpr, label="Image CNN LeNet, AUC="+str(auc))
fpr, tpr, _ = roc_curve(survival_yhat, preds2[:,1])
auc = round(roc_auc_score(survival_yhat, preds2[:,1]), 2)
plt.plot(fpr, tpr, label="Image CNN LeNet with regularisation, AUC="+str(auc))
fpr, tpr, _ = roc_curve(survival_yhat, preds3[:,1])
auc = round(roc_auc_score(survival_yhat, preds3[:,1]), 2)
plt.plot(fpr, tpr, label="Image CNN ShallowNet, AUC="+str(auc))
fpr, tpr, _ = roc_curve(survival_yhat, preds4[:,1])
auc = round(roc_auc_score(survival_yhat, preds4[:,1]), 2)
plt.plot(fpr, tpr, label="Image CNN VGG19 Transfer Learning, AUC="+str(auc))
fpr, tpr, _ = roc_curve(survival_yhat, preds5[:,1])
auc = round(roc_auc_score(survival_yhat, preds5[:,1]), 2)
plt.plot(fpr, tpr, label="Mixed NN, AUC="+str(auc))
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Survival Models Comparison')
plt.show()
