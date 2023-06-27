import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
import utils
import keras
import tensorflow as tf
from keras.models import model_from_json, load_model

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=False, help="path to input case images")
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
preds1 = [0 if preds1 < 0.5 else 1]

# AHA2
(testX, survival_yhat2) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal anteroseptum')
json_file = open('models/AHA2/aha2_VGG19.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights("models/AHA2/aha2_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds2 = model2.predict(testX)
preds2 = [0 if preds2 < 0.5 else 1]

# AHA3
(testX, survival_yhat3) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal inferoseptum')
json_file = open('models/AHA3/aha3_VGG19.json','r')
model3_json = json_file.read()
json_file.close()
model3 = model_from_json(model3_json)
model3.load_weights("models/AHA3/aha3_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds3 = model3.predict(testX)
preds3 = [0 if preds3 < 0.5 else 1]

# AHA4
(testX, survival_yhat4) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal inferior')
json_file = open('models/AHA4/aha4_VGG19.json','r')
model4_json = json_file.read()
json_file.close()
model4 = model_from_json(model4_json)
model4.load_weights("models/AHA4/aha4_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds4 = model4.predict(testX)
preds4 = [0 if preds4 < 0.5 else 1]

# AHA5
(testX, survival_yhat5) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal inferolateral')
json_file = open('models/AHA5/aha5_VGG19.json','r')
model5_json = json_file.read()
json_file.close()
model5 = model_from_json(model5_json)
model5.load_weights("models/AHA5/aha5_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds5 = model5.predict(testX)
preds5 = [0 if preds5 < 0.5 else 1]

# AHA6
(testX, survival_yhat6) = utils.load_basal_slice(args["directory"], patient_info, 224, name='p_basal anterolateral')
json_file = open('models/AHA6/aha6_VGG19.json','r')
model6_json = json_file.read()
json_file.close()
model6 = model_from_json(model6_json)
model6.load_weights("models/AHA6/aha6_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds6 = model6.predict(testX)
preds6 = [0 if preds6 < 0.5 else 1]

# AHA7
(testX, survival_yhat7) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid anterior')
json_file = open('models/AHA7/aha7_VGG19.json','r')
model7_json = json_file.read()
json_file.close()
model7 = model_from_json(model7_json)
model7.load_weights("models/AHA7/aha7_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds7 = model7.predict(testX)
preds7 = [0 if preds7 < 0.5 else 1]

# AHA8
(testX, survival_yhat8) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid anteroseptum')
json_file = open('models/AHA8/aha8_VGG19.json','r')
model8_json = json_file.read()
json_file.close()
model8 = model_from_json(model8_json)
model8.load_weights("models/AHA8/aha8_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds8 = model8.predict(testX)
preds8 = [0 if preds8 < 0.5 else 1]

# AHA9
(testX, survival_yhat9) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferoseptum')
json_file = open('models/AHA9/aha9_VGG19.json','r')
model9_json = json_file.read()
json_file.close()
model9 = model_from_json(model9_json)
model9.load_weights("models/AHA9/aha9_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds9 = model9.predict(testX)
preds9 = [0 if preds9 < 0.5 else 1]

# AHA10
(testX, survival_yhat10) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferior')
json_file = open('models/AHA10/aha10_VGG19.json','r')
model10_json = json_file.read()
json_file.close()
model10 = model_from_json(model10_json)
model10.load_weights("models/AHA10/aha10_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds10 = model10.predict(testX)
preds10 = [0 if preds10 < 0.5 else 1]

# AHA11
(testX, survival_yhat11) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid inferolateral')
json_file = open('models/AHA11/aha11_VGG19.json','r')
model11_json = json_file.read()
json_file.close()
model11 = model_from_json(model11_json)
model11.load_weights("models/AHA11/aha11_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds11 = model11.predict(testX)
preds11 = [0 if preds11 < 0.5 else 1]

# AHA12
(testX, survival_yhat12) = utils.load_mid_slice(args["directory"], patient_info, 224, name='p_mid anterolateral')
json_file = open('models/AHA12/aha12_VGG19.json','r')
model12_json = json_file.read()
json_file.close()
model12 = model_from_json(model12_json)
model12.load_weights("models/AHA12/aha12_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds12 = model12.predict(testX)
preds12 = [0 if preds12 < 0.5 else 1]

# AHA13
(testX, survival_yhat13) = utils.load_apical_slice(args["directory"], patient_info, 224, name='p_apical anterior')
json_file = open('models/AHA13/aha13_VGG19.json','r')
model13_json = json_file.read()
json_file.close()
model13 = model_from_json(model13_json)
model13.load_weights("models/AHA13/aha13_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds13 = model13.predict(testX)
preds13 = [0 if preds13 < 0.5 else 1]

# AHA14
(testX, survival_yhat14) = utils.load_apical_slice(args["directory"], patient_info, 224, name='p_apical septum')
json_file = open('models/AHA14/aha14_VGG19.json','r')
model14_json = json_file.read()
json_file.close()
model14 = model_from_json(model14_json)
model14.load_weights("models/AHA14/aha14_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds14 = model14.predict(testX)
preds14 = [0 if preds14 < 0.5 else 1]

# AHA15
(testX, survival_yhat15) = utils.load_apical_slice(args["directory"], patient_info, 224, name='p_apical inferior')
json_file = open('models/AHA15/aha15_VGG19.json','r')
model15_json = json_file.read()
json_file.close()
model15 = model_from_json(model15_json)
model15.load_weights("models/AHA15/aha15_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds15 = model15.predict(testX)
preds15 = [0 if preds15 < 0.5 else 1]

# AHA16
(testX, survival_yhat16) = utils.load_apical_slice(args["directory"], patient_info, 224, name='p_apical lateral')
json_file = open('models/AHA16/aha16_VGG19.json','r')
model16_json = json_file.read()
json_file.close()
model16 = model_from_json(model16_json)
model16.load_weights("models/AHA16/aha16_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds16 = model16.predict(testX)
preds16 = [0 if preds16 < 0.5 else 1]

# Run test data
test_data = [preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, preds13, preds14, preds15, preds16]

fig, ax = plt.subplots(
    figsize=(12, 8), nrows=1, ncols=1, subplot_kw=dict(projection="polar")
)
norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
utils.bullseye_plot(ax, test_data, cmap="RdPu", norm=norm)
plt.axis("off")
plt.show()
