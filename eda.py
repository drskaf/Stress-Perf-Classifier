import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define indices
dirs_1 = os.listdir('/Users/ebrahamalskaf/Documents/manual_images')
dirs_2 = os.listdir('/Users/ebrahamalskaf/Documents/test')
dirs = dirs_2 + dirs_1
for dir in dirs:
    if '.DS_Store' in dirs:
        dirs.remove('.DS_Store')
if '.\\20220511_indices.csv' in dirs:
    dirs.remove('.\\20220511_indices.csv')

# Create dataframe
df = pd.DataFrame(dirs, columns=['index'])
df['ID'] = df['index'].astype(int)
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
info_df = pd.merge(df, patient_df, on=['ID'])
