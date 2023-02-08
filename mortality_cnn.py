import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tf_cnns
import utils
import argparse

INPUT_DIM = [256, 256]
BATCH_SIZE = 32
NUM_EPOCHS = 100
STEP_PER_EPOCH = 50
NO_VALID_STEPS = 20
N_CLASSES = 2
CHECKPOINT_PATH = os.path.join("model_weights", "cp-{epoch:02d}")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())
