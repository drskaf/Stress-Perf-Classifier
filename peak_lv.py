import numpy as np
import pydicom
import os
import re
import utils
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

for root, dirs, files in os.walk(args["directory"], topdown=True):
