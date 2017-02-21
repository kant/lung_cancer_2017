SEED = 202

'''
 0  show all logs shown  (default)
 1  filter out INFO logs,
 2  additionally filter out WARNING logs, and
 3  additionally filter out ERROR logs.
'''
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#




import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')


import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)


# standard libs
import os
import pickle
from timeit import default_timer as timer
import csv
from datetime import datetime



# numerical libs
import matplotlib.pyplot as plt
import cv2
import math

import tensorflow as tf
tf.set_random_seed(SEED)

# my libs
from net.file import *
