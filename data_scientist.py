import numpy as np
import cv2
import glob
import pickle
from lesson_functions import *

################################################################################
def merge_dataset():
    """
    Merge the seperated dataset's path
    """

    cars = glob.glob('../Vehicle-Detection-Dataset/vehicles/GTI_Far/*.png')
    cars += glob.glob('../Vehicle-Detection-Dataset/vehicles/GTI_Left/*.png')
    cars += glob.glob('../Vehicle-Detection-Dataset/vehicles/GTI_MiddleClose/*.png')
    cars += glob.glob('../Vehicle-Detection-Dataset/vehicles/GTI_Right/*.png')
    cars += glob.glob('../Vehicle-Detection-Dataset/vehicles/KITTI_extracted/*.png')

    nocars = glob.glob('../Vehicle-Detection-Dataset/non-vehicles/Extras/*.png')
    nocars += glob.glob('../Vehicle-Detection-Dataset/non-vehicles/GTI/*.png')

    return cars, nocars
################################################################################

def save_train_test_split(pickle_file, X_train, X_valid, y_train, y_valid):

    train_test_split = {}
    train_test_split['X_train'] = X_train
    train_test_split['X_valid'] = X_valid
    train_test_split['y_train'] = y_train
    train_test_split['y_valid'] = y_valid

    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(train_test_split, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Save train_test_split {} failed: {}'.format(pickle_file, e))
        raise
    print('Save train_test_split into pickle')

################################################################################
