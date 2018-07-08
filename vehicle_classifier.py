import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2

################################################################################
def save_trained_model(pickle_file, classifier, Xscaler, config):
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {   'classifier':classifier,
                    'scaler':Xscaler,

                    'color_space': config['color_space'],
                    'orient': config['orient'],
                    'pix_per_cell': config['pix_per_cell'],
                    'cell_per_block': config['cell_per_block'],
                    'hog_channel': config['hog_channel'],
                    'spatial_size': config['spatial_size'],
                    'hist_bins': config['hist_bins'],
                    'spatial_feat': config['spatial_feat'],
                    'hist_feat': config['hist_feat'],
                    'hog_feat': config['hog_feat']
                },
                pfile, pickle.HIGHEST_PROTOCOL)
        print('Classifier saved to {}'.format(pickle_file))
    except Exception as e:
        print('Failed to save classifier to {} : {}'.format(pickle_file, e))
        raise

################################################################################

def load_trained_model(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            classifier = pickle.load(f)
        return classifier 
        print('Trained model {} is loaded !'.format(pickle_file))

    except Exception as e:
        print('Failed to load {}: {}!'.format(pickle_file, e))
        raise

################################################################################
