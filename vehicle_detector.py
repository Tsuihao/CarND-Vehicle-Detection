import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from lesson_functions import *


################################################################################
def generate_heatmap(img, bbox_list):

    h, w = img.shape[0:2]
    heatmap = np.zeros((h, w)).astype(np.float32)
    if bbox_list:
        for box in bbox_list:
            x1, y1 = box[0]
            x2, y2 = box[1]

            heatmap[y1:y2, x1:x2] += 1

    return heatmap
################################################################################

def apply_threshold(heatmap, threshold):

    heatmap[heatmap < threshold] = 0;
    return heatmap
################################################################################
#
# Generates grid used for car detection for visualizarion.
# The region: (0, ystart) to (image_width, ystop).
# Params: img - source float32 RGB image with normed channels in range [0..1].
#         ystart - top Y of region
#         ystop - bottom Y of region
#         scale - scale of searching window.
#         cells_per_step - instead of overlap, define how many cells to step
# Returns: image.
#
def generate_grid(img, ystart, ystop, scale, cells_per_step=8):

    pix_per_cell   = 8

    roi_width = img.shape[1]
    roi_height = ystop - ystart;

    if scale != 1:
        roi_width = np.int(roi_width / scale)
        roi_height = np.int(roi_height / scale)

    # Define blocks and steps as above
    nxblocks = (roi_width // pix_per_cell)-1
    nyblocks = (roi_height // pix_per_cell)-1
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    bboxes = []
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            box = [(xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)]
            bboxes.append(box)

    return bboxes

################################################################################

#
# Define a single function that can extract features using hog sub-sampling and make predictions.
# The region: (0, ystart) to (image_width, ystop).
# Params: img - source float32 RGB image with normed channels in range [0..1].
#         ystart - top Y of region
#         ystop - bottom Y of region
#         scale - scale of searching window.
#         cells_per_step - instead of overlap, define how many cells to step.
# Returns: list of bouding boxes [(X1,Y1), (X2,Y2)].
#
def find_cars(img, ystart, ystop, scale, classifier, X_scaler, config, cells_per_step = 2):

    color_space    = config['color_space']
    spatial_size   = config['spatial_size']
    hist_bins      = config['hist_bins']
    orient         = config['orient']
    pix_per_cell   = config['pix_per_cell']
    cell_per_block = config['cell_per_block']
    hog_channel    = config['hog_channel']

    spatial_feat   = config['spatial_feat']
    hist_feat      = config['hist_feat']
    hog_feat       = config['hog_feat']

    assert(hog_channel == 'ALL')
    assert(spatial_feat == True)
    assert(hist_feat == True)
    assert(hog_feat == True)

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = classifier.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = [(xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)]
                bboxes.append(box)

    return bboxes

################################################################################
