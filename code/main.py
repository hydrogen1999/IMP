import csv
import sys
import argparse
import numpy as np
import scipy.io as scio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import io, filters, feature, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

import core as core
import visualize
from helpers import cheat_interest_points, evaluate_correspondence



# This script
# (1) Loads and resizes images
# (2) Finds interest points in those images                 (you code this)
# (3) Describes each interest point with a local feature    (you code this)
# (4) Finds matching features                               (you code this)
# (5) Visualizes the matches
# (6) Evaluates the matches based on ground truth correspondences

def load_data(file_name):

    image1_file = "../data/"+ file_name +"/1.jpg"
    image2_file = "../data/"+file_name+"/2.jpg"
    eval_file = "../data/"+file_name+"/eval.mat"

    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))

    return image1, image2, eval_file

def main():

    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pair", required=True)

    args = parser.parse_args()

    # (1) Load in the data
    image1_color, image2_color, eval_file = load_data(args.pair)

    image1 = rgb2gray(image1_color)
    # image1 = image1[:,:,0] * 0.2989 + image1[:,:,1] * 0.5870 + image1[:,:,2] * 0.1140
    image2 = rgb2gray(image2_color)
    # image2 = image2[:,:,0] * 0.2989 + image2[:,:,1] * 0.5870 + image2[:,:,2] * 0.1140
    scale_factor = 1

    # Bilinear rescaling
    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))

    # width and height of each local feature, in pixels
    feature_width = 16

    # (2) Find distinctive points in each image. See Szeliski 4.1.1
    
    print("Getting interest points...")
    (x1, y1) = core.get_interest_points(image1,feature_width)
    (x2, y2) = core.get_interest_points(image2,feature_width)


    # (x1, y1, x2, y2) = cheat_interest_points(eval_file, scale_factor)
    plt.imshow(image1, cmap="gray")
    plt.scatter(x1, y1, alpha=0.9, s=3)
    plt.show()

    plt.imshow(image2, cmap="gray")
    plt.scatter(x2, y2, alpha=0.9, s=3)
    plt.show()
    print("Done!")

    # 3) Create feature vectors at each interest point. Szeliski 4.1.2

    print("Getting features...")
    image1_features = core.get_features(image1, x1, y1, feature_width)
    image2_features = core.get_features(image2, x2, y2, feature_width)
    print("Done!")

    # 4) Match features. Szeliski 4.1.3

    print("Matching features...")
    matches, confidences = core.match_features(image1_features, image2_features)
    print("Done!")

    # 5) Evaluation and visualization

    print("Matches: " + str(matches.shape[0]))
    # num_pts_to_visualize = matches.shape[0]
    num_pts_to_visualize = 50

    evaluate_correspondence(image1_color, image2_color, eval_file, scale_factor,x1, y1, x2, y2, matches, confidences, num_pts_to_visualize, args.pair + '_matches.jpg')

    return

if __name__ == '__main__':
    main()
