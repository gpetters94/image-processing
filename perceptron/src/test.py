import argparse
import os

import hog
import perceptron

from PIL import Image
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", help="test image", required=True)
parser.add_argument("-p", "--pedestrian", help="a pedestrian is present - add this tag if the image contains a pedestrian and omit the tag if it does not", action="store_true")
parser.add_argument("-w", "--weights", help="input file for weights (pickled)", required=True)

args = parser.parse_args()

# Load image, get HOG
whole_hog = hog.hog(Image.open(args.image))
weights = None

# Open weights
with open(args.weights, 'rb') as f:
    weights = pickle.load(f)

# Use perceptron
result = perceptron.test(whole_hog, weights)

if result == args.pedestrian:
    print("Success")
else:
    print("Failure")

exit(0 if result == args.pedestrian else 1)
