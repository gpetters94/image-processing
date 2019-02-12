import argparse
import math
import os

import hog
import perceptron

from PIL import Image
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--positive_dir", help="directory containing positive examples", required=True)
parser.add_argument("-n", "--negative_dir", help="directory containing negative examples", required=True)
parser.add_argument("-r", "--learning_rate", help="perceptron learning rate", required=True, type=float)
parser.add_argument("-e", "--error", help="acceptable error threshold", required=True, type=float)
parser.add_argument("-i", "--max_iterations", help="maximum iterations - stop after training for this long", required=False, type=int)
parser.add_argument("-o", "--output", help="output file for weights (pickled) - if this already exists then continue training on the weights in this file")
parser.add_argument("-d", "--debug", help="enable debug output", action="store_true")

args = parser.parse_args()

def debug_print(a):
    if args.debug:
        print(a)

data = []

# Read files
fcount = len(os.listdir(args.positive_dir)) + len(os.listdir(args.negative_dir))

try:
    with open('tmpfile', 'rb') as f:
        data = pickle.load(f)
        debug_print("Restarting interrupted training (remove tmpfile to start from stratch)")
except:
    debug_print("Reading data (%d files)" % fcount)

    n = 0
    pc = 0
    for f in os.scandir(args.positive_dir):
        data.append((hog.hog(Image.open(f.path)), 1))
        n += 1
        if n >= fcount / 10:
            n = 0
            pc += 10
            debug_print("%d%% read" % pc)

    for f in os.scandir(args.negative_dir):
        data.append((hog.hog(Image.open(f.path)), -1))
        n += 1
        if n >= fcount / 10:
            n = 0
            pc += 10
            debug_print("%d%% read" % pc)

debug_print("Done reading data")

# Save data to temporary file (in case of unexpected close)
with open('tmpfile', 'wb+') as f:
    pickle.dump(data, f)

debug_print("Training")

w = None
try:
    with open(args.output, 'rb') as f:
        weights = pickle.load(f)
        debug_print("Continuing training on weights (remove the file '%s' to start from stratch)" % str(args.output))
except:
    w = None

if args.max_iterations == None:
    weights = perceptron.train(data, args.learning_rate, args.error, -1, args.debug, w)
else:
    weights = perceptron.train(data, args.learning_rate, args.error, args.max_iterations, args.debug, w)

if args.output != None:
    with open(args.output, 'wb') as f:
        pickle.dump(weights, f)
    norm_squared = perceptron.dot(weights, weights)
    debug_print("Weights vector norm squared: %d (if this is 0 then something is wrong)" % norm_squared)
else:
    import pprint

    pp = pprint.PrettyPrinter()

    pp.pprint(weights)

#os.remove('tmpfile')
