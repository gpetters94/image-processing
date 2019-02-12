#/usr/bin/python3

import argparse
import math
import os.path
from PIL import Image
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input filename", required=True)
parser.add_argument("-o", "--output", help="output filename", required=True)
parser.add_argument("--ssd", help="variance for d", type=float, required=True)
parser.add_argument("--ssr", help="variance for r", type=float, required=True)
parser.add_argument("-w", "--window", help="window diameter (pixels)", type=int, default=3)
parser.add_argument("-d", "--debug", help="enable debug messages (not recommended)", action="store_true")

args = parser.parse_args()

if not os.path.isfile(args.input):
    print("Please provide a valid input file.")
    sys.exit(-1)

in_file = args.input
out_file = args.output

if args.ssd < 0:
    print("Please provide a valid d variance")
    sys.exit(-1)

var_d = args.ssd

if args.ssr < 0:
    print("Please provide a valid r variance")
    sys.exit(-1)

var_r = args.ssr

DEBUG=args.debug

def debug_print(str):
    if DEBUG:
        print(str)

if args.window != None and args.window % 2 == 0:
    print("Window size must be odd.")
    sys.exit(-1)

if args.window == None:
    window = 1
else:
    window = int((args.window - 1)/2)

def norm_squared(v1, v2):
    norm = 0

    for i in range(0, len(v1)):
        norm += (v1[i] - v2[i])**2

    return norm

def w(img, channel, i, j, k, l):
    #return math.exp((((i - k)**2 + (j - l)**2) / (-2 * var_d)) - (norm_squared(img[i, j], img[k, l]) / (2 * var_r)))
    return math.exp((((i - k)**2 + (j - l)**2) / (-2 * var_d)) - (norm_squared([i, j], [k, l]) / (2 * var_r)))

# Open file, read to Image
unpadded_img = Image.open(in_file)
unpadded_px = unpadded_img.load()

chan_count = len(unpadded_px[0, 0])

# Create output Image, padding with black
if chan_count == 3:
    padded_img = Image.new('RGB', (unpadded_img.width + 2*window, unpadded_img.height + 2*window), (0, 0, 0))
else:
    padded_img = Image.new('RGBA', (unpadded_img.width + 2*window, unpadded_img.height + 2*window), (0, 0, 0, 256))
padded_img.paste(unpadded_img, (window, window, unpadded_img.width+window, unpadded_img.height+window))
padded_px = padded_img.load()

# Loop over output pixels, applying the filter to each
for i in range(window, unpadded_img.width + window):
    for j in range(window, unpadded_img.height + window):
        debug_print("Pixel %d, %d" % (i, j))
        pixel = [0] * chan_count
        for c in range(0, chan_count):
            num = 0
            denom = 0
            for k in range(i - window, i + window):
                for l in range(j - window, j + window):
                    wsum = w(padded_px, c, i, j, k, l)
                    denom += wsum
                    num += (padded_px[k, l][c] * wsum)
            pixel[c] = (num / denom)
        debug_print(padded_px[i, j])
        unpadded_px[i-window, j-window] = tuple(list(map(int, pixel)))
        debug_print(padded_px[i, j])

# Write the new image to the output file
unpadded_img.save(out_file)
