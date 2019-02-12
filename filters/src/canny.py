#/usr/bin/python3

import argparse
import math
import os.path
from PIL import Image
import sys

import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input filename", required=True)
parser.add_argument("-o", "--output", help="output filename", required=True)
parser.add_argument("-w", "--window", help="window diameter for gaussian smoothing (pixels)", type=int, default=3)
parser.add_argument("--sigma", help="variance for gaussian smoothing (sigma^2)", type=float, required=True)
parser.add_argument("-k", help="center of gaussian smoothing", type=float, default=0)
parser.add_argument("-d", "--debug", help="enable debug messages (not recommended)", action="store_true")

args = parser.parse_args()

if not os.path.isfile(args.input):
    print("Please provide a valid input file.")
    sys.exit(-1)

in_file = args.input
out_file = args.output

if args.window % 2 == 0:
    print("Window size must be odd.")
    sys.exit(-1)

window = int((args.window-1)/2)
k = args.k

if args.sigma < 0:
    print("Please provide a valid variance")
    sys.exit(-1)

variance = args.sigma

def debug_print(str):
    if args.debug:
        print(str)

pp = pprint.PrettyPrinter()
def debug_pprint(str):
    if args.debug:
        pp.pprint(str)

def gaussian(x, y):
    global variance
    global k
    return (1 / (2 * math.pi * variance)) * math.exp(((x - k)**2 + (y - k)**2) / (-2 * variance))

def clamp(theta):
    # Convert to degrees
    theta = math.degrees(theta)

    # Convert to minimum angle
    while theta < 0:
        theta += 180
    while theta >= 180:
        theta -= 180

    # Round
    if theta >= 0 and theta < 22.5:
        theta = 0
    elif theta >= 22.5 and theta < 67.5:
        theta = 45
    elif theta >= 67.5 and theta < 112.5:
        theta = 90
    else:
        theta = 135

    return theta

# Load image as grayscale
unpadded_img = Image.open(in_file).convert('LA')
unpadded_px = unpadded_img.load()

chan_count = len(unpadded_px[0, 0])

# Open padded image and pad it
if chan_count == 3:
    padded_img = Image.new('RGB', (unpadded_img.width + 2*window, unpadded_img.height + 2*window), (0, 0, 0))
else:
    padded_img = Image.new('RGBA', (unpadded_img.width + 2*window, unpadded_img.height + 2*window), (0, 0, 0, 256))
padded_img.paste(unpadded_img, (window, window, unpadded_img.width+window, unpadded_img.height+window))
padded_px = padded_img.load()

# Create Gaussian matrix
matrix = []
total = 0

for r in range(-window, window+1):
    row = []
    for c in range(-window, window+1):
        rc = gaussian(r, c)
        row.append(rc)
        total += rc
    matrix.append(row)

debug_pprint(matrix)
debug_print(total)

# Normalize
for r in range(0, len(matrix)):
    for c in range(0, len(matrix[r])):
        matrix[r][c] /= total

debug_pprint(matrix)

# Perform the Gaussian blur
for x in range(window, unpadded_img.width + window):
    for y in range(window, unpadded_img.height + window):

        if(chan_count == 1):
            new_px = []
        else:
            new_px = [0] + ([256] * (chan_count - 1))

        for gx in range(-window, window+1):
            for gy in range(-window, window+1):
                for i in range(0, chan_count-1):
                    new_px[i] += int((matrix[gx][gy]) * (padded_px[(x - gx), (y - gy)])[i])

        unpadded_px[x-window, y-window] = tuple(new_px)

# Open another padded image and pad it
if chan_count == 1:
    padded_img = Image.new('L', (unpadded_img.width + 2, unpadded_img.height + 2), (0,))
else:
    padded_img = Image.new('LA', (unpadded_img.width + 2, unpadded_img.height + 2), (0, 256))
padded_img.paste(unpadded_img, (1, 1, unpadded_img.width+1, unpadded_img.height+1))
padded_px = padded_img.load()

# Find gradient
es = []
eo = []

mx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
my = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

for i in range(1, unpadded_img.width+1):
    esr = []
    eor = []
    for j in range(1, unpadded_img.height+1):
        gx = 0
        gy = 0

        for x in range(-1, 1):
            for y in range(-1, 1):
                gx += (mx[x][y] * padded_px[i+x, j+y][0])
                gy += (my[x][y] * padded_px[i+x, j+y][0])

        esr.append(math.sqrt(gx**2 + gy**2))
        eor.append(clamp(math.atan2(gx, gy)))
    es.append(esr)
    eo.append(eor)

# Add pad values for edges
for row in es:
    row.append(0)
    row.insert(0, 0)
es.append([0] * len(es))
es.insert(0, [0] * len(es[0]))

for row in eo:
    row.append(0)
    row.insert(0, 0)
eo.append([0] * len(eo))
eo.insert(0, [0] * len(eo[0]))

# Edge suppression
for i in range(1, unpadded_img.width):
    for j in range(1, unpadded_img.height):
        # Neighbor strengths
        n1 = 0
        n2 = 0

        # This pixel's strength
        ts = es[i][j]

        if ts == 0:
            n1 = es[i-1][j]
            n2 = es[i+1][j]
        elif ts == 45:
            n1 = es[i+1][j-1]
            n2 = es[i-1][j+1]
        elif ts == 90:
            n1 = es[i][j-1]
            n2 = es[i][j+1]
        else:
            n1 = es[i-1][j-1]
            n2 = es[i+1][j+1]

        debug_print("Edge: (%d, %d, %d)" % (n1, ts, n2))

        if ts < n1 or ts < n2:
            unpadded_px[i-1, j-1] = (0,)
        else:
            unpadded_px[i-1, j-1] = (int(ts),)

# Write new image out
unpadded_img.save(out_file)
