import argparse
import pickle

from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--left", help="Left image", required=True)
parser.add_argument("-r", "--right", help="Right image", required=True)
parser.add_argument("--occ", help="Occlude penalty", type=float, default=20.0)
parser.add_argument("-o", "--output", help="Output file (for disparity map)", required=True)

args = parser.parse_args()

# Load images
l_img = Image.open(args.left).convert('L')
l_px = l_img.load()

r_img = Image.open(args.right).convert('L')
r_px = r_img.load()

# Dynamic programming
value = {}
for x in range(l_img.width):
    for y in range(l_img.width):
        value[x, y] = (None, 0)

# Make disparity map
out_img = Image.new('L', (l_img.width, l_img.height))
out_px = out_img.load()

for y in range(l_img.height):
#for y in range(150, 200):
    print("Row %d / %d" % (y, l_img.height-1))

    for i in range(l_img.width):
        for j in range(l_img.width):
            if (i, j) == (0, 0):
                pass
            elif i == 0:
                occ = args.occ + value[i, j-1][1]
                value[i, j] = ((i, j-1), occ)
            elif j == 0:
                occ = args.occ + value[i-1, j][1]
                value[i, j] = ((i-1, j), occ)
            else:
                l_val = float(l_px[i, y])
                r_val = float(r_px[j, y])
                #dist = abs(r_val - l_val)
                dist = abs(r_val - l_val)**2

                match = (dist + value[i-1, j-1][1])
                occ = args.occ + value[i, j-1][1]
                occ2 = args.occ + value[i-1, j][1]

                if match == min(match, occ, occ2):
                    value[i, j] = ((i-1, j-1), match)
                elif occ2 == min(match, occ, occ2):
                    value[i, j] = ((i-1, j), occ2)
                else:
                    value[i, j] = ((i, j-1), occ)

    prev = (l_img.width-1, r_img.width-1)
    next = value[prev][0]
    path = []
    while next != None:
        if next[0] < prev[0] and next[1] < prev[1]:
            path.append(max(0, next[0] - next[1]))
        elif next[0] < prev[0] and next[1] == prev[1]:
            path.append("occ")
        prev = next
        (next, _) = value[prev]

    x = 0
    while path != []:
        p = path.pop()
        if p != "occ":
            out_px[x, y] = p*4
            x += 1

# Output disparity map
out_img.save(args.output)
