import argparse
import math

from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", help="Input image", type=str, required=True)
parser.add_argument("-w", "--width", help="Desired width", type=int, required=True)
parser.add_argument("-o", "--output", help="Output image", type=str, required=True)

args = parser.parse_args()

# Load image
img = Image.open(args.image).convert('LA')
px = img.load()

while img.width > args.width:
    print(img.width, img.height)

    grad = {}
    costs = {}
    for x in range(img.width):
        for y in range(img.height):
            grad[x, y] = 0
            costs[x, y] = px[x, y][0]

    gy_mat = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    gx_mat = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    # Calculate gradient intensities (Edges are just filled with nearest neighbor)
    for x in range(1, img.width-1):
        for y in range(1, img.height-1):
            gy, gx = (0, 0)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    gx += px[x+i, y+i][0] * gx_mat[i][j]
                    gy += px[x+i, y+i][0] * gy_mat[i][j]
            grad[x, y] = math.sqrt(gx**2 + gy**2)

    for x in range(img.width):
        grad[x, 0] = grad[x, 1]
        grad[x, img.height-1] = grad[x, img.height-2]

    for y in range(img.height):
        grad[0, y] = grad[1, y]
        grad[img.width-1, y] = grad[img.width-2, y]

    # Calculate square costs
    for x in range(img.width):
        costs[x, 0] = (None, grad[x, 0])
    for x in range(1, img.width-1):
        for y in range(1, img.height):
            next = (None, None)
            for n in [-1, 0, 1]:
                nval = grad[x+n, y-1]
                if next[1] == None or nval < next[1]:
                    next = ((x+n, y-1), nval + grad[x, y])
            costs[x, y] = (next[0], next[1] + grad[x, y])

        for y in range(1, img.height):
            next = (None, None)
            x = 0
            for n in [0, 1]:
                nval = grad[x+n, y-1]
                if next[1] == None or nval < next[1]:
                    next = ((x+n, y-1), nval + grad[x, y])
            costs[x, y] = (next[0], next[1] + grad[x, y])
            next = (None, None)
            x = img.width-1
            for n in [-1, 0]:
                nval = grad[x+n, y-1]
                if next[1] == None or nval < next[1]:
                    next = ((x+n, y-1), nval + grad[x, y])
            costs[x, y] = (next[0], next[1] + grad[x, y])

    # Delete paths until appropriate
    no_path = set()
    removed = set()
    path = []
    least = None
    current = None
    next = None
    for x in range(img.width):
        [x_next, x_cost] = costs[x, img.height-1]
        if (x, img.height-1) not in no_path and (least == None or x_cost < least):
            current = (x, img.height-1)
            next = x_next
            least = x_cost
    path.append(current)

    while next != None:
        [next, _] = costs[current]
        if next in no_path:
            break
        else:
            path.append(next)
            current = next

    if len(path) != img.height + 1:
        for x in path:
            no_path.add(x)
    else:
        for x in path:
            removed.add(x)

    img2 = Image.new('LA', (img.width-1, img.height))
    px2 = img2.load()

    for y in range(img2.height):
        seen = False
        for x in range(img2.width):
            if (x, y) in removed:
                seen = True
            px2[x, y] = px[(x+1 if seen else x), y]

    img = img2
    px = img.load()

# Write out result
if args.output[-3:] == "jpg" or args.output[-3:] == "jpeg":
    img.convert('L').save(args.output)
else:
    img.save(args.output)
