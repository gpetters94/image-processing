import math

'''
Modified from HW2 - clamp gradient direction to the range [0, 180)
'''
def clamp(theta):
    # Convert to degrees
    theta = math.degrees(theta)

    # Convert to minimum angle
    while theta < 0:
        theta += 180
    while theta >= 180:
        theta -= 180

    return theta

'''
Normalizes a list as a vector
'''
def normalize(vec):
    l = 0

    for e in vec:
        l += e**2

    norm = math.sqrt(l)

    if norm == 0:
        norm = 0.000001

    return [(x/norm) for x in vec]

'''
Perform convolution of the given matrix on the given image
Expects the image to already be padded and the matrix size to be odd
'''
def convolve(mat, img):
    px = img.load()
    mat_width = int((len(mat)-1)/2)
    mat_height = int((len(mat[0])-1)/2)

    output = [[] for _ in range(mat_width, img.width-mat_width)]

    for i in range(mat_width, img.width-mat_width):
        for j in range(mat_height, img.height-mat_height):
            wsum = 0

            for x in range(-mat_width, mat_width+1):
                for y in range(-mat_height, mat_height+1):
                    wsum += (px[i+x, j+y][0] * mat[x+mat_width][y+mat_height])

            output[i-mat_width].append(wsum)

    return output

'''
hog(): take an image (PIL image, not pixels) and returns its HOG
'''
def hog(img):

    def angle_split(angle, magnitude):
        angles = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
        split = {}

        if angle in angles:
            split[int(angle)] = magnitude
        else:
            (small, big) = (0, 0)
            for i in range(1, 10):
                big = angles[i]
                if big > angle:
                    bprop = (big-angle)/20
                    lprop = (angle-small)/20

                    split[int(small)] = magnitude * lprop
                    split[int(big)] = magnitude * bprop

                    break
                else:
                    small = big

        if 180 in split:
            split[0] = split[180]
            del split[180]

        return split

    # Preprocessing
    c_x = int(img.height/2)
    c_y = int(img.width/2)
    img = img.convert('LA').crop((c_y-33,c_x-65, c_y+33, c_x+65))
    px = img.load()

    # Gradients
    gx = convolve([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], img)
    gy = convolve([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], img)

    g = [[] for _ in range(1, img.width-1)]
    e = [[] for _ in range(1, img.width-1)]

    for i in range(0, img.width-2):
        for j in range(0, img.height-2):
            g[i].append(math.sqrt((gx[i][j])**2 + (gy[i][j])**2))
            e[i].append(clamp(math.atan2(gy[i][j], gx[i][j])))

    # Remove padding
    img = img.crop((1, 1, img.width-1, img.height-1))
    px = img.load()

    # 8x8 Histograms
    hgrams = [[[0 for _ in range(0, 9)] for _ in range(0, 16)] for _ in range(0, 8)]

    for hcell in range(0, 8):
        for vcell in range(0, 16):

            hgram = hgrams[hcell][vcell]

            for i in range(hcell * 8, (hcell * 8) + 8):
                for j in range(vcell * 8, (vcell * 8) + 8):
                    split = angle_split(e[i][j], g[i][j])

                    for a in split:
                        hgram[int(a/20)] += split[a]

    # 16x16 Normalization
    features = [[] for _ in range(0, 105)]

    c = 0
    for i in range(0, 7):
        for j in range(0, 15):

            f = hgrams[i][j] + hgrams[i][j+1] + hgrams[i+1][j] + hgrams[i+1][j+1]

            features[c] = normalize(f)

            c += 1

    # Concatenate All and Return
    hog = []

    for f in features:
        hog += f

    return hog


if __name__ == "__main__":
    import argparse
    import pprint

    from PIL import Image

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="input image", required=True, type=str)
    parser.add_argument("-o", "--output", help="output file", required=False, type=str)

    args = parser.parse_args()

    whole_hog = hog(Image.open(args.input))

    if args.output == None:
        pprinter = pprint.PrettyPrinter()
        pprinter.pprint(whole_hog)
    else:
        with open(args.output, 'w') as f:
            f.write(str(whole_hog))
