import cv2
import numpy as np


def standardize(data):
	data[0] = 0
	mean = np.mean(data)
	std = np.std(data)
	return (data - mean)/std

# create a feature vector concatenating each image
def generate_vector(img_path):

    image = cv2.imread(img_path)
    winSize = (4, 4)
    blockSize = (4, 4)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 3
    derivAperture = 1
    winSigma = 2.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 3
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (4, 4)
    padding = (2, 2)
    locations = ((10, 20),)
    hist = hog.compute(image)

    print hist


if __name__ == "__main__":
    generate_vector("img/Lenna.jpg")