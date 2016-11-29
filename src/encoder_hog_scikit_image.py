import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np

def standardize(data):
	data[0] = 0
	mean = np.mean(data)
	std = np.std(data)
	return (data - mean)/std

def generate_vector(img_path):

    # image = color.rgb2gray(data.astronaut())
    # image = data.imread('../img/Lenna.png')
    image = data.imread(img_path)
    image = color.rgb2gray(image)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualise=True)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    #
    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    # ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    # ax2.axis('off')
    # ax2.imshow(hog_image, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # ax1.set_adjustable('box-forced')
    # plt.show()
    print len(hog_image_rescaled)
    print len(hog_image_rescaled[0])

    return  hog_image_rescaled

if __name__ == "__main__":
    generate_vector('../img/cifar-10/test/cat/alley_cat_s_000013.png')