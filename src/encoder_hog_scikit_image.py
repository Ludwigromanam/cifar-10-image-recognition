import os
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np


def generate_vector(img_path):
    """Transforma a imagem em vetor, usando o HoG"""

    image = data.imread(img_path)
    image = color.rgb2gray(image)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualise=True)


    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))


    feature_vec = hog_image.flatten()
    return feature_vec


def encode():

    # test_folder = "../img/cifar-10/test"
    # class_names = os.listdir(test_folder) # there are a folde for each class
    #
    # # processing train folder
    # print "PROCESSING TEST FOLDER: "
    # X = []
    # y = []
    # count = 0
    # for name in class_names:
    #     files = os.listdir(test_folder+"/"+name)
    #
    #     # transform each file into a feature vector
    #     for file_name in files:
    #         vec = generate_vector(test_folder+"/"+name+"/"+file_name)
    #         X.append(vec.tolist())
    #
    #         y_vec = [0] * len(class_names) # <<<<<<<<<<<<<< HOT ENCODING REPRESENTATION <<<<<
    #         y_vec[class_names.index(name)] = 1
    #         y.append(y_vec)
    #
    #         count += 1
    #
    #         if count % 1000 == 0:
    #             print count, " images processed"
    #
    #
    # # randomizing positions
    # np.random.seed(42)
    # np.random.shuffle(X)
    # np.random.seed(42)
    # np.random.shuffle(y)
    #
    # np.save("cache/X_hog_encoded_images", X)
    # np.save("cache/Y_hog_encoded_images", y)

    X = np.load("cache/X_hog_encoded_images.npy")
    y = np.load("cache/Y_hog_encoded_images.npy")

    # spliting the dataset in thee groups
    X_train = X[:8000]
    y_train = y[:8000]

    X_validation = X_test = X[8000: 9000]
    y_validation = y_test = y[8000: 9000]

    X_test = X[9000: ]
    y_test = y[9000: ]

    return X_train, y_train, X_validation, y_validation, X_test, y_test
