import cv2
import numpy as np
import os

# standardize a list
def standardize(data):
	data[0] = 0
	mean = np.mean(data)
	std = np.std(data)
	return (data - mean)/std

# create a feature vector concatenating each image
def generate_vector(img_path):
	img = cv2.imread("./img/cifar-10/test/airplane/aeroplane_s_000002.png")
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray, None)

	# print len(kp)

	kp,des = sift.compute(gray,kp)

	# print des.sum(axis=0).size

	feature_vec = standardize(des.sum(axis=0))

	# imagem raw
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# feature_vec = standardize(img.flatten())

	# somando os canais
	# r = img[:,:,0].flatten()
	# g = img[:,:,1].flatten()
	# b = img[:,:,2].flatten()
	# feature_vec = standardize(r + g + b)

	# black and white
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# feature_vec = standardize(img.flatten())

	# print feature_vec
	# print feature_vec.size
	# exit()

	return feature_vec

def encode():

	test_folder = "./img/cifar-10/test"
	class_names = os.listdir(test_folder) # there are a folde for each class

	# processing train folder
	print "PROCESSING TEST FOLDER: "
	X = []
	y = []
	count  = 0
	for name in class_names:
		files = os.listdir(test_folder+"/"+name)

		# transform each file into a feature vector
		for file_name in files:
			vec = generate_vector(test_folder+"/"+name+"/"+file_name)
			X.append(vec.tolist())

			y_vec = [0] * len(class_names) # <<<<<<<<<<<<<< HOT ENCODING REPRESENTATION <<<<<
			y_vec[class_names.index(name)] = 1
			y.append(y_vec)

			count += 1

			if count % 1000 == 0:
				print count, " images processed"


	# randomizing positions
	np.random.seed(42)
	np.random.shuffle(X)
	np.random.seed(42)
	np.random.shuffle(y)


	# spliting the dataset in thee groups
	X_train = X[:8000]
	y_train = y[:8000]

	X_validation = X_test = X[8000: 9000]
	y_validation = y_test = y[8000: 9000]

	X_test = X[9000: ]
	y_test = y[9000: ]

	return X_train, y_train, X_validation, y_validation, X_test, y_test
