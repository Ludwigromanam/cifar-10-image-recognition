import cv2
import numpy as np
import os

# retorna uma image com apenas um dos canais
def get_img_channel(img, channel):

	img_copy = np.copy(img)

	if channel == "r":
		img_copy[:,:,1] = 0
		img_copy[:,:,2] = 0

	elif channel == "g":
		img_copy[:,:,0] = 0
		img_copy[:,:,2] = 0

	elif channel == "b":
		img_copy[:,:,0] = 0
		img_copy[:,:,1] = 0

	return img_copy


# generate histogram as a np.array
def hist(img):

	R = get_img_channel(img, "r").flatten()
	G = get_img_channel(img, "g").flatten()
	B = get_img_channel(img, "b").flatten()

	hist_R,_ = np.histogram(R, bins=256)
	hist_G,_ = np.histogram(G, bins=256)
	hist_B,_ = np.histogram(B, bins=256)

	return hist_R, hist_G, hist_B

# standardize a list
def standardize(data):
	data[0] = 0
	mean = np.mean(data)
	std = np.std(data)
	return (data - mean)/std

# create a feature vector concatenating each image
def generate_vector(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	hist_R, hist_G, hist_B = hist(img)
	feature_vec = np.hstack( [standardize(hist_R), standardize(hist_G), standardize(hist_B)])

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
