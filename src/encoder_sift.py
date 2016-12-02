import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.externals import joblib

# create a feature vector concatenating each image
def generate_vector(model, img_path):
	img = cv2.imread(img_path)
	gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	descriptors = get_descriptors(img_path)

	feature_vec = [0]*len(model.cluster_centers_)
	for descriptor in descriptors:
		prediction = model.predict(descriptor.reshape(1, -1))

		label = prediction[0]
		feature_vec[label] += 1

	return np.array(feature_vec)

def get_descriptors(file_path):
	img = cv2.imread(file_path)
	gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray, None)

	kp,des = sift.compute(gray,kp)

	if des is None:
		return []

	return des

def extract_descriptors():
	test_folder = "../img/cifar-10/test"
	class_names = os.listdir(test_folder) # there are a folde for each class
	descriptors = []

	# processing train folder
	print "PROCESSING TEST FOLDER (FOR DESCRIPTORS): "
	count  = 0
	for name in class_names:
		files = os.listdir(test_folder+"/"+name)

		# transform each file into a feature vector
		for file_name in files:
			file_path = test_folder+"/"+name+"/"+file_name

			temp = get_descriptors(file_path)
			descriptors.extend(temp)

			count += 1

			if count % 1000 == 0:
				print count, " images processed"

	return descriptors

def encode():

	# # uncomment to calculate kmeans on the fly
	# # descriptors = extract_descriptors()
	# #
	# # print "RUNNING K-MEANS: "
	# # cluster = KMeans(n_clusters=100, random_state=0).fit(descriptors)
	#
	# # using cached kmeans
	# k = joblib.load('cache/kmeans_100.pkl')
	#
	# test_folder = "../img/cifar-10/test"
	# class_names = os.listdir(test_folder) # there are a folde for each class
	#
	# # processing train folder
	# print "PROCESSING TEST FOLDER: "
	# X = []
	# y = []
	# count  = 0
	# for name in class_names:
	# 	files = os.listdir(test_folder+"/"+name)
	#
	# 	# transform each file into a feature vector
	# 	for file_name in files:
	# 		vec = generate_vector(k, test_folder+"/"+name+"/"+file_name)
	# 		X.append(vec.tolist())
	#
	# 		y_vec = [0] * len(class_names) # <<<<<<<<<<<<<< HOT ENCODING REPRESENTATION <<<<<
	# 		y_vec[class_names.index(name)] = 1
	# 		y.append(y_vec)
	#
	# 		count += 1
	#
	# 		if count % 1000 == 0:
	# 			print count, " images processed"
	#
	#
	# # randomizing positions
	# np.random.seed(42)
	# np.random.shuffle(X)
	# np.random.seed(42)
	# np.random.shuffle(y)
	#
	# np.save("cache/X_sift_100_clusters_encoded_images", X)
	# np.save("cache/Y_sift_100_clusters_encoded_images", y)

	X = np.load("cache/X_sift_100_clusters_encoded_images.npy")
	y = np.load("cache/Y_sift_100_clusters_encoded_images.npy")

	# spliting the dataset in thee groups
	X_train = X[:8000]
	y_train = y[:8000]

	X_validation = X_test = X[8000: 9000]
	y_validation = y_test = y[8000: 9000]

	X_test = X[9000: ]
	y_test = y[9000: ]

	return X_train, y_train, X_validation, y_validation, X_test, y_test
