import csv
import cv2
import numpy as np

car_images = []
steering_angles = []
with open('./training_data_4/driving_log.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		steering_center = float(row[3])
		
		# create adjusted steering measurements for the side camera images
		correction = 0.25
		steering_left = steering_center + correction
		steering_right = steering_center - correction
		
		# read in images from center, left and right cameras
		directory = "./training_data/IMG"
		img_center = np.asarray(cv2.imread(row[0]))
		img_left = np.asarray(cv2.imread(row[1]))
		img_right = np.asarray(cv2.imread(row[2]))
		
		img_center_flipped = np.fliplr(np.asarray(cv2.imread(row[0])))
		img_left_flipped = np.fliplr(np.asarray(cv2.imread(row[1])))
		img_right_flipped = np.fliplr(np.asarray(cv2.imread(row[2])))
		
		# add images and angles to data set
		car_images.extend([img_center, img_left, img_right, img_center_flipped, img_left_flipped, img_right_flipped])
		steering_angles.extend([steering_center, steering_left, steering_right, -steering_center, -steering_left, -steering_right])


X_train = np.array(car_images)
y_train = np.array(steering_angles)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Cropping2D, Lambda, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ( (50, 20) , (0,0) )))
model.add(Convolution2D(3, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (3, 3), strides = (2, 2)))
model.add(Convolution2D(24, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (3, 3), strides = (2, 2)))
model.add(Convolution2D(36, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (3, 3), strides = (2, 2)))
model.add(Convolution2D(48, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (3, 3), strides = (2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1048))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

model.save('test_model.h5')