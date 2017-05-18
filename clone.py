import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read CSV file to get driving data
lines = []
with open('../drivingdata/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

# Load the images and measurements
for line in lines:
	steeringCorrection = 0.1
	steeringAngles = []
	steeringAngles[0] = float(line[3])
	steeringAngles[1] = steeringAngles[0] + steeringCorrection
	steeringAngles[2] = steeringAngles[0] - steeringCorrection
	for i in range(3):
		# get relative path for image so this works on all machines
		source_path = line[i]
		filename = source_path.split('/')[-1]
		local_path = '../drivingdata/IMG/' + filename
		image = cv2.imread(local_path)
		images.append(image)
		# create flipped version
		images.append(np.fliplr(image))
		# load the steering measurements
		measurement = steeringAngles[i]
		measurements.append(measurement)
		# create flipped measurement
		measurements.append(-measurement)

# Convert to numpy arrays
x_train = np.array(images)
y_train = np.array(measurements)

# Create the model
model = Sequential()

# Normalize Data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

exit()
