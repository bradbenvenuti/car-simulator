import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
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

# Split data into train / validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=192):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		for offset in range(0, num_samples, 32):
			batch_samples = samples[offset:offset+32]

			images = []
			measurements = []

			# Load the images and measurements
			for batch_sample in batch_samples:
				steeringCorrection = 0.2
				steeringAngles = []
				steeringAngles.append(float(batch_sample[3]))
				steeringAngles.append(steeringAngles[0] + steeringCorrection)
				steeringAngles.append(steeringAngles[0] - steeringCorrection)
				for i in range(3):
					# get relative path for image so this works on all machines
					source_path = batch_sample[i]
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
			yield sklearn.utils.shuffle(x_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=192)
validation_generator = generator(validation_samples, batch_size=192)

# Create the model
model = Sequential()

# Normalize Data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Crop Data
model.add(Cropping2D(cropping=((50,20), (0,0))))

# Convolutions
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=9024, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')

exit()
