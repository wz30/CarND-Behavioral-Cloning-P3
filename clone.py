import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		lines.append(line)
print (len(lines))

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = 'data/IMG/' + filename
	image = cv2.imread(current_path)
	
	#print (len(images))
	images.append(image)
	#try:
	measurement = float(line[3])
	#except ValueError:
	#	print ('Line is corupt')
	#	break

	measurements.append(measurement)
print (len(images))
X_train = np.array(images)
print (len(X_train))
Y_train = np.array(measurements)
from keras.models import Sequential
from keras.layers import Flatten, Dense


model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2,shuffle=True, nb_epoch=7)

model.save('model.h5')
