#import the package
import cv2
import csv
import numpy as np

def getLines(datapath, skipHeader=False):
	#reading lines from driving.csv
	lines = []
	with open(datapath+'/driving_log.csv') as csv_file:
		reader = csv.reader(csv_file)
		if skipHeader:
			next(reader, None)
		for line in reader:
			lines.append(line)
	return lines
def extractimage(datapath, imagepath, measurement, images, measurements):
	originImage = cv2.imread(datapath+'/'+imagepath.strip())
	image = cv2.cvtColor(originImage, cv2.COLOR_BGR2RGB)
	images.append(image)
	measurements.append(measurement)
	#flip image
	images.append(cv2.flip(image, 1))
	measurements.append(measurement*-1.0)

def extractimages(datapath, skipHeader=False, correction=0.2):
	#loading image from driving logs
	lines = getLines(datapath,skipHeader)
	images = []
	measurements = []
	for line in lines:
		#line[3] means the fourth element
		measurement = float(line[3])
		#center
		extractimage(datapath, line[0], measurement, images, measurements)
		#left
		extractimage(datapath, line[1], measurement+correction, images, measurements)
		#right
		extractimage(datapath, line[2], measurement-correction, images, measurements)
	return (np.array(images), np.array(measurements))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def nVidiaModel():
	#using model from Nvidia model
	model = Sequential()
	model.add(Lambda(lambda x: (x/255.0)-0.5,input_shape=(160,320, 3)))
	model.add(Cropping2D(cropping=((50,20),(0,0))))
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

def train(model, input, output, modelFile, epochs = 10):
	#train model
	model.compile(loss='mse', optimizer = 'adam')
	model.fit(input, output, validation_split=0.2, shuffle=True,nb_epoch=epochs)
	model.save(modelFile)
	print("Saving model at"+modelFile)

print('Extracting images')
#skip header because of null
x, y = extractimages('../data', skipHeader=True)
#Use model
model = nVidiaModel()
print('Training model')
#Train  model
train(model, x, y, 'newModel.h5')
print('Ending')
