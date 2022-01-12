import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def main():
	trainX, trainY, testX, testY = data_preparation()

	#train_model(trainX, trainY, testX, testY)

	model = load_model("mnist_model.h5")
	print()
	model.summary()
	print()

	n = 5
	predictedY = model.predict(testX[:n])

	plot_predicted_data(testX[:n], testY[:n], n, predictedY)

	#print(np.argmax(predictedY))
	#plot_single_data(testX[:1], testY[:1])

	

def plot_predicted_data(testX, testY, n, predictedY):

	plt.figure(figsize=(20, 20))

	for i in range(n):
		print(f'\n\nPredicted Value: {np.argmax(predictedY[i])}\n')

		plt.subplot(3, 3, i+1)
		plt.imshow(testX[i], cmap = 'gray')
		#plt.title(y[i])
		plt.title(np.argmax(testY[i]))

	plt.show()
	plt.close()

def train_model(trainX, trainY, testX, testY):

	model = build_model()
	
	# Train model
	my_callbacks = [EarlyStopping(monitor='val_loss', patience=10),
					History()]

	model.fit(trainX, trainY, epochs = 100, callbacks= my_callbacks, validation_split = 0.2)
	model.evaluate(testX, testY)

	predictedY = model.predict(testX[:5])

	for i in range(5):
		print(f'\n\nPredicted Value: {np.argmax(predictedY[i])}\n')
		#print(predictedY.shape)
	

	# Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
	model.save("mnist_model.h5")

	


def build_model():

	inpt = Input(shape=(28, 28))
	flat = Flatten()(inpt)

	fc = Dense(16, activation='relu')(flat)
	fc = Dense(32, activation='relu')(fc)
	fc = Dense(16, activation='relu')(fc)
	output = Dense(10, activation='softmax')(fc)

	model = Model(inpt, output, name='mnist_NN')

	opt = Adam(learning_rate=0.01)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

	return model


def data_preparation():	

	# Load data
	(trainX, trainY), (testX, testY) = load_data()

	# Type conversion
	trainX = trainX.astype("float32")
	testX = testX.astype("float32")

	# Range Conversion
	trainX /= 255
	testX /= 255
	
	# One-Hot Conversion
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)

	return trainX, trainY, testX, testY


def plot_single_data(x, y):
	plt.figure(figsize=(20, 20))

	plt.imshow(x[0], cmap = 'gray')
	plt.title(np.argmax(y[0]))

	plt.show()
	plt.close()


def plot_data(x, y, n):
	plt.figure(figsize=(20, 20))

	for i in range(n):
		plt.subplot(3, 3, i+1)
		plt.imshow(x[i], cmap = 'gray')
		#plt.title(y[i])
		plt.title(np.argmax(y[i]))

	plt.show()
	plt.close()


if __name__ == "__main__":
    main()
