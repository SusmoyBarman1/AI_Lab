import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Input, Flatten, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def main():

	(trainX, trainY), (testX, testY) = load_data()

	trainX, trainY, testX, testY = data_preparation(trainX, trainY, testX, testY)

	model = build_model()
	'''
	print()
	model.summary()
	'''

	
	model.fit(trainX, trainY, epochs = 2, validation_split = 0.2)
	model.evaluate(testX, testY)
	
	#!mkdir -p saved_model
	model.save('saved_model/my_model')

	predictedY = model.predict(testX[:5])
	print('\n\nPredicted Value')
	print(np.argmax(predictedY))


def build_model():

	inpt = Input(shape=(28, 28))
	flat = Flatten()(inpt)

	fc = Dense(16, activation='relu')(flat)
	fc = Dense(32, activation='relu')(fc)
	fc = Dense(16, activation='relu')(fc)
	output = Dense(10, activation='softmax')(fc)

	model = Model(inpt, output, name='mnist_NN')
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

	return model


def data_preparation(trainX, trainY, testX, testY):	

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


def plot_data(x, y):
	plt.figure(figsize=(20, 20))

	for i in range(9):
		plt.subplot(3, 3, i+1)
		plt.imshow(x[i], cmap = 'gray')
		plt.title(y[i])

	plt.show()
	plt.close()


if __name__ == "__main__":
    main()













'''
# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model')


new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()



'''