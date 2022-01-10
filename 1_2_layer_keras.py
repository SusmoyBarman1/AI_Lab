import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

import numpy as np


def buildModel():

	input_layer = Input(shape=(1,))
	output_layer = Dense(1, activation='sigmoid')(input_layer)

	model = Model(input_layer, output_layer)

	#Compile
	model.compile(optimizer='rmsprop', loss='mse', metrics='accuracy')

	return model


def buildTrainData():
    x = np.arange(1, 80000)
    return x

def buildTestData():
    x = np.arange(80000, 90000)
    return x

def printData(arr):
    for num in arr:
        print(num)


def main():

    #create train data
    trainX = buildTrainData()
    printData(trainX)

    a = 5; b = 8
    trainY = a * trainX + b
    #printData(trainY)


    #create test data
    testX = buildTestData()
    #printData(testX)

    testY = a * testX + b
    #printData(testY) 


    model = buildModel()
    #model.summary()

if __name__ == "__main__":
    main()



'''
tf.keras.Input(
    shape=None, batch_size=None, name=None, dtype=None, sparse=None, tensor=None,
    ragged=None, type_spec=None, **kwargs
)

tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)

compile(
    optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs
)

fit(
    x=None, y=None, batch_size=None, epochs=1, verbose='auto',
    callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
    class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)

evaluate(
    x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
    callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    return_dict=False, **kwargs
)
'''
