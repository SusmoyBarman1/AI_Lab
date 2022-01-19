'''
Binary Classification. (2 classes)
'''
from cProfile import label
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import random
import numpy as np
from dataPreprocess import imgPrepare

DIR = '/home/cse/Desktop/codes/'

def main():
    #model = buildModel()
    #modelSummary(model)

    preprocess_data()


def preprocess_data():

    # Load Car Image Data
    car = 'car/'
    carData = imgPrepare(car)
    n = carData.shape[0]
    print(f"carData.shape: {carData.shape}")

    # Load Hen Image Data
    hen = 'hen/'
    henData = imgPrepare(hen)
    m = henData.shape[0]
    print(f"henData.shape: {henData.shape}")

    # Concatenating carData and henData
    imgDB = np.concatenate((carData, henData), axis=0)
    print(f"\nimgDB.shape: {imgDB.shape}")
    print(imgDB.max(), imgDB.min())
    # Prepare Data
    carLabel = np.zeros(n, dtype=np.uint8)
    henLabel = np.ones(m, dtype=np.uint8)

    #print(carLabel)
    #print(henLabel)

    # Concatenate label
    labelDB = np.concatenate((carLabel, henLabel))
    print(f"\n{labelDB}\nlabelDB.shape: {labelDB.shape}")

    # To Categorical
    labelDB = to_categorical(labelDB)
    print(f"\nlabelDB TO_Categorical: \n{labelDB}")

    # Shuffle Data
    print("\n\nShuffle data\n")
    n = imgDB.shape[0]

    indeces = np.arange(n)
    print(indeces)
    random.shuffle(indeces)
    print(indeces)
    imgDB = imgDB[indeces]
    labelDB = labelDB[indeces]

    # split train test data
    m = int(n*0.7)
    trainX = imgDB[:m]
    testX = imgDB[m:]

    trainY = labelDB[:m]
    testY = labelDB[m:]

    print("\n\nShapes of train test data.\n")
    print(f"trainX.shape: {trainX.shape}\n")
    print(f"testX.shape: {testX.shape}\n")
    print(f"trainY.shape: {trainY.shape}\n")
    print(f"testY.shape: {testY.shape}\n")


    

def modelSummary(model):
    print()
    model.summary()
    print()

def buildModel():

    '''
    Original Model
    '''
    baseModel = VGG16(include_top = False, input_shape=(256, 256, 3))
    
    inputs = baseModel.input
    baseModelOutput = baseModel.output

    '''
    Our Model
    '''
    fc = Flatten()(baseModelOutput)
    fc = Dense(16, activation='relu')(fc)
    outputs = Dense(2, activation='softmax')(fc)
    
    '''
    Concatening Original model and our model
    '''
    new_model = Model(inputs, outputs)
    
    '''
    Making the main network non-trainable
    '''
    for layer in baseModel.layers:
        layer.trainable = False
    

    return new_model


if __name__ == "__main__":
    main()