import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt


def main():
    model = buildModel()
    model.summary()
    

def buildModel():

    inpt = Input(shape=(50, 50))

    flat = Flatten()(inpt)

    fc = Dense(32, activation='relu')(flat)
    fc = Dense(16, activation='relu')(fc)
    
    output = Dense(10, activation='softmax')(fc)

    model = Model(inpt, output)

    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mse', metrics='accuracy')

    return model


if __name__ == "__main__":
    main()