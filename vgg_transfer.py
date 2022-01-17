'''
Binary Classification. (2 classes)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from email.mime import base
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

def main():
    model = buildModel()
    model.summary()

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
    outputs = Dense(2, activation='relu')(fc)
    
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