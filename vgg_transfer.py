'''
Binary Classification. (2 classes)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from email.mime import base
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping

def main():
    model = buildModel()
    #modelSummary(model)

    imgSet = prepare_image_data()
    labelSet = prepare_label_data()
    trainImg, testImg, trainLabel, testLabel = split_data(imgSet, labelSet)

    model.compile(
        loss='mse',
        optimizer='adam'
    )
    
    model.fit(
        trainImg,
        trainLabel,
        batch_size=8,
        epochs=100,
        validation_split=0.05,
        callback = EarlyStopping(
            monitor = 'val_loss',
            patience=10
        )
    )

    prediction = model.predict(testImg[:10])


def split_data(imgSet, labelSet):
    trainImg = imgSet[:70]
    testImg = imgSet[70:]
    
    trainLabel = labelSet[:70]
    testLabel = labelSet[70:]

    return trainImg, testImg, trainLabel, testLabel


def prepare_label_data():
    # Prepare data for  2 classes
    # 0: Class 1
    # 1: Class2

    # Convert labels into one-hot vectors
    # return  one-hot vectors
    return

def prepare_image_data():
    # Load Image
    # resize image
    # Put images into a 4D numpy array (n, 256, 245, 3)
    # Preprocess the 4D numpy array according to VGG16
    # Return the 4D numpy array dataset
    
    return

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