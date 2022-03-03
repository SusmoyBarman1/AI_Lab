import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense


'''
Creating main model
'''
def create_model(input_shape=(28, 28, 3)):
    x_input = Input(input_shape)

    layer1 = Conv2D(128, (3, 3), strides=(2, 2))(x_input)

    layer2 = Conv2D(64, (7, 7), strides=(2, 2))(layer1)

    layer3 = Conv2D(32, (1, 1), strides=(1, 1))(layer2)

    layer4 = Conv2D(16, (3, 3), strides=(1, 1))(layer3)

    
    FC = Flatten()(layer4)

    FC = Dense(10)(FC)

    output = Activation('softmax')(FC)

    model = Model(x_input, output)

    return model


def main():

    model = create_model()

    print('\n\n\n\n-----------------Model Summary----------------------\n\n')
    print(model.summary())


if __name__ == '__main__':
    main()   