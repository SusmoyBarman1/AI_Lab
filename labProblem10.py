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

    L1 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='L1_conv')(x_input)
    L1 = Activation('relu', name='L1_activation')(L1)

    L2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='L2_conv')(L1)
    L2 = Activation('relu', name='L2_activation')(L2)

    L3 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='L3_conv')(L2)
    L3 = Activation('relu', name='L3_activation')(L3)

    L4 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='L4_conv')(L3)
    L4 = Activation('relu', name='L4_activation')(L4)

    
    FC = Flatten()(L4)

    FC = Dense(10)(FC)

    output = Activation('softmax', name='FC_activation')(FC)

    model = Model(x_input, output)

    return model


def main():

    model = create_model()

    print('\n\n\n\n-----------------Model Summary----------------------\n\n')
    print(model.summary())


if __name__ == '__main__':
    main()   