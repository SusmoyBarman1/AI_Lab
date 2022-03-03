import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Activation
from tensorflow.keras.optimizers import Adam


def modelSummary(model):
    print()
    model.summary()
    print()

def buildModel(input_shape=(28, 28, 3)):
    '''
    Our Model
    '''
    x_input = Input(input_shape)

    L1 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='L1_conv')(x_input)
    L1 = Activation('relu', name='L1_activation')(L1)

    L2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='L2_conv')(L1)
    L2 = Activation('relu', name='L2_activation')(L2)

    L3 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='L3_conv')(L2)
    L3 = Activation('relu', name='L3_activation')(L3)

    L4 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='L4_conv')(L3)
    L4 = Activation('relu', name='L4_activation')(L4)

    fc = Flatten()(L4)
    fc = Dense(512, activation='relu')(fc)
    fc = Dense(128, activation='relu')(fc)
    fc = Dense(64, activation='relu')(fc)
    fc = Dense(16, activation='relu')(fc)
    outputs = Dense(2, activation='softmax')(fc)
    
    model = Model(x_input, outputs, name='Basic_Binary_model')
    
    #Compile Model
    learning_rate= 5e-5
    opt = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(trainX, trainY, testX, testY):
    model = buildModel()
    modelSummary(model)
    
    
    # Train model
    my_callbacks = [EarlyStopping(monitor='loss', patience=3)]

    print("\n\nFitting with training data\n\n")
    model.fit(trainX, trainY,
              epochs = 10,
              callbacks=my_callbacks)
    
    print("\n\nEvaluating with test data\n\n")
    model.evaluate(testX, testY)
    



def main(): 
    
    # Train the model
    train_model(trainX, trainY, testX, testY)
    #train_model()

if __name__ == "__main__":
    main()