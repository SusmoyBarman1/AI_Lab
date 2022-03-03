import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Input, Conv2D
from tensorflow.keras.optimizers import Adam


def vgg16(imgSize = (224, 224, 3)):

    x_input = Input(imgSize)
    
    # conv 64, 2 times
    layer64 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape = imgSize)(x_input)
    layer64 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(layer64)  
    layer64 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer64)

    # conv 128, 2 times
    layer128 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(layer64)
    layer128 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(layer128)
    layer128 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer128)
    
    # conv 256, 3 times
    layer256 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(layer128)
    layer256 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(layer256)
    layer256 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(layer256)
    layer256 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer256)

    # conv 512, 3 timesmodelSummary
    layer512 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(layer256)
    layer512 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(layer512)
    layer512 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(layer512)
    
    layer512 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer512)

    # conv 512, 3 times
    layer512 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(layer512)
    layer512 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(layer512)
    layer512 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(layer512)
    
    layer512 = MaxPooling2D(pool_size=(2, 2), strides=2)(layer512)

    # Flatten
    fc = Flatten()(layer512)

    #FC layer
    fc = Dense(4096, activation='relu')(fc)
    fc = Dense(4096, activation='relu')(fc)

    # output layer
    output = Dense(1000, activation='softmax')(fc)

    vgg = Model(x_input, output)


    #Compile Model
    learning_rate= 5e-5
    opt = Adam(learning_rate=learning_rate)
    
    vgg.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return vgg


def train_model(trainX, trainY, testX, testY):
#def train_model():
    model = vgg16()
    model.summary()
    
    
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
