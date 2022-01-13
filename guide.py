'''
Command List
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
If do not work here then
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
    
python3 --version
mkdir Tensorflow
cd Tensorflow
python3 -m venv --system-site-packages ./
source ./bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow==2.6
Install Keras: pip install keras==2.6.0
pip install opencv-contrib-python matplotlib scipy scikit-learn
deactivate




Again Activate tensorflow using “source ./bin/activate” command  inside Tensorflow Folder,
type “python” to type code

To run code:
First activate tensorflow: source Tensorflow/bin/activate
Change directory to your working directory
Run code with command: python filename.py


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import numpy as np

def main():
    model = build_model()

def build_model():
    inputs = Input(1)
    outputs = Dense(1)(inputs)
    model = Model(inputs, outputs)
    model.compile(loss = 'mse')
    model.summary()

    return model


main()


1st Day Running Code:

# This is a sample Python script.
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Activation, Dense  # windows


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def main():
   model = build_model()
   model.compile(loss='mse', metrics='accuracy')


def build_model():
   # create layers
   inputs = Input(2, )
   x = Dense(2, activation='elu')(inputs)
   x = Dense(2, activation='relu')(x)
   x = Dense(2, activation='sigmoid')(x)
   outputs = Dense(1)(x)
   # create model
   model = Model(inputs, outputs)
   model.summary()

   return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/




Day 3: Digit Recognition

from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, History

def main():
    trainX, trainY, testX, testY = prepare_data()
    model = build_model()
    callbackList = [EarlyStopping(monitor = 'val_loss', patience = 10), History()]
    model.fit(trainX,trainY, epochs =100, callbacks = callbackList, validation_split=0.2)
    model.compile(metrics = 'accuracy')
    model.evaluate(testX, testY)
    #model save
    modelpath='/home/cse/tens---flow/digitModel.hdf5'
    model.save(modelpath)

def build_model():
    inputs = Input((28,28))
    x = Flatten()(inputs)
    x = Dense(16, activation='sigmoid')(x)
    x = Dense(8, activation='sigmoid')(x)
    outputs = Dense(10, activation = 'sigmoid')(x)
    #create model
    model = Model(inputs, outputs)
    model.summary()
    model.compile(loss='mse', optimizer ='rmsprop')

    return model

def prepare_data():
    (trainX,trainY), (testX,testY) = load_data()
    print(trainX.shape, trainY.shape,testX.shape,testY.shape)
    # plot_data(trainX[:9],trainY[:9])
    
    print(trainX.dtype,trainX.max(),trainX.min())
    trainX = trainX.astype(np.float32)
    testX = testX.astype(np.float32)
    trainX /= 255
    testX /=255
    print(trainX.dtype,trainX.max(),trainX.min())

    #convert numeric values 0,1,2....,9 into one hot vector
    #0: 1 0 0 0 0 0 0 0 0
    #1: 0 1 0 0 0 0 0 0 0
    print(trainY[:5])
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print(trainY[:5])
    return trainX, trainY, testX, testY

def plot_data(x,y):
    plt.figure(figsize = (20,20))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(x[i], cmap = 'gray')
        plt.title(y[i])
    plt.show()
    plt.close()
if __name__ == "__main__":
    main()


Image Classification:
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
# from tensorflow.keras.preprocessing import load_image
import cv2
import matplotlib.pyplot as plt
import numpy as np
def main():
    model = VGG16()
    # model.summary()
    imgPath = '/home/cse/tens---flow/elephant.jpeg'
    # img=load_img(imgPath)
    bgrImg =cv2.imread(imgPath)
    print(bgrImg.shape)
    rgbImg=cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    print(rgbImg.shape)
    rgbImg=cv2.resize(rgbImg,(224,224))
    display_img(rgbImg)
    rgbImg = np.expand_dims(rgbImg,axis=0)
    print(rgbImg.shape)
    
    # display_img(rgbImg)
    # print(rgbImg.shape)
    prediction = model.predict(rgbImg)
    prediction = decode_predictions(prediction)
    print(prediction)

def display_img(img):
    plt.imshow(img)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()











https://drive.google.com/drive/folders/1kFWgA9sxa7lZDr1fhZSTh9zopzy_Ky4k?usp=sharing

v
