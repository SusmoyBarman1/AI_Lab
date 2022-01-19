import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist, boston_housing

def main():

    mnist_data()
    fashion_mnist_data()

def mnist_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()  #fashion_mnist.load_data()    boston_housing.load_data()
    print()
    print(trainX.shape, testX.shape)
    showImg(trainX[:10])

def fashion_mnist_data():
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data() # mnist.load_data()  boston_housing.load_data()
    print()
    print(trainX.shape, testX.shape)
    showImg(trainX[:10])

def showImg(img):
    plt.figure(figsize=(20, 20))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(img[i], cmap='gray')

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()