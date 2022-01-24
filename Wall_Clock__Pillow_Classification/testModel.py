import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

# Global variable: Base Directory Path
DIR = './'
img_size = 256

def imgPrepare(object):
    imgDir = DIR + object
    saveDir = DIR + 'resized_' + object

    imgList = os.listdir(imgDir)
    n = len(imgList)
    imgDataset = []
    
    for i in range(n):
        imgPath = imgDir + imgList[i]
        
        if (os.path.exists(imgPath)):
            # Load Image
            img = cv2.imread(imgPath)
            #print(f"{imgPath}, img.shape: {img.shape}")

            # Resize Image
            resizedImg = cv2.resize(img, (img_size, img_size))  
            #print(f"resizedImg.shape: {resizedImg.shape}", end='\n\n')

            # BGR to RGB
            rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
            
            # Put the image to array
            imgDataset.append(rgbImg)

        else:
            print(f"{imgPath} not exists")
    
    # Convert the list to numpy array
    imgDataset = np.array(imgDataset, dtype=np.uint8)
    #print(f"imgDataset.shape: {imgDataset.shape}", end='\n\n')
    
    return imgDataset

def preprocess_test_data():

    # Load Car Image Data
    clock = 'test_clock/'
    clockData = imgPrepare(clock)
    n = clockData.shape[0]
    print(f"clockData.shape: {clockData.shape}")

    # Load Hen Image Data
    pillow = 'test_pillow/'
    pillowData = imgPrepare(pillow)
    m = pillowData.shape[0]
    print(f"pillowData.shape: {pillowData.shape}")

    # Concatenating carData and henData
    imgDB = np.concatenate((clockData, pillowData), axis=0)
    print(f"\nimgDB.shape: {imgDB.shape}")
    print(imgDB.max(), imgDB.min())
    
    # Prepare Data
    clockLabel = np.zeros(n, dtype=np.uint8) # clockLabel --> 0
    pillowLabel = np.ones(m, dtype=np.uint8) # pillowLable --> 1

    #print(clockLabel)
    #print(pillowLabel)

    # Concatenate label
    labelDB = np.concatenate((clockLabel, pillowLabel))
    print(f"\n{labelDB}\nlabelDB.shape: {labelDB.shape}")

    # To Categorical
    labelDB = to_categorical(labelDB)
    print(f"\nlabelDB TO_Categorical: \n{labelDB}")

    # Shuffle Data
    print("\n\nShuffle data\n")
    n = imgDB.shape[0]

    indeces = np.arange(n)
    #print(indeces)
    random.shuffle(indeces)
    #print(indeces)
    imgDB = imgDB[indeces]
    labelDB = labelDB[indeces]
    
    return imgDB, labelDB

def plot_predicted_data(test_imgDB, test_labelDB, model):
    plt.figure(figsize=(20, 20))

    predictedY = model.predict(test_imgDB)

    n=6
    for i in range(n):
        print(f'\n\nindex:-{i+1}  Predicted Value: {np.argmax(predictedY[i])}\n')

        plt.subplot(3, 3, i+1)
        plt.imshow(test_imgDB[i], cmap = 'gray')
        plt.title(np.argmax(test_labelDB[i]))
        plt.axis('off')
    
    plt.show()
    plt.close()



def modelSummary(model):
    print()
    model.summary()
    print()


def main(): 
    
    # Load pre-trained model
    model = load_model("clock_pillow_model.h5")
    modelSummary(model)

    # Plot the testing data
    test_imgDB, test_labelDB = preprocess_test_data()
    plot_predicted_data(test_imgDB, test_labelDB, model)

if __name__ == "__main__":
    main()

