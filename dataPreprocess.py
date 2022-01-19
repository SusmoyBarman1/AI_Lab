import os
import cv2
import numpy as np

DIR = '/home/cse/Desktop/codes/'
img_size = 256

def main():

    # Local Directory name
    car = 'car/'
    hen = 'hen/'
    resizedCar = 'resized_car/'
    resizedHen = 'resized_hen/'

    # Working with Car
    print("\n----------\nPreparing Car Dataset")
    imgPrepare(car)
    print("\n\n-----------------------\nChecking The resized Image")
    checkResizedImage(resizedCar)

    # Working with hen
    print("\n----------\nPrepare Hen Dataset")
    imgPrepare(hen)
    print("\n\n-----------------------\nChecking The resized Image")
    checkResizedImage(resizedHen)


def checkResizedImage(object):
    imgDir = DIR + object

    imgList = os.listdir(imgDir)
    n = len(imgList)

    print(f"\nPrinting resized {object}\n")
    for i in range(n):
        imgPath = imgDir + imgList[i]
        
        if (os.path.exists(imgPath)):
            # Load Image
            img = cv2.imread(imgPath)
            print(f"{imgPath}, img.shape: {img.shape}")

        else:
            print(f"{imgPath} not exists")
    print('\n\n')


'''
Data Preparation function
'''
def imgPrepare(object):
    imgDir = DIR + object
    saveDir = DIR + 'resized_' + object

    # Creating new directory
    if not os.path.isdir(saveDir):
        print('\nThe directory is not present. Creating a new one..\n')
        os.mkdir(saveDir)
    else:
        print('\nThe directory is present.\n')


    imgList = os.listdir(imgDir)
    n = len(imgList)
    imgDataset = []
    count = 1
    for i in range(n):
        imgPath = imgDir + imgList[i]
        
        if (os.path.exists(imgPath)):
            # Load Image
            img = cv2.imread(imgPath)
            print(f"{imgPath}, img.shape: {img.shape}")

            # Resize Image
            resizedImg = cv2.resize(img, (img_size, img_size))  
            #print(f"resizedImg.shape: {resizedImg.shape}", end='\n\n')

            # BGR to RGB
            rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
            
            # Put the image to array
            imgDataset.append(rgbImg)

            # Write Resized Image
            imgName = object[:-1] + str(count) + '.jpg'
            imgWritePath = saveDir + imgName
            #print(imgWritePath, imgList[i])
            cv2.imwrite(imgWritePath, resizedImg)

            count = count + 1

        else:
            print(f"{imgPath} not exists")
    
    # Convert the list to numpy array\
    imgDataset = np.array(imgDataset, dtype=np.uint8)
    print(f"imgDataset.shape: {imgDataset.shape}", end='\n\n')


if __name__ == "__main__":
    main()