import os
import cv2
import numpy as np

DIR = '/home/cse/Desktop/codes/'
img_size = 256

def main():

    # Local Directory name
    car = 'car/'
    hen = 'hen/'


    print("\n----------\nPreparing Car Dataset")
    imgPrepare(car)
    print("\n----------\nPrepare Hen Dataset")
    imgPrepare(hen)

def imgPrepare(car):
    imgDir = DIR + car
    imgList = os.listdir(imgDir)

    n = len(imgList)
    imgDataset = []
    for i in range(n):
        imgPath = imgDir + imgList[i]
        
        if (os.path.exists(imgPath)):
            # Load Image
            img = cv2.imread(imgPath)
            print(f"{imgPath}, img.shape: {img.shape}", end=' ')

            # Resize Image
            resizedImg = cv2.resize(img, (img_size, img_size))  
            print(f"resizedImg.shape: {resizedImg.shape}", end='\n\n')

            # BGR to RGB
            rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
            
            # Put the image to array
            imgDataset.append(rgbImg)

        else:
            print(f"{imgPath} not exists")
    
    # Convert the list to numpy array\
    imgDataset = np.array(imgDataset, dtype=np.uint8)
    print(f"imgDataset.shape: {imgDataset.shape}", end='\n\n')


if __name__ == "__main__":
    main()