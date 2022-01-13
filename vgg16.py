import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
	# Load a pre-trained model.
	model = VGG16()
	model.summary()
	
	# Load image
	imgPath = '/home/cse/Desktop/codes/hen.jpeg'	
	bgrImg = cv2.imread(imgPath)
	print(bgrImg.shape)
	
	# Convert image from BGR into RGB format
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	
	# Reshape image so that it can fit into the model.
	''' display_img(rgbImg) '''
	rgbImg = cv2.resize(rgbImg, (224, 224))
	''' display_img(rgbImg) '''
	
	# Expand dimension since model accepts 4D data.
	print(rgbImg.shape)
	rgbImg = np.expand_dims(rgbImg, axis = 0)
	print(rgbImg.shape)
	
	# Preprocess image
	rgbImg = preprocess_input(rgbImg)
	
	# Predict which class the loaded image belongs to
	prediction = model.predict(rgbImg)
	prediction = decode_predictions(prediction)
	print(prediction)
	
	
def display_img(img):
	plt.imshow(img)
	plt.show()
	plt.close()

if __name__ == '__main__':
	main()