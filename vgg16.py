import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.applications.vgg16 import VGG16


def main():

	model = VGG16(include_top=False)
	modelSummary(model)

def modelSummary(model):
	print()
	model.summary()
	print()


if __name__ == "__main__":
	main()