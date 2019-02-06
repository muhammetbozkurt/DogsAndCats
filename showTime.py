import os
import cv2
import tensorflow as tf
from tensorflow.keras import models
import numpy as np

width = 75
DATADIR = "C:/python/keras/PetImages"
CATEGORIES = ["Dog","Cat"]
model = models.load_model("dogCatDisq.model")
while(True):
	catDog = input("Dog-> 0\nCat->1\n")
	if(int(catDog)!=1 and int(catDog)!=0):
		break
	path = os.path.join(DATADIR,CATEGORIES[int(catDog)])
	num = input("num of picture\n")
	img = cv2.imread(os.path.join(path,num+".jpg"),cv2.IMREAD_GRAYSCALE)
	img1=img
	img = cv2.resize(img,(width,width))
	new_in = np.array(img).reshape(-1,width,width,1)
	print("-----------------------------------------------\n")
	prediction = model.predict(new_in)
	print("\n\n\n",prediction)
	print("-----------------------------------------------")
	if(round(prediction[0][0]) == 1):
		print('Cat\n\n\n')
	else:
		print("Dog\n\n\n")
	cv2.imshow("pencere",img1)
	cv2.waitKey(0) #herhangi bir basılana kadar bekliyor
	cv2.destroyAllWindows()

