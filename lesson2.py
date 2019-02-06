import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
from random import shuffle

width = 75
training_data=[]
def training_data_creator():
    DATADIR = "C:/python/keras/PetImages"
    CATAGORIES = ["Dog","Cat"]
    for catagory in CATAGORIES:
        path = os.path.join(DATADIR,catagory)
        class_num = CATAGORIES.index(catagory)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                training_data.append([cv2.resize(img_array,(width,width)), class_num])
            except Exception as e:
                pass


training_data_creator()

shuffle(training_data)

X = []
y = []
for feature, label in training_data:
    X.append(feature)
    y.append(label)
X = np.array(X).reshape(-1, width,width,1) # learn

print(X[0])

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()