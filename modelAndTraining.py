import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255 #for scaling

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))#ters yaptım bakalım
model.add(Activation("relu"))
model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(5,5)))#44
model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(Conv2D(64,(3,3)))#40
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4,4)))#10

model.add(Flatten()) #this converts our 3D feature maps to 1D array

model.add(Dense(64))

model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",
			optimizer="adam",
			metrics=['accuracy'])

model.fit(X,y,batch_size=32,epochs=10,validation_split=0.1)

model.save("dogCatDisq.model")