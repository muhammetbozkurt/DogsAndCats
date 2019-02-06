import pickle
from tensorflow.keras import models
import numpy as np
import tensorflow.keras.backend as k #for changing lr value

model = models.load_model("dogCatDisq.model")

model.set_value(model.optimizer.lr, 1e-3)#for changing lr value

x = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

model.fit(x,y,batch_size=64,validation_split=0.1,epochs = 40)