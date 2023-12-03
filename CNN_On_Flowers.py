from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Dropout, BatchNormalization
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

from matplotlib import pyplot as plt

import random

import cv2
from imutils import paths
import numpy as np
import os
from sklearn.model_selection import train_test_split

def createmodel(h,w,d,c):
    model = Sequential()

    #first layer
    model.add(Conv2D(50,(5,5),padding = "same",input_shape= (h,w,d)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))
    model.add(Dropout(0.5))
    

    #second layer
    model.add(Conv2D(30,(5,5),padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(20,(3,3),padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))

    model.add(Conv2D(30,(3,3),padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2)))
    model.add(Dropout(0.15))
    
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(300))
    model.add(Activation("relu"))
    model.add(Dense(c))
    model.add(Activation("softmax"))

    return model


h,w,d = 128,128,3
n_classes = 12
epochs = 10
bsize = 16
data = []
labels = []

lab_dict = {'bluebell':0,'buttercup':1,'crocus':2,'daffodil':3,'daisy':4,'dandelion':5,'iris':6,'lilyvalley':7,'pansy':8,'snowdrop':9,'sunflower':10,'tulip':11}

imagepaths=sorted(list(paths.list_images("Flowers")))

random.seed(0)
random.shuffle(imagepaths)

for imgpath in imagepaths:
    image = cv2.imread(imgpath)
    image = cv2.resize(image,(h,w))
    image = np.array(image)
    data.append(image)
    lab = imgpath.split(os.path.sep)[-2]
    labels.append(lab_dict[lab])

data = np.array(data, np.float64)
labels = np.array(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size= 0.15,random_state = 0)
(trainX,ValX,trainY,ValY) = train_test_split(trainX,trainY,test_size= 0.15,random_state = 0)

print("Train data Shape: ",trainX.shape)
print("Validation data Shape: ",ValX.shape)
print("Test data Shape: ",testX.shape)

trainY = to_categorical(trainY,num_classes = n_classes)
ValY = to_categorical(ValY,num_classes = n_classes)
testY = to_categorical(testY,num_classes = n_classes)


aug = ImageDataGenerator(rescale = 1./255, rotation_range = 30, width_shift_range = 0.1, height_shift_range = 0.1,
                         shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest")




model = createmodel(h,w,d,n_classes)
model.compile(loss = "categorical_crossentropy",optimizer = "adam", metrics = ['accuracy'])

p = 3
es = EarlyStopping(monitor="val_accuracy", patience = p)
ck = ModelCheckpoint(filepath = "flowers_best.model",monitor="val_accuracy", save_best_only = "True")

callbacks = [es,ck]

H = model.fit(x=trainX,y=trainY,batch_size = bsize, validation_data = (ValX,ValY), epochs = epochs, verbose = 2)

model.save("digits.model")


plt.style.use("ggplot")
plt.figure(figsize=[8,6])
N = len(H.history["loss"])
plt.plot(np.arange(0,N),H.history["loss"],label = "train-loss")
plt.plot(np.arange(0,N),H.history["val_loss"],label = "val-loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss.png")

plt.style.use("ggplot")
plt.figure(figsize=[8,6])
N = len(H.history["loss"])
plt.plot(np.arange(0,N),H.history["accuracy"],label = "train-acc")
plt.plot(np.arange(0,N),H.history["val_accuracy"],label = "val-acc")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("accuracy.png")


print("Model Evaluation")
print(model.evaluate(testX,testY))

print("MOdel Predictions")
predY = model.predict(testX)
predY = np.argmax(predY,axis = 1)
testY = np.argmax(testY,axis = 1)
print(predY)
print(testY)






















