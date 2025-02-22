import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

# Non-Binary Image Classification using Convolution Neural Networks

path = 'flowers'

labels = []
X_train = []
Y_train = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
classifier.load_weights("model/model_weights.h5")
classifier._make_predict_function()   
print(classifier.summary())


imgs = ['sun.jpg','tup.jpg','dai.jpg','dan.jpg','ros.jpg']
for i in range(len(imgs)):
    image = cv2.imread('testImages/'+imgs[i])
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)
    print(str(labels[predict])+" "+str(imgs[i]))










