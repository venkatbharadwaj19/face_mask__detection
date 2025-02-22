from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Covid-19 Face Mask Detection Using TensorFlow, Keras and OpenCV")
main.geometry("1300x1200")

global filename
global classifier
global labels

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def readLabels(filename):
    global labels
    labels = []
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)


def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    readLabels(filename)
    text.insert(END,"Dataset loaded\n")


def trainCNN():
    global classifier
    text.delete('1.0', END)
    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')
    text.insert(END,"CNN is training on total images : "+str(len(X_train))+"\n")

    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"CNN Mask Detection Training Model Prediction Accuracy = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(200, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(100, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 64, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"CNN Mask Detection Training Model Prediction Accuracy = "+str(accuracy))

    

def maskDetection():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    if len(faces) > 0:
        image = cv2.imread(filename)
        img = cv2.resize(image, (64,64))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,64,64,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = classifier.predict(img)
        predict = np.argmax(preds)

        img = cv2.imread(filename)
        img = cv2.resize(img, (400,400))
        text.insert(END,'Image Detected as : '+labels[predict]+"\n")
        cv2.putText(img, 'Image Detected as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('Image Detected as : '+labels[predict], img)
        cv2.waitKey(0)
    else:
        image = cv2.imread(filename)
        text.insert(END,"Face not detected in uploaded image\n")
        cv2.putText(image, 'Face not detected in uploaded image', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('Face not detected in uploaded image', image)
        cv2.waitKey(0)
    

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
    #for i in range(len(accuracy)):
    #    accuracy[i] = accuracy[i] * 100

     
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Accuracy & Loss Graph')
    plt.show()

def close():
    main.destroy()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Covid-19 Face Mask Detection Using TensorFlow, Keras and OpenCV',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Mask Detection Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

markovButton = Button(main, text="Train Mask Images Using CNN", command=trainCNN)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)

lexButton = Button(main, text="Upload Test Image & Detect Mask", command=maskDetection)
lexButton.place(x=50,y=250)
lexButton.config(font=font1)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
