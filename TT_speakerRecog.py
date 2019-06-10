# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 00:08:48 2018

@author: admin
"""

from keras.layers import Activation
from PIL import Image
import numpy as np
import os
import cv2
#import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.utils import np_utils
#from keras.models import model_from_json
data=[]
labels=[]

train_data = []
train_labels = []
test_data = []
test_labels = []

sp1 =os.listdir("speaker1")
for s1 in sp1 :
    imag=cv2.imread("speaker1/"+s1)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)    


a1 = len(data)  # 20 
b1 = int(a1*0.9)   # 16 file dau

for i in range(b1):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(b1, a1):
    test_data.append(data[j])
    test_labels.append(labels[j])

    
 
sp2 =os.listdir("speaker2")
for s2 in sp2 :
    imag=cv2.imread("speaker2/"+ s2)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)


a2 = a1 + b1  # 36
#print(a2)

for i in range(a1,a2):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a2, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])

#print(train_labels)
#print(test_labels)

a2 = len(data) # 40 
#print(a2)


sp3 =os.listdir("dat")
for s3 in sp3:
    imag=cv2.imread("dat/"+ s3)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)
    
   
a3 = a2 + b1 # 56

for i in range(a2,a3):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a3, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])
    
  
a3 = len(data) # 60


sp4 =os.listdir("speaker4")
for s4 in sp4:
    imag=cv2.imread("speaker4/"+ s4)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)
    
a4 = a3 + b1 # 76

for i in range(a3,a4):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a4, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])
    
a4 = len(data)  # 80
    
 
sp5 =os.listdir("speaker5")
for s5 in sp5 :
    imag=cv2.imread("speaker5/"+ s5)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(4)

a5 = a4 + b1 # 96
for i in range(a4,a5):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a5, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])
 
    
a5 = len(data)# 100

sp6 =os.listdir("speaker6")
for s6 in sp6 :
    imag=cv2.imread("speaker6/"+ s6)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(5)

a6 = a5 + b1    # 116

for i in range(a5,a6):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a6, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])
    
    
a6 = len(data) # 120


sp7 =os.listdir("speaker7")
for s7 in sp7 :
    imag=cv2.imread("speaker7/"+ s7)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(6)  
    
a7 = a6 + b1 # 138
for i in range(a6,a7):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a7, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])

a7 = len(data) # 140 


sp8 =os.listdir("speaker8")
for s8 in sp8 :
    imag=cv2.imread("speaker8/"+ s8)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(7)    
a8 = a7 + b1 # 158
for i in range(a7,a8):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a8, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])

a8 = len(data) # 160 

sp9 =os.listdir("speaker9")
for s9 in sp9 :
    imag=cv2.imread("speaker9/"+ s9)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(8)
a9 = a8 +b1  # 178
for i in range(a8,a9):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a9, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])
a9 = len(data)# 180


sp10 =os.listdir("speaker10")
for s10 in sp10 :
    imag=cv2.imread("speaker10/"+ s10)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(9)
    
a10 = a9 + b1 # 198
for i in range(a9,a10):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a10, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])
 
a10 = len(data) # 200


sp11 =os.listdir("speaker11")
for s11 in sp11 :
    imag=cv2.imread("speaker11/"+ s11)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(10)
a11 = a10 + b1  # 218
for i in range(a10,a11):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a11, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])
a11 = len (data) # 220

sp12 =os.listdir("speaker12")
for s12 in sp12 :
    imag=cv2.imread("speaker12/"+ s12)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(11)    
    
a12 = a11 + b1 # 238
for i in range(a11,a12):
    train_data.append(data[i])
    train_labels.append(labels[i])

for j in range(a12, len(data)):
    test_data.append(data[j])
    test_labels.append(labels[j])


train_speakers=np.array(train_data)
new_train_labels=np.array(train_labels)

#print(new_train_labels)

test_speakers = np.array(test_data)
new_test_labels = np.array(test_labels)


num_class = len(np.unique(new_test_labels))


np.save("train_speakers_1",train_speakers)
np.save("new_train_labels_1",new_train_labels)
np.save("test_speakers_1",test_speakers)
np.save("new_test_labels_1",new_test_labels)


load_train_speakers=np.load("train_speakers_1.npy")
load_test_speakers = np.load("test_speakers_1.npy")
load_train_labels=np.load("new_train_labels_1.npy")
load_test_labels = np.load("new_test_labels_1.npy")

x_train = load_train_speakers.astype('float32')/255
x_test = load_test_speakers.astype('float32')/255


#One hot encoding
y_train =  np_utils.to_categorical(load_train_labels,num_class) #Converts a class vector (integers) to binary class matrix
y_test =  np_utils.to_categorical(load_test_labels,num_class)

### create model for training data

model = Sequential()
model.add(Conv2D(8, (3, 3),  padding='same',
                 input_shape=(50, 50 , 3)))

model.add(Activation('relu'))

model.add(Conv2D(8, (3, 3)) )

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(16, (3, 3))  )

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(240))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(12))
model.add(Activation('softmax'))
model.summary()  # prints a summary representation of your model
model.compile(loss='categorical_crossentropy', optimizer= 'adam', 
                  metrics=['accuracy'])

#  train and evaluate model 
model.fit(x_train,y_train,batch_size= 24
          ,epochs= 100 ,verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print("\n")
print('Test accuracy:', score[1] )
print("\n")

# serialize model to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")

