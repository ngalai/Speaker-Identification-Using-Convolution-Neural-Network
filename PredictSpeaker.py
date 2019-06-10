### Predict speaker by a file is lived 
"""
Created on Mon Aug  6 10:51:43 2018

@author: admin
"""
import numpy as np 
from keras.models import model_from_json
import cv2
from PIL import Image
import pandas as pd
#from keras.utils import np_utils
from spectrogram import Spec
from ProcessSignal import CropImage 


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(x_test, y_test, verbose=0)

def convert_to_array(img):    
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((128, 100))
    return np.array(image)

names = ['Nga','Thuy','Tvv1' ,'Tvv2' ,'Tvv3' ,'Tvv4' ,'Tvv5',
         'Tvv6' , 'Tvv7','Tvv8','Tvv9','Tvv10']


def get_speaker_name(label):
    
    return names[label]
   
         
def predict_speaker(file):
    
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255    
    a=[]
    a.append(ar)
    a=np.array(a)
    score=loaded_model.predict(a,verbose= 1)    
    #print(score)
    label_index=np.argmax(score)
    print("label: ",label_index)  
    #acc=np.max(score)
    speaker = get_speaker_name(label_index)
    print("name : ",speaker )
    print("\n\n")
    print(pd.DataFrame(score, columns = names ) )
    print("\n\n")
    print("The predicted is a "+ speaker)
        
image = Spec("demo.png")   
img = CropImage(image , "re_demo.png")
#img = "s2_25.png" 
predict_speaker(img)
