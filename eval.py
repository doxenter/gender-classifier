from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from scipy import ndimage, misc
from PIL import Image
import h5py
import os 
import tensorflow as tf 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback, EarlyStopping
from keras.preprocessing import image
import time

#----------------------------------------#
#        EVALUATION ON NEW DATAS         #
#----------------------------------------#

model = load_model('model_new.h5')


def MF_classifier_response(pred):
    #1 se femmina 0 se maschio,adiamo a printare il risultato a terminale e il relativo grado di confidenza 
    if pred < 0.5:
        print(">> valutazione:   maschio"+"    |   grado di confidenza: "+str(1-pred[0][0])+"%")
    else :
        print(">> valutazione:   femmina"+"    |   grado di confidenza: "+str(pred[0][0])+"%")

def MF_evaluate_img(img):
    #effettua la valutazione da parte della rete dell'immagine che abbiamo
    #inserito all'interno della rete
    img = img.resize((150, 150), Image.ANTIALIAS)
    x = image.img_to_array(img)
    print(x.shape)
    x = x.reshape(1,150,150,3)/255
    a = time.time()
    pred = model.predict(x)
    print(">> tempo di risposta : "+ str(time.time()-a))
    MF_classifier_response(pred)
    

while True:

    path = input("inserisci il percordo dell'immagine: \n >> ") #ex C:/Users/atled/Desktop/imgtest/h.jpg
    try :
        img = image.load_img(path, target_size=(150,150)) 
        MF_evaluate_img(img)
    except : 
        print("percorso inserito non valido")

