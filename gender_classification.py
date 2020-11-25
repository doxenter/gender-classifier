import h5py
import os 
import tensorflow as tf 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback, EarlyStopping
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
#----------------------------------------#
#             DATA IMPORTING             #
#----------------------------------------#

#importiamo i dati dalle cartelle
print (os.getcwd())
train,test,valid = os.listdir("./dataset1")
test,train,valid
os.chdir('./train')
train_man, train_woman=os.listdir('./')
train_man, train_woman
train_man_names = os.listdir(train_man)
print(train_man_names[:10])
train_woman_names = os.listdir(train_woman)
print(train_man_names[:10])


#----------------------------------------#
#             DATA INSPECTION            #
#----------------------------------------#

#andiamo ora ad effettuare una ispezione dei dati interni al dataset ver verificare
#la validità dei dati by inspection

#parametri per i grafici di inspection
nrows = 4
ncols = 4

#indice generico
pic_index = 0

#plotting immagini
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_man_pix = [os.path.join(train_man, fname) 
                for fname in train_man_names[pic_index-8:pic_index]]
next_woman_pix = [os.path.join(train_woman, fname) 
                for fname in train_woman_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_man_pix+next_woman_pix):
  
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


#----------------------------------------#
#                NN MODEL                #
#----------------------------------------#

#definiamo le caatteristiche della rete sequenziale e effettiamo un ritorno del
#modello generato dalla funzione get_model()
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
    return model

model = get_model()        
model.summary()

os.chdir('../') 
print(os.listdir())


#impostazioni dataset aumentato per i dati di training
train_datagen = ImageDataGenerator(rescale=1/255,               #normalizziamo le immagini i cui pixel
    featurewise_center=False,                                   #variano da 0 a 255 in modo di inserire
    samplewise_center=False,                                    #nella rete dati che variano da 0 a 1
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=None,
    shear_range=0.15,
    zoom_range=0.15,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None
)


#impostazioni dataset aumentato per i dati di validazione
validation_datagen = ImageDataGenerator(rescale=1/255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=None,
    shear_range=0.15,
    zoom_range=0.15,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None
)

#il seguente comando è per salvare la rete a ogni epoca, utile nel caso di interruzioni del processo di training
#checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', verbose=1,   
#    save_best_only=True)#filepath=checkpoint_filepath)

early_stopping = EarlyStopping(monitor='val_acc',       #implementiamo un early stopping in cui se 
        min_delta=0.005,                                #la accuratezza non subisce significativi incrementi
        patience=10,                                    #per 10 epoche consecutive interrompiamo il training
        verbose=1,
        restore_best_weights=True,
    )

callbacks = [                                           #funzioni da richiamare quando finisce il training
            early_stopping,                             #sulla singola epoca
            checkpoint
            ]


#utilizziamo un flow di dati derivanti dal dataset aumentato in batch di dimensione 32
train_generator = train_datagen.flow_from_directory(
        './train',  
        target_size=(150, 150),  # Inseriamo immagini 150x150
        batch_size=32,
        # siccome abbiamo una funzione di costo di tipo binary crossentropy dobbiamo attuare una classificazione binaria
        class_mode='binary')

#analogo al train generator ma per i dati di validazione
validation_generator = validation_datagen.flow_from_directory(
        './valid',  
        target_size=(150, 150),  
        batch_size=32,
        class_mode='binary')

#----------------------------------------#
#                NN MODEL                #
#----------------------------------------#
#la funzione fit va ad effettuare il training della rete neurale inserendo 
#il flow di dati provenienti dal dataset aumentato
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=70,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8,
      callbacks=callbacks
        )



model.save('model_new.h5')  #salviamo i pesi e la storia del modello allenato nel file h5


#----------------------------------------#
#                STATISTICS              #
#----------------------------------------#
#funzioni per verifica statistiche della rete
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()