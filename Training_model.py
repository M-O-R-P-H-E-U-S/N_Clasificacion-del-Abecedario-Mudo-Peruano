import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importando el dataset en formato csv

train = pd.read_csv('Archivo csv de entrenamiento')
test = pd.read_csv('Archivo csv de prueba')

# Definiendo las etiquetas

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']

# Dimensiones de nuestras fotos en px x px

t = 50

# Mosrando las 5 primeras filas

print(train.head())

labels = train['label'].values

plt.figure(figsize = (18, 8))
sns.countplot(x = labels)
plt.xlabel('Conteo')
plt.ylabel('Etiquetas')
plt.grid()
plt.show()
# Eliminando etiquetas de entrenamiento de nuestros datos de entrenamiento para que podamos separarlos
train.drop('label', axis = 1, inplace = True)

images = train.values
images = np.array([np.reshape(i, (t, t,3)) for i in images])
images = np.array([i.flatten() for i in images])

from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

# Split our data into x_train, xtest, y_train and y_test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)

y = y_test.copy()

# Comenzando a cargar nuestros modulos de tensorflow y definiendo el tamaño de nuestro bach
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

batch_size = 128
num_classes = 23
epochs = 100

# Normalizando nuestras imagenes, dividiendo entre 255 para que nuestro array contenga valores entre 0 y 1.
x_train = x_train/255
x_test  = x_test/255

# Remodelando al tamaño requerido por TF y Keras
x_train = x_train.reshape(x_train.shape[0], t, t, 3)
x_test  = x_test.reshape(x_test.shape[0], t, t, 3)

from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(featurewise_center=False, 
    samplewise_center=False, 
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=15,
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.3, 
    zoom_range=0.1,
    channel_shift_range=0.0, 
    fill_mode="nearest",
    horizontal_flip=False, 
    vertical_flip=True, 
    validation_split=0.0,)

datagen.fit(x_train)

# Comienza la construcción del modelo en Keras: Sequential.
model = Sequential()

# Primera capa Conv2D con un numero de filtros: 64.  Ancho de la ventana convolucional: 3.
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', input_shape = (t, t, 3)))

# Introduciendo la primera capa de pooling, para reducir las dimensiones de salida
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.20))

# Primera capa Conv2D con un numero de filtros: 64  Ancho de la ventana convolucional: 3.
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

# Introduciendo la tercera capa de pooling, para reducir las dimensiones de salida
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.20))

# Primera capa Conv2D con un numero de filtros: 64  Ancho de la ventana convolucional: 3.
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

# Introduciendo la cuarta capa de pooling, para reducir las dimensiones de salida
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.20))

# La instruccion Flatten() convierte los elemntos de la matriz de la imagen en un array plano.
model.add(Flatten())

# Cuarta capa con 128 nudos.
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

# Capa de salida con activacion "softmax" y 24 nudos.
model.add(Dense(num_classes, activation = 'softmax'))

# Compilando nuestro modelo
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])

print(model.evaluate(x_train, y_train))

model.summary()

history = model.fit(datagen.flow(x_train, y_train,batch_size = batch_size), validation_data = (x_test, y_test), epochs = epochs)

# Viendo nuestro historial de formacion de forma grafica

plt.plot(history.history['accuracy'],'g', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r',label = 'val_accuracy')
plt.title("Accuracy")
plt.xlabel('Epocas')
plt.ylabel('Accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

plt.plot(history.history['loss'],'g', label='loss')
plt.plot(history.history['val_loss'], 'r',label = 'val_loss')
plt.title("Loss")
plt.xlabel('Epocas')
plt.ylabel('Loss')
plt.legend(['train','test'])
plt.grid()
plt.show()

test_labels = test['label']
test.drop('label', axis = 1, inplace = True)

test_images = test.values
test_images = np.array([np.reshape(i, (t, t,3)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], t, t, 3)

test_images.shape

y_pred = model.predict(test_images)

# Obtiendo nuestra puntuacion de accuracy
from sklearn.metrics import accuracy_score

accuracy_score(test_labels, y_pred.round())

import tensorflow as tf
tf.keras.utils.plot_model(model, to_file="C:/Users/angel/Downloads/Spyder/model.jpg",show_shapes=True,show_dtype=True, show_layer_names=True,rankdir="TB",expand_nested=False,dpi=96,)
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='C:/Users/angel/Downloads/Spyder/model_plot.png', show_shapes=True, show_layer_names=True)
tf.keras.utils.plot_model(model, to_file='C:/Users/angel/Downloads/Spyder/model_plot01.jpg', show_shapes=True)
prediction = model.predict([x_test])

for i in range(10):
    print(class_names[np.argmax(prediction[i])])
    print(np.argmax(prediction[i]))
    
for i in range(10):
    print(class_names[np.argmax(y_test[i])])
    print("labels",labels[i])

plt.figure(figsize=(15, 19))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.imshow(x_test[i],cmap = plt.cm.binary)
    plt.title("prediction: "+ class_names[np.argmax(prediction[i])] +"\n-----------\n" +"Actual: " + class_names[np.argmax(y_test[i])])
plt.show()

print("Se esta evaluando el modelo")
test_loss, test_acc = model.evaluate(x_train,y_train, verbose=2)

print("El accuracy del modelo es:")
print(test_acc)

model.save("RGB_50_100Epocas.h5")

print("Modelo guardado")