from keras.models import load_model
import cv2 as cv
import tensorflow as tf
import numpy as np

# Cargando el modelo entrenado

model = tf.keras.models.load_model("Archivo con extension h5")

names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']

# Con 03 canales (RGB)
def prepare(filepath):
    img_array = cv.imread(filepath)
    # En caso se use para escala de grises cv.imread(filepath, cv.IMREAD_GRAYSCALE) y se cambian los 3 por 1
    img_array = img_array / 255
    # Los 50 representan las dimensiones de los frames
    new_array = cv.resize(img_array, (50, 50),3)
    return new_array.reshape(-1, 50, 50, 3)

prediction = model.predict([prepare("DEBE_SER_AA.jpeg")])
print("Debe ser: A . Obtengo: ",names[np.argmax(prediction[0])])

prediction = model.predict([prepare("DEBE_SER_A.jpg")])
print("Debe ser: A . Obtengo: ",names[np.argmax(prediction[0])])

prediction = model.predict([prepare("DEBE_SER_G.jpeg")])
print("Debe ser: G . Obtengo: ",names[np.argmax(prediction[0])])

prediction = model.predict([prepare("DEBE_SER_G_A.jpg")])
print("Debe ser: G o A . Obtengo: ",names[np.argmax(prediction[0])])

prediction = model.predict([prepare("DEBE_SER_L.jpeg")])
print("Debe ser: L . Obtengo: ",names[np.argmax(prediction[0])])

prediction = model.predict([prepare("DEBE_SER_W.jpg")])
print("Debe ser: W . Obtengo: ",names[np.argmax(prediction[0])])

prediction = model.predict([prepare("DEBE_SER_Y.jpeg")])
print("Debe ser: Y . Obtengo: ",names[np.argmax(prediction[0])])
