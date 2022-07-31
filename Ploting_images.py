import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importando el dataset en csv

train = pd.read_csv('Archivo de entrenamiento csv')
test = pd.read_csv('Archivo de testeo csv')

# Definiendo las clases

names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ','O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Dimensiones de nuestras fotos en px x px

t = 50

# Creando arreglos de entrenamiento y prueba

train_set = np.array(train, dtype = 'float32')
test_set = np.array(test, dtype='float32')

# Definiendo las dimensiones de nuestro ploteo

W_grid = 5
L_grid = 5

fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25))

axes = axes.ravel() 

# Obteniendo nuestro dataset de entrenamiento

n_train = len(train_set) 

# Selecionando un numero aleatorio desde 0 a n_train, y ploteando por index

# El siguiente ploteo es solo para GRAYSCALAR

for i in np.arange(0, W_grid * L_grid): 

    index = np.random.randint(0, n_train)
    fig.colorbar(axes[i].imshow( train_set[index,1:].reshape((t,t,3)) ), ax=axes[i])
    label_index = int(train_set[index,0])
    axes[i].set_title(names[label_index], fontsize = 27)
    axes[i].axis('off')
    
plt.subplots_adjust(wspace = 0.1 )

# Preparando el conjunto de datos de entrenamiento y prueba

X_train = train_set[:, 1:]/255
Y_train = train_set[:, 0]

X_test = test_set[:, 1:]/255
Y_test = test_set[:,0]

# Visualizando las imagenes de entrenamiento

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.colorbar(plt.imshow(X_train[i].reshape((t,t,'Numero de canales')), cmap=plt.cm.binary))
    label_index = int(Y_train[i])
    plt.title(names[label_index])
plt.show()
