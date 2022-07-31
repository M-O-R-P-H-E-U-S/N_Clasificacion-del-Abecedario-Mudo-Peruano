Reconocimiento del abecedario en el Lenguaje de Señas Peruano(LSP)
========

Introduccion
-----

Ha habido personas sordas en todos los momentos de la historia humana, y en todas partes en que estas han formado una comunidad, una lengua de señas ha surgido.

El lenguaje de señas es un tipo de lenguaje que utiliza movimientos de manos, expresiones faciales y lenguaje corporal para comunicarse. Lo utilizan las personas sordas y las personas que no pueden hablar pero si oir. Pero también lo utilizan personas oyentes, con mayor frecuencia parientes de las personas sordas, e intérpretes que permiten que los sordos y comunidades más amplias se comuniquen entre sí.

El Lenguaje de Señas(LSP) es una lengua originaria peruana creada por la comunidad sorda del Peru. De igual manera a todas las lenguas, a partir de ella se construye la identidad, la cultura, el conocimiento ancestral y las prácticas sociales de sus usuarios. 

Por tal motio el Lenguaje de Señas entre países y regiones puede diferir en diversas partes de su estructura y gramatica, el Lenguaje de Señas Peruano(LSP) es diferente al que se usa en Brasil, USA, Bolivia, Uruguay, Argentina o en Chile.

Con la ayuda de la ciencia y la tecnología avanzadas, sean desarrollado muchas técnicas para que las personas sordas se puedan comunicar con fluidez.


Al igual que un infante aprende a reconocer objetos, necesitamos elaborar un algoritmo para que la computadora pueda aprender, millones de imágenes antes de que pueda generalizar la entrada y hacer predicciones para imágenes que nunca antes había visto.

Las computadoras "ven" de una manera diferente a como lo hacemos nosotros. Su mundo consiste sólo en números. Cada imagen se puede representar como conjuntos bidimensionales de números, conocidos como píxeles.

Pero el hecho de que perciban las imágenes de manera diferente no significa que no puedan ser entrenadas para reconocer patrones como lo hacemos los humanos.

Para enseñar un algoritmo sobre cómo reconocer objetos en imágenes, utilizamos un tipo específico de red neuronal artificial: una red neuronal convolucional (CNN). 

Una Red Neuronal Convolucional(CNN) toma una imagen como entrada, donde distingue sus objetos en función de tres planos de color e identifica varios espacios de color. Ademas mide las dimensiones de la imagen. Ilustrando el proceso anterior:

![cnn](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/cnn.png)

En la imagen anterior, tenemos varios colores basados en el plano de tres colores que es rojo(R), verde(G) y azul(B). A continuación, se identifican los diversos espacios de color en los que se encuentran las imágenes, como RGB, CMYK, escala de grises y muchos más. Una capacidad practica de las redes neuronales convolucionales es que reducen la dimensión de la imagen hasta el punto de que es más fácil de procesar. 

Motivacion
-----

Facilitar la comunicacion entre personas no sordomudas y personas sordomudas. Fortaleciendo el desarrollo personal y profesional de las personas sordomudas en la sociedad.  

Objetivos de Investigacion
-----

Construir un algoritmo que mediante la visualizacion de signos estaticos del Lenguaje de Señas pueda convertirlos a texto, haciendo uso de las redes neuronales convolucionales por siglas en ingles CNN.

Realidad
-----

El Lenguaje de Señas Peruano(LSP) no cuenta con el debido reconocimiento por parte de las políticas públicas. A pesar de ser una lengua oficial(Ley 29535), hasta el momento los textos especializados que listan las lenguas peruanas no la incluyen. 
La educación de la persona sorda en nuestro país está todavía estancada en el viejo paradigma clínico y oralista. Las niñas sordas y los niños sordos en el Perú no reciben oportunamente acceso a la lengua de señas, y quedan así privados de oportunidades valiosas para su desarrollo academico y social, lo que constituye un acto de negligencia que vulnera sus derechos básicos como persona.

Problematica
-----

El Lenguaje de Señas es un lenguaje visual, por ello nuestro proyecto no asimila los siguientes signos: j, ñ, ll, rr, s, z. Signos realizados mediante movimientos de zig-zag o de repeticion, a diferencia de los demas signos que se pueden entender de manera estatica. Debido a su complejidad visual para poder formar parte de nuestro algoritmo de reconocimiento de signos, no son asimilados en nuestro proyecto.   

Metodologia
-----

Para poder construir nuestro dataset, comenzamos grabando nuestra mano ya sea la mano izquierda o la derecha debido a que los signos se interpretan de la misma manera sin importar con que mano se realizan. Grabando cada signo durante 40s siendo 02 videos por signo(02 manos), en total 46 videos de 40s cada uno. 

Creamos las siguientes carpetas: GS_28, GS_50, RGB_28, RGB_50. Donde se almacenaran nuestros dataset conforme sea customizados. 

Seguidamente ordenamos nuestos videos en carpetas con el nombre de las letras del abecedario, cargamos nuestro dataset de videos, definiendo las dimensiones en pixeles que deseamos obtener de los frames de nuestros videos y customizando los canales a analisar(RGB y GRAYSCALE), al siguiente script:

```
from_mp4_to_jpg.py
```

Ordenamos nuestro nuevo dataset de fotos en carpetas con el nombre de las letras del abecedario. 
Cargamos nuestro dataset de fotos,ordenandolos en las filas por etiquetas y en las columnas por pixeles , al siguiente script:

```
convert_to_csv.py
```

Con un tal de 784 columnas para los frames de (28px)x(28px), 2500 columnas para los frames de (50px)x(50px) en escalas de grises. Y en los 03 canales(RGB) 2352 columnas para los frames de (28px)x(28px), 7500 columnas para los frames de (50px)x(50px).

Ahora que tenemos nuestro dataset en archivos csv,mezclaremos las filas con susrespectivas etiquetas para ser mas dinamico el aprendizaje de nuestro modelo.
Para ello cargaremos nuestro dataset de archivos csv al siguiente script:

```
shuffle.py
```
Seguidamente separamos cada archivo csv en dos tipos, el primero sera para el entrenamiento del modelo y el segundo sera para la prueba(testeo) del modelo este ultimo archivo sera de menor cantidad. Todos los archivos(csv) de entrenamiento tienen en total 32992 filas y los archivos(csv) de prueba(testeo) tienen en total 13000 filas. 

Obteniendo 04 conjuntos de datasets:

*Primer Conjunto: S_GRAY_28, T_GRAY_28

*Segundo Conjunto: S_GRAY_50, T_GRAY_50

*Tercer Conjunto: S_RGB_28, T_RGB_28

*Cuarto Conjunto: S_RGB_50, T_RGB_50

S_ : Data de entrenamiento.
T_ : Data de prueba(testeo).


Experimental
-----

Iniciamos ploteando nuestros frames con sus respectivas etiquetas, en un grid de (5)x(5): 

```
Ploting_images.py
```

Seguidamente ploteamos nuestro conteo de frames por etiquetas, para ello hacemos uso del siguiente script:

```
Ploting_images.py
```
Ademas en el codigo anterior se construye nuestro modelo de entrenamiento(Sequential), veamoslo detenidamente:

```
# La funcion ImageDataGenerator, permitira a nuestro modelo aprender mediante rotaciones de nuestros frames en su mismo plano
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
```

Guardando el modelo de entrenamiento de nuestros 04 conjuntos de dataset, en archivos de extension h5:

*Modelo GRAY_28: GRAY_28_100Epocas.h5

*Modelo GRAY_50: GRAY_50_100Epocas.h5

*Modelo RGB_28: RGB_28_100Epocas.h5

*Modelo RGB_28: RGB_50_100Epocas.h5

Una vez guardados los modelos los cargamos en el siguiente sript:

```
Using_model.py
```

Usando las siguientes imagenes obtenidas de una mano distinta y de fondo distinto al original usado para la elaboracion del dataset, las cargamos al script anterior y predecimos en texto el signo que se visualiza.

![DEBE_SER_A](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/DEBE_SER_A.jpg)

![DEBE_SER_G](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/DEBE_SER_G.jpeg)

![DEBE_SER_G_A](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/DEBE_SER_G_A.jpg)

![DEBE_SER_L](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/DEBE_SER_L.jpeg)

![DEBE_SER_W](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/DEBE_SER_W.jpg)

![DEBE_SER_Y](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/DEBE_SER_Y.jpeg)

Para poder visualizar los diagramas de los modelos se usa la siguiente funcion:
```
tf.keras.utils.plot_model("Modelo", to_file="Nombre de la imagen en extension jpg o jpeg u otra" , show_shapes=True)
```

Resultados
-----

Visualizacion de los frames con su respectivas etiquetas:

*GRAY_BINARY_28

![GRAY_BINARY_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/GRAY_BINARY_28.png)

*GRAY_GREEN_28

![GRAY_GREEN_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/GRAY_GREEN_28.png)

*GRAY_BINARY_50

![GRAY_BINARY_50](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/GRAY_BINARY_50.png)

*GRAY_GREEN_50

![GRAY_GREEN_50](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/GRAY_GREEN_50.png)

*RGB_BINARY_28

![RGB_BINARY_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/RGB_BINARY_28.png)

*RGB_BINARY_50

![RGB_BINARY_50](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/RGB_BINARY_50.png)


Ahora, mostrandose los ploteos del conteo de frames por etiqueta de cada uno de nuestros conjuntos de dataset:

*Conteo de GRAY_28

![CONTEO_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/CONTEO_28.png)

*Conteo GRAY_50

![Conteo](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/Conteo.png)

*Conteo RGB_28

![RGB_conteo_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/RGB_conteo_28.png)

*Conteo RGB_50

![RGB_conteo_50](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/RGB_conteo_50.png)

Ahora mostramos las graficas de "Accuracy vs Epocas" y "Loss vs Epocas", obtenidas de cada dataset:

*GRAY_28

![GRAY_Accuracy_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/GRAY_Accuracy_28.png)

![GRAY_Loss_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/GRAY_Loss_28.png)

*GRAY_50

![GRAY_Accuracy_50](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/GRAY_Accuracy_50.png)

![GRAY_Loss_50](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/GRAY_Loss_50.png)

*RGB_28

![RGB_Accuracy_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/RGB_Accuracy_28.png)

![RGB_Loss_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/RGB_Loss_28.png)

*RGB_50

![RGB_Accuracy_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/RGB_Accuracy_50.png)

![RGB_Loss_50](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/RGB_Loss_50.png)


Ahora la evaluacion el modelo, con dataset GRAY_28:
```
722/722 - 3s - loss: 0.0375 - accuracy: 0.9877 - 3s/epoch - 4ms/step
```
El accuracy del modelo, con dataset GRAY_28:
```
0.9877457618713379
```

Ahora la evaluacion el modelo, con dataset GRAY_50:
```
722/722 - 8s - loss: 2.5148e-04 - accuracy: 1.0000 - 8s/epoch - 11ms/step
```
El accuracy del modelo, con dataset GRAY_50:
```
0.9999567270278931
```

Ahora la evaluacion el modelo, con dataset RGB_28:
```
722/722 - 3s - loss: 0.0238 - accuracy: 0.9935 - 3s/epoch - 4ms/step
```
El accuracy del modelo, con dataset RGB_28:
```
0.9935048222541809
```

Ahora la evaluacion el modelo, con dataset RGB_50:
```
722/722 - 8s - loss: 6.2035e-04 - accuracy: 0.9999 - 8s/epoch - 11ms/step
```
El accuracy del modelo, con dataset RGB_50:
```
0.9999133944511414
```

Ahora mostrando el diagrama del modelo:

*MODEL_RGB_28:
![MODEL_RGB_28](https://github.com/M-O-R-P-H-E-U-S/N_Clasificacion-del-Abecedario-Mudo-Peruano/blob/main/MODEL_RGB_28.png)


Ahora predeciremos a que letras pertenece cada signo mostrado en la parte experimental.

* GRAY_28_100Epocas.h5:
```
1/1 [==============================] - 0s 47ms/step
Debe ser: A . Obtengo:  C
1/1 [==============================] - 0s 11ms/step
Debe ser: A . Obtengo:  C
1/1 [==============================] - 0s 12ms/step
Debe ser: G . Obtengo:  G
1/1 [==============================] - 0s 13ms/step
Debe ser: G o A . Obtengo:  P
1/1 [==============================] - 0s 12ms/step
Debe ser: L . Obtengo:  C
1/1 [==============================] - 0s 12ms/step
Debe ser: W . Obtengo:  L
1/1 [==============================] - 0s 13ms/step
Debe ser: Y . Obtengo:  C
```
* GRAY_50_100Epocas.h5:
```
1/1 [==============================] - 0s 53ms/step
Debe ser: A . Obtengo:  A
1/1 [==============================] - 0s 16ms/step
Debe ser: A . Obtengo:  A
1/1 [==============================] - 0s 13ms/step
Debe ser: G . Obtengo:  H
1/1 [==============================] - 0s 13ms/step
Debe ser: G o A . Obtengo:  L
1/1 [==============================] - 0s 12ms/step
Debe ser: L . Obtengo:  F
1/1 [==============================] - 0s 12ms/step
Debe ser: W . Obtengo:  F
1/1 [==============================] - 0s 14ms/step
Debe ser: Y . Obtengo:  K
```
* RGB_28_100Epocas.h5:
```
1/1 [==============================] - 0s 48ms/step
Debe ser: A . Obtengo:  P
1/1 [==============================] - 0s 24ms/step
Debe ser: A . Obtengo:  P
1/1 [==============================] - 0s 13ms/step
Debe ser: G . Obtengo:  F
1/1 [==============================] - 0s 11ms/step
Debe ser: G o A . Obtengo:  P
1/1 [==============================] - 0s 12ms/step
Debe ser: L . Obtengo:  F
1/1 [==============================] - 0s 11ms/step
Debe ser: W . Obtengo:  P
1/1 [==============================] - 0s 11ms/step
Debe ser: Y . Obtengo:  F
```

* RGB_50_100Epocas.h5:

```
1/1 [==============================] - 0s 40ms/step
Debe ser: A . Obtengo:  C
1/1 [==============================] - 0s 22ms/step
Debe ser: A . Obtengo:  C
1/1 [==============================] - 0s 11ms/step
Debe ser: G . Obtengo:  H
1/1 [==============================] - 0s 12ms/step
Debe ser: G o A . Obtengo:  P
1/1 [==============================] - 0s 11ms/step
Debe ser: L . Obtengo:  F
1/1 [==============================] - 0s 11ms/step
Debe ser: W . Obtengo:  P
1/1 [==============================] - 0s 11ms/step
Debe ser: Y . Obtengo:  P
```

Conclusiones
-----

* De los accuracy de los datasets usados en el modelo Sequential podemos concluir que "El accuracy del modelo, con dataset GRAY_50" es el mas alto dentro de nuestros 04 conjuntos de datasets.

* Cuantos mas canales tenga nuestros frames mas lento sera el aprendizaje de nuestra red neuronal convolucional(CNN) pero obtendremos un mejor accuracy que si usaramos tan solo 01 canal. 

* Podemos concluir que el accuracy de (28px)x(28px) en RGB es mayor pero no lo suficientemente como para no trabajar con frames de (28px)x(28px) en escalas de grises.

* Podemos concluir que el accuracy de (50px)x(50px) en escala de grises es mayor pero no lo suficientemente como para no trabajar con frames de (28px)x(28px) en RGB, hecho que no fue esperado dado que en RGB se tiene mas informacion de los frames y se podria tener un mejor aprendizaje.

* Al usar frames de mayor dimensiones de obtenemos una mejor calidad de las imagenes, en relacion (50px)x(50px) a (28px)x(28px).

* Se concluye que el modelo predice mejor a escalas de grises y con dimensiones de (50px)x(50px).

* Se concluye que no no hay una relacion directa en el numero de epocas de entrenamiento y la precision del modelo. En experimentos anteriores con 10 epocas de entrenamiento se obtuvieron mejores resultados que si se entrenasen a 100 epocas. Estando frentes a un caso de overfitting

* Nuestro dataset de entrenamiento es de 32992 para poder predecir con mayor precision se necesita un dataset mas robusto alrededor del millon para que nuestra red neuronal convolucional aprenda correctamente cada signo del Lenguaje de Señas Peruano(LSP).

Trabajos Futuros
-----

El presente proyecto traza el camino para poder desarrollar un algoritmo mas sofisticado que cuente con un dataset mas robusto, donde se pueda facilitar la comunicacion entre las personas no sordomudas y las personas sordomudas.
