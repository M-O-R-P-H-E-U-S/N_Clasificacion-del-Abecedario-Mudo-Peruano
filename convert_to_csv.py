from PIL import Image
import numpy as np
import os
import csv

# Definiendo una funcion que nos permita abrir nuestro dataset de fotos

def FileList(myDir, format='.jpg'):
    fileList = []
    
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# Definiendo una funcion que nos permita retornar cada letra del alfabeto de manera ordenada(ascendente)

def listAlphabet():
  return [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Cargando las imagenes en orden alafabetico

for i in listAlphabet():    
    
    myFileList = FileList('Ruta donde se encuentra nuestro dataset de fotos' + i + ' ')
    
    for file in myFileList:
        
        # Cargando la imagen nuestra imagen
        img_file = Image.open(file)
        
        # Redifiniendo la variable
        r = img_file

        # Redimensionando nuestras fotos en arreglos, en Grayscale el numero de canales es 1, en RGB es 3
        
        r = np.asarray(r.getdata(), dtype=np.int).reshape((r.size[1], r.size[0],'Numero de canales'))


        r = r.flatten()

        with open("Archivo con extencion csv que deseamos crear", 'a',newline="")as f:
            
            # Escribiendo nuestro archivo csv, para obtener nuestro dataset en csv
            
            writer = csv.writer(f)
            writer.writerow(r)