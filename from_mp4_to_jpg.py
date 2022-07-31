import cv2 as cv


capture = cv.VideoCapture('Abriendo carpeta donde se encuentra nuestros videos')

# Inicializando el conteo de las fotos

cont = 0
path = 'Carpeta donde guardaremos nuestras fotos'
    
while (capture.isOpened()):
        
    ret, frame = capture.read()

    # Convirtiendo nuestros frames(fotogramas) a escala de grises
        
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
    # Definiendo las dimensiones de nuestras fotos
    
    width = 50
    height = 50

    dsize = (width, height)
    gray = cv.resize(gray, dsize)
        
    if (ret == True):
        
        # Ordenando nuestras fotos de manera ascendente 
        cv.imwrite(path + 'imagen_%04d.jpg' % cont,gray)
        
        cont += 1
        if (cv.waitKey(1) == ord('s')):
            break
    else:
        break
        
capture.release()
cv.destroyAllWindows()