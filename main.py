import numpy as np
from skimage.io import imshow, imread
import skimage.io as io
import cv2
import os

# ========================================================
#                   MOSTRAR IMAGEN
# ========================================================
def mostarImagen(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    cv2.imshow(rutaImagen, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    alto, ancho, canales = imagen.shape

    print(f"Imagen: {rutaImagen} Alto: {alto}, Ancho: {ancho}")
    return


# ========================================================
#                 MOSTRAR MATRIZ Y TAMAÑO
# ========================================================
def mostrarMatriz(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)    
    alto, ancho, canales = imagen.shape
    print(f"Imagen: {rutaImagen} Alto: {alto}, Ancho: {ancho}")
    print(imagen)
    return

# ========================================================
#                   RECORTAR IMAGEN
# ========================================================

def recortar_imagen(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int)-> None:
    try:
        # Abrir la imagen
        image = cv2.imread(ruta_img)

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardar la imagen recortada en la ruta indicada
        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de" + str(image_crop.shape))
    except Exception as e:
        print("Ha ocurrido un error:", str(e))

# ========================================================
#                       GREYSCALE
# ========================================================  

def greyscale(rutaImagen: str, rutaImagenGrey: str):
    imagen = cv2.imread(rutaImagen)
    print(imagen)
    if imagen is not None:
        alto, ancho, canales = imagen.shape
        for row in range(alto):
            for col in range(ancho):
                pixel = imagen[row, col]
                grayscale = sum(pixel) // 3
                imagen[row, col] = [grayscale, grayscale, grayscale]

        cv2.imwrite(rutaImagenGrey, imagen)

        cv2.imshow('Imagen greyscaled', imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error al cargar la imagen.")
    
# ========================================================
#                       TRASPUESTA
# ========================================================    
    
def traspuesta(rutaImagen: str, rutaImagenTraspuesta: str):
    imagen = cv2.imread(rutaImagen)
    if imagen is not None:
        # Obtener la matriz traspuesta de la imagen
        imagen = np.transpose(imagen, (1, 0, 2))  # Intercambiar las dimensiones 0 y 1

        cv2.imwrite(rutaImagenTraspuesta, imagen)

        cv2.imshow('Imagen Traspuesta', imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error al cargar la imagen.")

# ========================================================
#                   MATRIZ INVERSA (EJ 7)
# ========================================================
def inversa(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    # Calculate the determinant of the matrix
    determinante = np.linalg.det(imagen)

    if determinante != 0:
        # Calculate the inverse using cv2.invert()
        matrizInversa = cv2.invert(imagen)
        
        if matrizInversa[1] is not None:
            inverse = matrizInversa[1]
            print("Inverse Matrix:")
            print(inverse)
        else:
            print("Matrix is singular (non-invertible).")
    else:
        print("Matrix is singular (non-invertible).")
    return

# ========================================================
#              MULTIPLICAR MATRICES (EJ 9)
# ========================================================

def multiplicar(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    filas, columnas, canales = imagen.shape
    matrizIdentidad = np.eye(filas, columnas)
    print(matrizIdentidad)
    antiDiagonal = np.fliplr(matrizIdentidad)
    resultado = np.dot(imagen, antiDiagonal)
    
    cv2.imshow('Multiplicar', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# ========================================================
#                   NEGATIVO (EJ 10)
# ========================================================
def negativo(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    matrizAuxiliar = np.full_like(imagen, 255)
    resultado = matrizAuxiliar - imagen
    
    cv2.imshow('Negativo', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return       

# ========================================================
#                       MAIN
# ========================================================

ardillaOriginal = "imagenes\\ardilla.png"
perroOriginal = "imagenes\\perro.png"

ardillaRecortada = "imagenes\\ardillaRecortada.png"
perroRecortada = "imagenes\\perroRecortada.png"

ardillaGris = "imagenes\\ardillaGris.png"
perroGris = "imagenes\\perroGris.png"

#mostarImagen(ardillaOriginal) # EJ 1
#mostarImagen(perroOriginal) # EJ 1

#recortar_imagen(perroOriginal, perroRecortada, 0, 700, 0, 700)
#recortar_imagen(ardillaOriginal, ardillaRecortada, 0, 700, 0, 700)

#mostarImagen(ardillaRecortada)
#mostrarMatriz(ardillaRecortada)


# DEL PUNTO 7 EN ADELANTE
#inversa(ardillaOriginal) # EJ 7

multiplicar(ardillaRecortada)
#negativo(ardillaRecortada)



