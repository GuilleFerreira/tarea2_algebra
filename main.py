import numpy as np
from skimage.io import imshow, imread
import skimage.io as io
import cv2
import os

original = "images\\tomas.png"
recortada = "imagesCut\\tomas.png"
greyscalee = "imagesCut\\tomasGrey.png"
traspuestaa = "imagesCut\\tomasTraspuesta.png"

def recortar_imagen_v2(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int)-> None:
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
        
recortar_imagen_v2(original,recortada, 0, 300, 0, 300)


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

                #blue_channel = pixel_value[0]
                #green_channel = pixel_value[1]
                #red_channel = pixel_value[2]
                
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
        
greyscale(recortada, greyscalee)
traspuesta(recortada, traspuestaa)