import numpy as np
import cv2


# ========================================================
#                   MOSTRAR IMAGEN
# ========================================================

def mostrarImagen(titulo, imagen):
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(imagen.shape) == 3:
        alto, ancho, canales = imagen.shape
    else:
        alto, ancho = imagen.shape

    print(f"Imagen: {titulo} Alto: {alto}, Ancho: {ancho} \n")
    return


# ========================================================
#                   CARGAR IMAGEN
# ========================================================

def cargarImagen(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    if imagen is None:
        print(f"Error al cargar la imagen {rutaImagen}. \n")
        return
    mostrarImagen(rutaImagen, imagen)
    return


# ========================================================
#                 MOSTRAR MATRIZ Y TAMAÑO
# ========================================================

def mostrarMatriz(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)   
    if imagen is None:
        print(f"Error al cargar la imagen {rutaImagen}.")
        return 
    alto, ancho, canales = imagen.shape
    print(f"Imagen: {rutaImagen} Alto: {alto}, Ancho: {ancho} \n")
    print(imagen, "\n")
    return


# ========================================================
#                   RECORTAR IMAGEN
# ========================================================

def recortarImagen(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int)-> None:
    try:
        # Abrir la imagen
        image = cv2.imread(ruta_img)

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardar la imagen recortada en la ruta indicada
        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de " + str(image_crop.shape) + "\n")
    except Exception as e:
        print("Ha ocurrido un error:", str(e) + "\n")


# ========================================================
#                       GREYSCALE
# ========================================================  

def greyscale(rutaImagen: str, rutaImagenGrey: str):
    imagen = cv2.imread(rutaImagen)
    if imagen is None:
        print(f"Error al cargar la imagen {rutaImagen}. \n")
        return
    
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(rutaImagenGrey, imagen)
    mostrarImagen("Imagen Gris", imagen)
    return

    
# ========================================================
#                       TRASPUESTA
# ========================================================    
    
def traspuesta(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    if imagen is None:
        print(f"Error al cargar la imagen {rutaImagen}. \n")
        return
    
    imagen = np.transpose(imagen, (1, 0, 2)) # Obtener la matriz traspuesta de la imagen

    mostrarImagen(f"Imagen Traspuesta {rutaImagen}", imagen)
    print(f"Matriz traspuesta {rutaImagen}: \n", imagen, "\n")
    return


# ========================================================
#                   MATRIZ INVERSA (EJ 7)
# ========================================================

def inversa(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    if imagen is None:
        print(f"Error al cargar la imagen {rutaImagen}. \n")
        return
    
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # Convertir la imagen a escala de grises
    determinante = np.linalg.det(imagen) # Calcular el determinante de la matriz

    if determinante != 0:
        matrizInversa = np.linalg.inv(imagen) # Calcular la inversa de la matriz
        print(f"Matriz Inversa {rutaImagen}: \n")
        print(matrizInversa, "\n")
    else:
        print("No existe inversa para esta imagen. \n")
    return


# ========================================================
#              MULTIPLICAR POR ESCALAR (EJ 8)
# ========================================================

def multiplicarPorEscalar(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    if imagen is None:
        print(f"Error al cargar la imagen {rutaImagen}. \n")
        return
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Escalares. 
    escalar1 = 1.7 # CASO: escalar > 1
    escalar2 = 0.5 # CASO: 0 < escalar < 1

    # Multiplicar la matriz por los escalares
    matrizResultante1 = imagen * escalar1
    matrizResultante2 = imagen * escalar2

    # Aplicamos np.clip para asegurarse de que el valor máximo sea 255.
    matrizResultante1 = np.clip(matrizResultante1, 0, 255).astype(np.uint8)
    matrizResultante2 = np.clip(matrizResultante2, 0, 255).astype(np.uint8)
    
    # Mostramos las dos imagenes multiplicadas por los escalares.
    mostrarImagen(f"Multiplicar por escalar {escalar1}", matrizResultante1)
    mostrarImagen(f"Multiplicar por escalar {escalar2}", matrizResultante2)
    return 


# ========================================================
#              MULTIPLICAR MATRICES (EJ 9)
# ========================================================

def multiplicarMatrices(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    if imagen is None:
        print(f"Error al cargar la imagen {rutaImagen}. \n")
        return
    
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    alto, ancho = imagen.shape
    
    matrizIdentidad = np.identity(alto, dtype=np.uint8)
    antiDiagonal = np.fliplr(matrizIdentidad)
    
    resultado1 = np.dot(imagen, antiDiagonal)
    resultado2 = np.dot(antiDiagonal, imagen)
    
    mostrarImagen("Multiplicar imagen x antidiagonal", resultado1)
    mostrarImagen("Multiplicar antidiagonal x imagen", resultado2)
    return


# ========================================================
#                   NEGATIVO (EJ 10)
# ========================================================

def negativo(rutaImagen: str):
    imagen = cv2.imread(rutaImagen)
    if imagen is None:
        print(f"Error al cargar la imagen {rutaImagen}. \n")
        return
    matrizAuxiliar = np.full_like(imagen, 255)
    resultado = matrizAuxiliar - imagen
    
    mostrarImagen("Negativo", resultado)
    return


# ========================================================
#                       MAIN
# ========================================================

ardillaOriginal = "imagenes/ardilla.jpg"
perroOriginal = "imagenes/perro.jpg"

ardillaRecortada = "imagenes/ardillaRecortada.jpg"
perroRecortada = "imagenes/perroRecortada.jpg"

ardillaGris = "imagenes/ardillaGris.jpg"
perroGris = "imagenes/perroGris.jpg"

# EJ 1 Y 2
cargarImagen(ardillaOriginal)
cargarImagen(perroOriginal)

# EJ 3
#recortarImagen(ardillaOriginal, ardillaRecortada, 0, 700, 0, 700)
#recortarImagen(perroOriginal, perroRecortada, 0, 700, 0, 700)
#cargarImagen(ardillaRecortada)
#cargarImagen(perroRecortada)

# EJ 4
#mostrarMatriz(ardillaRecortada)

# EJ 5
#traspuesta(ardillaRecortada)
#traspuesta(perroRecortada)

# EJ 6
#greyscale(ardillaRecortada, ardillaGris)
#greyscale(perroRecortada, perroGris)

# EJ 7
#inversa(ardillaGris)
#inversa(perroGris)

# EJ 8
#multiplicarPorEscalar(ardillaGris)

# EJ 9
#multiplicarMatrices(ardillaGris)

# EJ 10
#negativo(ardillaGris)



