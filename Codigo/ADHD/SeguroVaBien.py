import os
import numpy as np
from networkx.algorithms.bipartite.basic import color
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, Model

import argparse

# Definir valores por defecto para ejecutar.
EJECUTAR_PCA = False
EJECUTAR_ENTRENAMIENTO = False
EJECUTAR_TESTS = True


# Función que aplica la reduccion de dimensionalidad
def reduccion_dimensionalidad(directorio):
    grafos_reducidos = {}
    for archivo in os.listdir(directorio):
        if archivo.endswith(".txt"):
            ruta_archivo = os.path.join(directorio, archivo)
            print(f"Leyendo el archivo: {archivo}")
            try:
                matriz_adyacencia = np.loadtxt(ruta_archivo)
                # Nos quedamos con el triangulo superior derecho
                indices_superiores = np.triu_indices(matriz_adyacencia.shape[0], k=1)
                triangulo_superior = matriz_adyacencia[indices_superiores]

                vector_grafos = np.ravel(triangulo_superior)
                # Guardamos el vector en el diccionario
                grafos_reducidos[archivo] = vector_grafos
                print(f"Grafo {archivo} procesado correctamente.")
            except Exception as e:
                print(f"Error al leer el archivo {archivo}: {e}")
    # Aplicamos transformacion para que sea aceptado directamente por el PCA
    grafos_reducidos = np.array(list(grafos_reducidos.values()))
    return grafos_reducidos


# Función que aplica PCA y grafica los resultados
def aplicar_pca_y_graficar(matriz_vectores, titulo, n_componentes):
    # Definir el color del texto según el título
    if titulo == "ADHD":
        color_titulo = "\033[31m"
    elif titulo == "TD":
        color_titulo = "\033[32m"
    elif titulo == "ADHD y TD Combinados":
        color_titulo = "\033[34m"

    fin_color = "\033[0m"

    # Aplicamos PCA
    pca = PCA(n_componentes)
    matriz_reducida = pca.fit_transform(matriz_vectores)

    # Prints para depurar resultados obtenidos
    print(f"Varianza{color_titulo} {titulo}{fin_color} con{color_titulo} {n_componentes}"
          f"{fin_color} componentes:{color_titulo} {pca.explained_variance_ratio_.cumsum()}{fin_color}")
    print(f"Resultado total:{color_titulo}{100 * pca.explained_variance_ratio_.cumsum()[-1]}{fin_color}")
    # Resultados PCA
    # plt.figure(figsize=(10, 7))
    # plt.scatter(matriz_reducida[:, 0], matriz_reducida[:, 1], c='b', marker='o')

    # Títulos y etiquetas de los ejes
    # plt.title(f'PCA de {titulo}')
    # plt.xlabel(f'Numero de Componentes: {n_componentes}')
    # plt.grid()
    # plt.show()

    # Plot explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cumulative_variance_ratio, marker='o')
    plt.xlabel('Numeros de Componentes Principales')
    plt.ylabel('Relación de varianza explicada')
    plt.title(f'{titulo}')
    plt.show()


# Definir el autoencoder como función
def crear_autoencoder(input_dim):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(5000, activation='relu')(inputs)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dense(250, activation='relu')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(15, activation='relu')(x)
    encoded = layers.Dense(1, activation='relu')(x)

    # Decoder
    x = layers.Dense(15, activation='relu')(encoded)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(250, activation='relu')(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(5000, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)

    # Definir el modelo autoencoder como una función de encoder + decoder
    autoencoder_model = Model(inputs, decoded)
    cuello_botella = Model(inputs, encoded)

    return autoencoder_model, cuello_botella


class Main:
    def __init__(self):
        # Ruta para leer los ficheros
        self.directorio_ADHD = r'C:\Users\Jorge\Desktop\Universidad\TFG\Datos\data\ADHD\graphs\adhd'
        self.directorio_TD = r'C:\Users\Jorge\Desktop\Universidad\TFG\Datos\data\ADHD\graphs\td'
        self.grafos_reducidos_ADHD = None
        self.grafos_reducidos_TD = None
        self.grafos_reducidos_Combinados = None

        # Modelamos los datos para que tengan la estructura necesaria de la PCA
        # ADHD
        self.grafos_reducidos_ADHD = reduccion_dimensionalidad(self.directorio_ADHD)
        # TD
        self.grafos_reducidos_TD = reduccion_dimensionalidad(self.directorio_TD)
        # ADHD & TD
        self.grafos_reducidos_Combinados = np.vstack((self.grafos_reducidos_ADHD, self.grafos_reducidos_TD))

    """Divide los datos de entrenamiento y test"""

    def tests(self, grafos):
        # Usamos el train split de sklearn para dividir de manera aleatoria
        contador = 0
        lista_porcentajes = []
        grafos_entrenamiento, grafos_test = train_test_split(grafos, test_size=0.2, random_state=42)
        # Creamos el autoencoder y lo entrenamos con el entrenamiento
        autoencoder, encoder = crear_autoencoder(input_dim=17955)
        # Compilamos y aprendemos
        learning_rate = 0.0005  # Puedes probar con diferentes valores como 0.001, 0.0001, etc.
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mse')
        autoencoder.fit(grafos_entrenamiento, grafos_entrenamiento, epochs=20, batch_size=64)

        for grafo in grafos_test:
            reconstruccion, cuello_botella = self.obtener_resultados(grafo, autoencoder, encoder)
            contador += 1
            print(f"GRAFO Nº {contador}:")
            print(f"Cuello de Botella {contador}: {cuello_botella}")
            print(f"{len(grafo)}")
            diferencias = grafo != reconstruccion  # Esto crea un array de booleanos
            # Contar cuántos elementos no coinciden (cuántos 'True' hay)
            numero_diferencias = np.sum(diferencias)
            porcentaje_acierto = 100 - ((numero_diferencias / len(grafo)) * 100)
            lista_porcentajes.append(porcentaje_acierto)
            # Mostrar los resultados
            print(f"{np.sum(reconstruccion)}")
            print(f"{np.sum(grafo)}")
            print(f"{reconstruccion}")
            print(f"El porcentaje de acierto es: {porcentaje_acierto}")
        porcentaje_acierto_medio = sum(lista_porcentajes) / len(lista_porcentajes)
        return porcentaje_acierto_medio

    # Itera para llamar a entrenar_autoencoder en todos el grafo y obtiene los resultados despues
    def bucle_entrenar_y_resultados(self, grafos):
        # Creamos el Autocodificador general
        autoencoder, encoder = crear_autoencoder(input_dim=17955)
        contador = 0
        lista_porcentajes = []
        # Compilamos el modelo
        learning_rate = 0.0005  # Puedes probar con diferentes valores como 0.001, 0.0001, etc.
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mse')
        # ENTRENAMOS UNA SOLA VEZ CON TODOS LOS GRAFOS
        autoencoder.fit(grafos, grafos, epochs=50, batch_size=64)

        for grafo in grafos:
            reconstruccion, cuello_botella = self.obtener_resultados(grafo, autoencoder, encoder)
            contador += 1
            print(f"GRAFO Nº {contador}:")
            print(f"Cuello de Botella {contador}: {cuello_botella}")
            diferencias = grafo != reconstruccion  # Esto crea un array de booleanos
            # Contar cuántos elementos no coinciden (cuántos 'True' hay)
            numero_diferencias = np.sum(diferencias)
            porcentaje_acierto = 100 - ((numero_diferencias / len(grafo)) * 100)
            lista_porcentajes.append(porcentaje_acierto)

            # Mostrar los resultados
            print(f"El porcentaje de acierto es: {porcentaje_acierto}")

        # Retornamos el porcentaje medio
        porcentaje_acierto_medio = sum(lista_porcentajes) / len(lista_porcentajes)
        return porcentaje_acierto_medio

    # Una vez esta la red neuronal entrenada, esta funcion
    # Autoencoder: La reconstruccion
    # Encoder: El cuello de botella
    def obtener_resultados(self, grafo, autoencoder, encoder):
        grafo = np.expand_dims(grafo, axis=0)  # Convertimos de 1D a 2D
        reconstruccion = autoencoder.predict(grafo)
        cuello_botella = encoder.predict(grafo)
        return reconstruccion, cuello_botella


class Run:
    @staticmethod
    def ejecutar():
        # Instancia de la clase Main
        main = Main()

        # Iterar sobre los grafos reducidos de ADHD
        # main.entrenar_autoencoders()

        # porcentaje_medio_adhd = main.bucle_entrenar_y_resultados(main.grafos_reducidos_ADHD)
        # porcentaje_medio_td = main.bucle_entrenar_y_resultados(main.grafos_reducidos_TD)
        # porcentaje_medio_combinados = main.bucle_entrenar_y_resultados(main.grafos_reducidos_Combinados)
        # print(f"ADHD: {porcentaje_medio_adhd}, TD: {porcentaje_medio_td}, Combinados: {porcentaje_medio_combinados}")
        acierto_test_adhd = main.tests(main.grafos_reducidos_ADHD)

        acierto_test_td = main.tests(main.grafos_reducidos_TD)
        acierto_test_Combinados = main.tests(main.grafos_reducidos_Combinados)
        print(f"ADHD: {acierto_test_adhd}, TD: {acierto_test_td}, Combinados: {acierto_test_Combinados} ")

        # iteramos el numero de componentes principales para ver su variabilidad
        ##for n in range(2, 50):
        # ADHD
        ##aplicar_pca_y_graficar(main.grafos_reducidos_ADHD,"ADHD", n)
        # TD
        ##aplicar_pca_y_graficar(main.grafos_reducidos_TD, "TD", n)
        # Combined
        ##aplicar_pca_y_graficar(main.grafos_reducidos_Combinados, "ADHD y TD Combinados", n)


# Ejecutamos el flujo principal
if __name__ == "__main__":
    Run.ejecutar()
