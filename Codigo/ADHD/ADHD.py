import os
import numpy as np
from keras.src.layers import GaussianNoise
from networkx.algorithms.bipartite.basic import color
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, Model


# Función que aplica la reduccion de dimensionalidad
def reduccion_dimensionalidad(directorio):
    grafos_reducidos = {}
    for archivo in os.listdir(directorio):
        if archivo.endswith(".txt"):
            ruta_archivo = os.path.join(directorio, archivo)
            try:
                matriz_adyacencia = np.loadtxt(ruta_archivo)
                # Nos quedamos con el triangulo superior derecho
                indices_superiores = np.triu_indices(matriz_adyacencia.shape[0], k=1)
                triangulo_superior = matriz_adyacencia[indices_superiores]

                vector_grafos = np.ravel(triangulo_superior)
                # Guardamos el vector en el diccionario
                grafos_reducidos[archivo] = vector_grafos
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

    #Prints para depurar resultados obtenidos
    print(f"Varianza{color_titulo} {titulo}{fin_color} con{color_titulo} {n_componentes}"
          f"{fin_color} componentes:{color_titulo} {pca.explained_variance_ratio_.cumsum()}{fin_color}")
    print(f"Resultado total:{color_titulo}{100 * pca.explained_variance_ratio_.cumsum()[-1]}{fin_color}")
    # Resultados PCA
    #plt.figure(figsize=(10, 7))
    #plt.scatter(matriz_reducida[:, 0], matriz_reducida[:, 1], c='b', marker='o')

    # Títulos y etiquetas de los ejes
    #plt.title(f'PCA de {titulo}')
    #plt.xlabel(f'Numero de Componentes: {n_componentes}')
    #plt.grid()
    #plt.show()

    # Plot explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cumulative_variance_ratio, marker='o')
    plt.xlabel('Numeros de Componentes Principales')
    plt.ylabel('Relación de varianza explicada')
    plt.title(f'{titulo}')
    plt.show()


# Definir el autoencoder como función
def crear_autoencoder(input_dim, entrada):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(5000, activation='relu')(inputs)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dense(250, activation='relu')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(50, activation='relu')(x)
    encoded = layers.Dense(entrada, activation='relu')(x)

    # Decoder
    x = layers.Dense(50, activation='relu')(encoded)
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


def crear_stacked_autoencoder(input_dim, latent_dim_final):
    # Definir dimensiones para cada nivel de apilamiento
    dims = [5000, 1000, 500, 250, 100, 50, latent_dim_final]
    input_layer = layers.Input(shape=(input_dim,))
    encoders = []
    decoders = []
    autoencoders = []

    # Construir los autoencoders apilados
    current_input = input_layer
    current_dim = input_dim
    # Construir y entrenar cada nivel de autoencoder
    for i, dim in enumerate(dims):

        # Añadir GaussianNoise al inicio para crear ruido en la entrada
        #Creamos el encoded para la capa en cuestion
        encoded = layers.Dense(dim, activation='relu')(current_input)
        decoded = layers.Dense(current_dim, activation='sigmoid')(encoded)

        # Crear modelo de autoencoder para esta capa
        autoencoder = Model(current_input, decoded)
        # Crear modelo de encoder para esta capa
        encoder = Model(current_input, encoded)

        # Guardar modelos
        autoencoders.append(autoencoder)
        encoders.append(encoder)

        # Modificamos la siguiente entrada para que sea el cuello de botella del anterior
        current_input = layers.Input(shape=(dim,))
        current_dim = dim
    return autoencoders, encoders

class Main:
    def __init__(self):
        # Ruta para leer los ficheros
        self.directorio_ADHD = r'C:\Users\Jorge\Desktop\Universidad\TFG\Datos\data\ADHD\graphs\adhd'
        self.directorio_TD=  r'C:\Users\Jorge\Desktop\Universidad\TFG\Datos\data\ADHD\graphs\td'
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
        autoencoder, encoder = crear_autoencoder(input_dim=17955, entrada=10)
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

    # Funcion tests pero para los autoencoders apilados
    def test_stacked_autoencoders(self, grafos):

        grafos_entrenamiento, grafos_test = train_test_split(grafos, test_size=0.2, random_state=42)

        autoencoders, encoders = crear_stacked_autoencoder(input_dim=17955, latent_dim_final=10)

        # Entrenamiento del autoencoder
        noisy_data = np.array([self.aplicar_ruido(g, 0.1) for g in grafos_entrenamiento])

        for i, autoencoder in enumerate(autoencoders):
            print(f"Entrenando nivel {i + 1} del stacked autoencoder")
            optimizer = keras.optimizers.Adam(learning_rate=0.0005)
            autoencoder.compile(optimizer=optimizer, loss='mse')
            autoencoder.fit(noisy_data, noisy_data, epochs=30, batch_size=64)

            if i < len(encoders) - 1:
                noisy_data = encoders[i].predict(noisy_data)

        # Lista de umbrales
        threshold_list = np.concatenate([
            np.geomspace(0.0001, 0.1, num=50, endpoint=False),
            np.linspace(0.1, 0.5, num=10)
        ])

        # Diccionarios para almacenar los valores de cada métrica para cada umbral
        all_tpr = {thr: [] for thr in threshold_list}
        all_tnr = {thr: [] for thr in threshold_list}
        all_fpr = {thr: [] for thr in threshold_list}
        all_fnr = {thr: [] for thr in threshold_list}

        # Proceso de test
        for grafo in grafos_test:
            input_data = np.expand_dims(grafo, axis=0)

            # Paso por los encoders
            current_representation = input_data
            for encoder in encoders:
                current_representation = encoder.predict(current_representation, verbose=0)

            cuello_botella = current_representation

            # Paso por los decoders
            current_reconstruction = cuello_botella
            for i in range(len(autoencoders) - 1, -1, -1):
                decoder_layer = autoencoders[i].layers[-1]
                current_reconstruction = decoder_layer(current_reconstruction)

            reconstruccion_mod_base = current_reconstruction.numpy()[0]

            # Evaluamos la reconstrucción en cada umbral
            for threshold in threshold_list:
                reconstruccion_mod = (reconstruccion_mod_base > float(threshold)).astype(int)
                tpr, tnr, fpr, fnr = self.matriz_confusion(grafo, reconstruccion_mod)

                # Guardamos los valores en cada umbral
                all_tpr[threshold].append(tpr)
                all_tnr[threshold].append(tnr)
                all_fpr[threshold].append(fpr)
                all_fnr[threshold].append(fnr)
                print(f"UMBRAL: {threshold}")
                print(f"TPR:{tpr} TNR:{tnr} FPR:{fpr} FNR:{fnr}")
        # Calculamos las medias finales para cada umbral
        mean_tpr_list = [np.mean(all_tpr[thr]) for thr in threshold_list]
        mean_tnr_list = [np.mean(all_tnr[thr]) for thr in threshold_list]
        mean_fpr_list = [np.mean(all_fpr[thr]) for thr in threshold_list]
        mean_fnr_list = [np.mean(all_fnr[thr]) for thr in threshold_list]

        # Graficamos los resultados
        self.graficar_resultados(threshold_list, mean_tpr_list, mean_tnr_list, mean_fpr_list, mean_fnr_list)


    # Itera para llamar a entrenar_autoencoder en todos el grafo y obtiene los resultados despues
    def bucle_entrenar_y_resultados(self, grafos):
        # Creamos el Autocodificador general
        autoencoder, encoder = crear_autoencoder(input_dim=17955, entrada = 1)
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

        #Retornamos el porcentaje medio
        porcentaje_acierto_medio = sum(lista_porcentajes) / len(lista_porcentajes)
        return porcentaje_acierto_medio


    # Una vez esta la red neuronal entrenada, esta funcion
    # Autoencoder: La reconstruccion
    # Encoder: El cuello de botella
    def obtener_resultados(self,grafo, autoencoder, encoder):
        grafo = np.expand_dims(grafo, axis=0)  # Convertimos de 1D a 2D
        reconstruccion = autoencoder.predict(grafo, verbose=0)
        cuello_botella = encoder.predict(grafo, verbose=0)
        return reconstruccion, cuello_botella

    def tests_matriz(self, grafos, cuello_botellaE):
        # Usamos el train split de sklearn para dividir de manera aleatoria
        contador = 0
        lista_porcentajes = []
        grafos_entrenamiento, grafos_test = train_test_split(grafos, test_size=0.2, random_state=42)
        # Creamos el autoencoder y lo entrenamos con el entrenamiento
        autoencoder, encoder = crear_autoencoder(input_dim=17955, entrada=cuello_botellaE)
        # Compilamos y aprendemos
        learning_rate = 0.0005  # Puedes probar con diferentes valores como 0.001, 0.0001, etc.

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mse')
        autoencoder.fit(grafos_entrenamiento, grafos_entrenamiento, epochs=20, batch_size=64)

        for grafo in grafos_test:
            reconstruccion, cuello_botella = self.obtener_resultados(grafo, autoencoder, encoder)
            reconstruccion_mod = reconstruccion[0]
            contador += 1

            self.matriz_confusion(grafo, reconstruccion_mod)

    def aplicar_ruido(self,matriz, p):
        for i in range(len(matriz)):
            if np.random.rand() < p:
                matriz[i] = 1 - matriz[i]
        return matriz


    def matriz_confusion(self, grafo, grafo_reconstruido):
        # Inicializar matriz de confusión
        tn = 0  # 0s reconstruidos donde había 0s
        fn = 0  # 0s reconstruidos donde había 1s
        fp = 0  # 1s reconstruidos donde había 0s
        tp = 0  # 1s reconstruidos donde había 1s
        error = 0

        for i in range(len(grafo)):
            if grafo[i] == 0:
                if grafo_reconstruido[i] == 0:
                    tn += 1
                elif grafo_reconstruido[i] == 1:
                    fp += 1
            elif grafo[i] == 1:
                if grafo_reconstruido[i] == 0:
                    fn += 1
                elif grafo_reconstruido[i] == 1:
                    tp += 1
            else:
                error += 1

        # Matriz de confusión en valores absolutos
        matriz_confusion = np.array([[tn, fp], [fn, tp]])
        # Cálculo de tasas (evitar divisiones por cero)
        tpr = (tp / (tp + fn)) * 100
        tnr = (tn / (tn + fp)) * 100
        fpr = (fp / (fp + tn)) * 100
        fnr = (fn / (fn + tp)) * 100

        # Mostrar resultados (COMENTADO PARA EL UMBRAL !DESCOMENTAR!
        ##print(f"\n(TPR): {tpr:.4f}%, (TNR): {tnr:.4f}%, (FPR): {fpr:.4f}%, (FNR): {fnr:.4f}%")
        return tpr, tnr, fpr, fnr

    def graficar_resultados(self, threshold_list, tpr, tnr, fpr, fnr):
        """
        Grafica la evolución de una única métrica (TPR, TNR, FPR o FNR) en función del umbral.

        Parámetros:
            - threshold_list: Lista de umbrales utilizados.
            - tasa_list: Lista con los valores de la métrica.
            - nombre_tasa: Nombre de la métrica a graficar (ej: "TPR", "TNR").
        """
        plt.figure(figsize=(8, 5))
        plt.plot(threshold_list, tpr,marker="o", linestyle="-", label="TPR", color="blue")
        plt.plot(threshold_list, tnr, marker="o", linestyle="-", label="TNR", color="green")
        plt.plot(threshold_list, fpr, marker="o", linestyle="-", label="FPR", color="red")
        plt.plot(threshold_list, fnr, marker="o", linestyle="-", label="FNR", color="brown")


        # Configuración del gráfico
        plt.xlabel("Threshold")
        plt.ylabel("Rate")
        plt.legend()
        plt.grid(True)
        plt.xscale("log")  # Escala logarítmica en el eje X para mejor visualización
        plt.show()


class Run:
    @staticmethod
    def ejecutar(ejecutar_pca, ejecutar_entrenamiento, ejecutar_tests, ejecutar_tests_autoencoder, ejecutar_tests_stacked_autoencoder):
        main = Main()

        if ejecutar_pca:
            print("🔵 Ejecutando PCA...")
            for n in range(2, 50):
                aplicar_pca_y_graficar(main.grafos_reducidos_ADHD, "ADHD", n)
                aplicar_pca_y_graficar(main.grafos_reducidos_TD, "TD", n)
                aplicar_pca_y_graficar(main.grafos_reducidos_Combinados, "ADHD y TD Combinados", n)

        if ejecutar_entrenamiento:
            print("🟢 Ejecutando entrenamiento del autoencoder...")
            porcentaje_medio_adhd = main.bucle_entrenar_y_resultados(main.grafos_reducidos_ADHD)
            porcentaje_medio_td = main.bucle_entrenar_y_resultados(main.grafos_reducidos_TD)
            porcentaje_medio_combinados = main.bucle_entrenar_y_resultados(main.grafos_reducidos_Combinados)
            print(
                f"Resultados Entrenamiento -> ADHD: {porcentaje_medio_adhd}, TD: {porcentaje_medio_td}, Combinados: {porcentaje_medio_combinados}")

        if ejecutar_tests:
            print("🟠 Ejecutando tests...")
            acierto_test_adhd = main.tests(main.grafos_reducidos_ADHD)
            acierto_test_td = main.tests(main.grafos_reducidos_TD)
            acierto_test_Combinados = main.tests(main.grafos_reducidos_Combinados)
            print(
                f"Resultados Tests -> ADHD: {acierto_test_adhd}, TD: {acierto_test_td}, Combinados: {acierto_test_Combinados}")

        if ejecutar_tests_autoencoder:
            print("🟠 Ejecutando matriz confusion...")
            for i in range(20,40):
                main.tests_matriz(main.grafos_reducidos_ADHD, i)
            main.tests_matriz(main.grafos_reducidos_TD, 1)
            main.tests_matriz(main.grafos_reducidos_Combinados, 1)

        if ejecutar_tests_stacked_autoencoder:
            print("🟠 Ejecutando matriz confusion...")
            main.test_stacked_autoencoders(main.grafos_reducidos_Combinados)





