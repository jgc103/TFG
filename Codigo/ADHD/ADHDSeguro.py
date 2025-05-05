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
import seaborn as sns  # Aquí importamos seaborn para el heatmap


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
    plt.figure(figsize=(10, 7))
    plt.scatter(matriz_reducida[:, 0], matriz_reducida[:, 1], c='b', marker='o')

    # Títulos y etiquetas de los ejes
    plt.title(f'PCA de {titulo}')
    plt.xlabel(f'Numero de Componentes: {n_componentes}')
    plt.grid()
    plt.show()

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
        self.threshold_list = None
        self.threshold_list_no_denoising = None


        # Modelamos los datos para que tengan la estructura necesaria de la PCA
        # ADHD
        self.grafos_reducidos_ADHD = reduccion_dimensionalidad(self.directorio_ADHD)
        # TD
        self.grafos_reducidos_TD = reduccion_dimensionalidad(self.directorio_TD)
        # ADHD & TD
        self.grafos_reducidos_Combinados = np.vstack((self.grafos_reducidos_ADHD, self.grafos_reducidos_TD))
        #Threshold list
        # Lista de umbrales
        self.threshold_list = np.linspace(0.1, 0.5, num=30)

        self.threshold_list_no_denoising = np.concatenate([
            np.geomspace(0.0001, 0.1, num=25, endpoint=False),
            np.linspace(0.1, 0.5, num=5)
        ])


    """Divide los datos de entrenamiento y test"""
    def tests(self, grafos):
        # Usamos el train split de sklearn para dividir de manera aleatoria
        contador = 0
        lista_porcentajes = []
        grafos_entrenamiento, grafos_test = train_test_split(grafos, test_size=0.2, random_state=42)
        # Creamos el autoencoder y lo entrenamos con el entrenamiento
        autoencoder, encoder = crear_autoencoder(input_dim=17955, entrada=10)
        # Compilamos y aprendemos
        learning_rate = 0.0005
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
    def test_stacked_autoencoders(self, grafos, booleano_denoising, noise, mapa_calor):

        grafos_entrenamiento, grafos_test = train_test_split(grafos, test_size=0.2, random_state=42)

        autoencoders, encoders = crear_stacked_autoencoder(input_dim=17955, latent_dim_final=10)
        if booleano_denoising:
            threshold_list = self.threshold_list
        else:
            threshold_list = self.threshold_list_no_denoising

        # Entrenamiento del autoencoder
        if booleano_denoising:
            noisy_data = np.array([self.aplicar_ruido(g, noise) for g in grafos_entrenamiento])
        else:
            noisy_data = np.array(grafos_entrenamiento)

        original_data = np.array(grafos_entrenamiento)

        for i, autoencoder in enumerate(autoencoders):
            print(f"Entrenando nivel {i + 1} del stacked autoencoder")
            optimizer = keras.optimizers.Adam(learning_rate=0.0005)
            autoencoder.compile(optimizer=optimizer, loss='mse')
            autoencoder.fit(noisy_data, original_data, epochs=30, batch_size=128)

            if i < len(encoders) - 1:
                noisy_data = encoders[i].predict(noisy_data)
                original_data = encoders[i].predict(original_data)

        if mapa_calor:
            threshold_list = [0.145]
        # Diccionarios para almacenar los valores de cada métrica para cada umbral
        all_tpr = {thr: [] for thr in threshold_list}
        all_tnr = {thr: [] for thr in threshold_list}
        all_fpr = {thr: [] for thr in threshold_list}
        all_fnr = {thr: [] for thr in threshold_list}

        # Creamos las listas para evaluar su mapa de calor
        binarios_originales = []
        binarios_reconstruidos = []


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


                # Llamamos a la funcion de analizar los porcentajes por arista
                if mapa_calor:
                    binarios_originales.append(grafo)
                    binarios_reconstruidos.append(reconstruccion_mod)

                # Guardamos los valores en cada umbral
                all_tpr[threshold].append(tpr)
                all_tnr[threshold].append(tnr)
                all_fpr[threshold].append(fpr)
                all_fnr[threshold].append(fnr)
                #print(f"UMBRAL: {threshold}")
                #print(f"TPR:{tpr} TNR:{tnr} FPR:{fpr} FNR:{fnr}")

        if mapa_calor:
            self.contar_aciertos_y_porcentaje_por_arista(
                binarios_originales, binarios_reconstruidos,num_nodos=190)

        # Calculamos las medias finales para cada umbral
        mean_tpr_list = [np.mean(all_tpr[thr]) for thr in threshold_list]
        mean_tnr_list = [np.mean(all_tnr[thr]) for thr in threshold_list]
        mean_fpr_list = [np.mean(all_fpr[thr]) for thr in threshold_list]
        mean_fnr_list = [np.mean(all_fnr[thr]) for thr in threshold_list]

        return {
            "TPR": mean_tpr_list,
            "TNR": mean_tnr_list,
            "FPR": mean_fpr_list,
            "FNR": mean_fnr_list,
        }


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


    ############################
    ####FUNCIONES AUXILIARES####
    ############################

    # Una vez esta la red neuronal entrenada, esta funcion
    # Autoencoder: La reconstruccion
    # Encoder: El cuello de botella
    def obtener_resultados(self, grafo, autoencoder, encoder):
        grafo = np.expand_dims(grafo, axis=0)  # Convertimos de 1D a 2D
        reconstruccion = autoencoder.predict(grafo, verbose=0)
        cuello_botella = encoder.predict(grafo, verbose=0)
        return reconstruccion, cuello_botella

    def contar_aciertos_y_porcentaje_por_arista(self, grafos_originales, grafos_reconstruidos, num_nodos):
        """
        Cuenta, para cada arista, los aciertos, presencia, TP, FN, FP, TN,
        y calcula los porcentajes respectivos.
        """

        grafos_originales = np.array(grafos_originales)
        grafos_reconstruidos = np.array(grafos_reconstruidos)

        grafos_comprobar_originales = grafos_reconstruidos
        # Convertimos cada vector a una tupla para poder usar sets
        grafos_como_tuplas = [tuple(grafo) for grafo in grafos_comprobar_originales]

        # Creamos un conjunto con las tuplas (los sets eliminan duplicados)
        conjunto_unicos = set(grafos_como_tuplas)

        # Mostramos cuántos son únicos
        print(f"Total de grafos únicos reconstruidos: {len(conjunto_unicos)} de {len(grafos_comprobar_originales)}")

        assert grafos_originales.shape == grafos_reconstruidos.shape, "Dimensiones incompatibles"

        total_grafos = grafos_originales.shape[0]
        num_aristas = grafos_originales.shape[1]
        # Inicialización de contadores
        aciertos_por_arista = np.zeros(num_aristas, dtype=int)
        verdaderos_positivos = np.zeros(num_aristas, dtype=int)
        falsos_negativos = np.zeros(num_aristas, dtype=int)
        falsos_positivos = np.zeros(num_aristas, dtype=int)
        verdaderos_negativos = np.zeros(num_aristas, dtype=int)
        total_positivos = np.zeros(num_aristas, dtype=int)
        total_negativos = np.zeros(num_aristas, dtype=int)

        # Recorrido por cada grafo y arista
        for g in range(total_grafos):
            for i in range(num_aristas):
                real = grafos_originales[g, i]
                pred = grafos_reconstruidos[g, i]

                if real == pred:
                    aciertos_por_arista[i] += 1

                if real == 1:
                    total_positivos[i] += 1
                    if pred == 1:
                        verdaderos_positivos[i] += 1
                    else:
                        falsos_negativos[i] += 1
                else:
                    total_negativos[i] += 1
                    if pred == 1:
                        falsos_positivos[i] += 1
                    else:
                        verdaderos_negativos[i] += 1

        # Porcentajes
        porcentaje_por_arista = (aciertos_por_arista / total_grafos) * 100
        presencia_real = (total_positivos / total_grafos) * 100

        with np.errstate(divide='ignore', invalid='ignore'):
            porcentaje_tp_sobre_1s = np.where(total_positivos != 0,
                                              (verdaderos_positivos / total_positivos) * 100,
                                              0)
            porcentaje_fn_sobre_1s = np.where(total_positivos != 0,
                                              (falsos_negativos / total_positivos) * 100,
                                              0)
            porcentaje_fp_sobre_0s = np.where(total_negativos != 0,
                                              (falsos_positivos / total_negativos) * 100,
                                              0)
            porcentaje_tn_sobre_0s = np.where(total_negativos != 0,
                                              (verdaderos_negativos / total_negativos) * 100,
                                              0)

        # Concatenar resultados
        todo = np.vstack((
            aciertos_por_arista,
            porcentaje_por_arista,
            presencia_real,
            porcentaje_tp_sobre_1s,
            porcentaje_fn_sobre_1s,
            porcentaje_fp_sobre_0s,
            porcentaje_tn_sobre_0s
        )).T

        # Cabecera alineada
        header = (
            "NumAciertos     %Acierto     %Presencia     %TP_sobre_1s     %FN_sobre_1s"
            "     %FP_sobre_0s     %TN_sobre_0s"
        )

        # Guardar archivo con formato alineado
        np.savetxt("porcentaje_y_presencia.txt", todo,
                   fmt="%13.0f%13.2f%14.2f%17.2f%17.2f%17.2f%17.2f",
                   header=header,
                   comments='')

        # Guardar aciertos absolutos si lo necesitas
        np.savetxt("aciertos.txt", aciertos_por_arista, fmt="%d")

        # Mostrar heatmap con %TP sobre 1s
        self.mostrar_heatmap(porcentaje_tp_sobre_1s / 100, num_nodos)

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


    #########################
    #######GRAFICAS##########
    #########################

    def mostrar_heatmap(self, vector_valores, num_nodos, titulo="Porcentaje de aciertos por arista"):
        """
        Reconstruye matriz simétrica desde vector (triángulo superior) y guarda un heatmap,
        pero solo muestra la parte del triángulo superior.
        """
        # Reconstruir la matriz completa a partir del vector (triángulo superior)
        matriz = np.zeros((num_nodos, num_nodos))
        idx = 0
        for i in range(num_nodos):
            for j in range(i + 1, num_nodos):
                matriz[i, j] = vector_valores[idx]
                matriz[j, i] = vector_valores[idx]
                idx += 1

        # Creamos una máscara para ocultar la parte inferior de la matriz (triángulo inferior)
        mask = np.tril(np.ones_like(matriz, dtype=bool))

        # Mostrar el heatmap solo para el triángulo superior (sin la parte inferior)
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz, mask=mask, cmap="viridis", square=True, cbar_kws={'label': 'Porcentaje de aciertos'})
        plt.title(titulo)
        plt.xlabel("Nodo")
        plt.ylabel("Nodo")
        plt.tight_layout()
        plt.savefig("heatmap_porcentaje_aciertos.png")
        plt.close()

    def graficar_resultados(self, threshold_list, resultados_con_ruido):
        """
        Grafica la evolución de una única métrica (TPR, TNR, FPR o FNR) en función del umbral.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_list, resultados_con_ruido["TPR"],marker="o", linestyle="-", label="TPR", color="blue")
        plt.plot(threshold_list, resultados_con_ruido["FPR"], marker="o", linestyle="-", label="FPR", color="red")


        # Configuración del gráfico
        plt.xlabel("Threshold")
        plt.ylabel("Rate")
        plt.title("Comparativa métricas con y sin denoising")
        plt.legend()
        plt.grid(True)
        plt.xscale("linear")  # Cambia a log si quieres
        plt.tight_layout()
        plt.show()

    def graficar_resultados_comparativa(self, threshold_list, resultados_con_ruido, resultados_sin_ruido):
        """
        Compara las métricas de rendimiento entre dos configuraciones (con y sin ruido).
        """
        plt.figure(figsize=(10, 6))
        # Con ruido (azul)
        plt.plot(threshold_list, resultados_con_ruido["TPR"], marker="o", linestyle="-", label="TPR con ruido",
                 color="blue")
        plt.plot(threshold_list, resultados_con_ruido["FPR"], marker="o", linestyle="-", label="FPR con ruido",
                 color="navy")

        # Sin ruido (verde)
        plt.plot(threshold_list, resultados_sin_ruido["TPR"], marker="s", linestyle="--", label="TPR sin ruido",
                 color="green")
        plt.plot(threshold_list, resultados_sin_ruido["FPR"], marker="s", linestyle="--", label="FPR sin ruido",
                 color="darkgreen")

        plt.xlabel("Threshold")
        plt.ylabel("Rate")
        plt.title("Comparativa métricas con y sin denoising")
        plt.legend()
        plt.grid(True)
        plt.xscale("linear")  # Cambia a log si quieres
        plt.tight_layout()
        plt.show()

    def graficar_curvas_roc_comparativa(self, threshold_list, resultados_con_ruido, resultados_sin_ruido):
        """
        Grafica la comparación entre dos curvas ROC: una con ruido y otra sin ruido.
        """
        fpr_ruido = resultados_con_ruido["FPR"]
        tpr_ruido = resultados_con_ruido["TPR"]

        fpr_sin = resultados_sin_ruido["FPR"]
        tpr_sin = resultados_sin_ruido["TPR"]

        plt.figure(figsize=(7, 7))

        # Curva ROC con ruido
        plt.plot(fpr_ruido, tpr_ruido, marker="o", linestyle="-", color="blue", label="ROC con ruido")
        for i in range(len(threshold_list)):
            plt.text(fpr_ruido[i], tpr_ruido[i], f'{threshold_list[i]:.2f}', fontsize=8, color='blue', ha='right')

        # Curva ROC sin ruido
        plt.plot(fpr_sin, tpr_sin, marker="s", linestyle="--", color="green", label="ROC sin ruido")
        for i in range(len(threshold_list)):
            plt.text(fpr_sin[i], tpr_sin[i], f'{threshold_list[i]:.2f}', fontsize=8, color='green', ha='right')

        # Línea aleatoria
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio")

        # Configuración del gráfico
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("Comparativa de Curvas ROC (con vs sin ruido)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class Run:
    @staticmethod
    def ejecutar(ejecutar_pca, ejecutar_entrenamiento, ejecutar_tests, ejecutar_tests_autoencoder, ejecutar_tests_stacked_autoencoder,
                 ejecutar_tests_stacked_autoencoder_heatmap):
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

            res_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, True, 0.01, False)
            res_no_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, False, 0.01, False)
            main.graficar_curvas_roc_comparativa(main.threshold_list, res_denoising, res_no_denoising)

            res_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, True, 0.05, False)
            res_no_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, False, 0.05, False)
            main.graficar_curvas_roc_comparativa(main.threshold_list, res_denoising, res_no_denoising)
            """
            res_no_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, False, 0.1)
            res_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, True, 0.1)
            main.graficar_curvas_roc_comparativa(main.threshold_list, res_denoising, res_no_denoising)

            res_no_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, False, 0.15)
            res_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, True, 0.15)
            main.graficar_curvas_roc_comparativa(main.threshold_list, res_denoising, res_no_denoising)

            res_no_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, False, 0.2)
            res_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, True, 0.2)
            main.graficar_curvas_roc_comparativa(main.threshold_list, res_denoising, res_no_denoising)
            """
        if ejecutar_tests_stacked_autoencoder_heatmap:
            print("🟠 Ejecutando mapa de calor...")
            res_denoising = main.test_stacked_autoencoders(main.grafos_reducidos_Combinados, True, 0.1, True)
            print(res_denoising)











