import ADHD
import argparse
from ADHD import Run



def parse_args():
    parser = argparse.ArgumentParser(description="Script de ejecución para el TFG.")

    # Argumentos para controlar qué se ejecuta
    parser.add_argument('--PCA', action='store_true', help='Ejecutar el análisis PCA')
    parser.add_argument('--tests', action='store_true', help='Ejecutar los tests')
    parser.add_argument('--entrenamiento', action='store_true', help='Ejecutar el entrenamiento del autoencoder')
    parser.add_argument('--matrizConfusion', action='store_true', help='Calcular la matriz de confusion para cada grafo')

    # Aquí le pasas las opciones desde la terminal
    return parser.parse_args()

if __name__ == "__main__":
    # Valores por defecto para cuando se ejecuta desde el compilador
    ejecutar_pca = False
    ejecutar_entrenamiento = False
    ejecutar_tests = False
    ejecutar_matriz_confusion = True

    # Procesar los argumentos de la terminal
    args = parse_args()

    # Sobrescribir los valores predeterminados con los argumentos de la terminal, si es que se pasan
    if args.PCA:
        ejecutar_pca = True
    if args.tests:
        ejecutar_tests = True
    if args.entrenamiento:
        ejecutar_entrenamiento = True
    if args.matrizConfusion:
        ejecutar_matriz_confusion = True

    # Ejecutar según los valores de las variables
    Run.ejecutar(ejecutar_pca, ejecutar_entrenamiento, ejecutar_tests, ejecutar_matriz_confusion)