import torch
import os
import argparse
import time
import manejo_de_datos
import torch.nn as nn
import numpy as np
import pandas as pd


from sklearn.metrics import confusion_matrix
from modelo import MLP


def main():
    t_inicio = time.time()
    print(f'Corriendo con: {"cuda" if torch.cuda.is_available() else "cpu"}')
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Manejo de argumentos
    args       = handle_args()
    modelo_fp  = args['modelo']
    perdida_m  = args['modelo_perdida']
    num_clases = args['clases']

    # Cargador de datos
    print('Creando cargador de datos...')
    c_evaluacion, tam_ventana = manejo_de_datos.crea_cargador_evaluacion('../datos/evaluacion.csv')
    t_usado = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_inicio))

    # Cargado del modelo
    id_modelo  = os.path.basename(modelo_fp)
    modelo = MLP(tam_ventana)
    modelo = modelo.to(dispositivo)

    # Graficar pérdida a través del tiempo
    if perdida_m:
        graficar_perdida(perdida_m, f'{id_modelo}_perdida.png', dispositivo)

    # Evaluar
    manejo_de_datos.carga_modelo(modelo_fp, modelo, dispositivo)
    print(f'Evaluando... ({modelo_fp})')
    evaluar(modelo, c_evaluacion, dispositivo, num_clases)
    t_usado = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_inicio))
    print(f'Evaluación terminada, tiempo usado = {t_usado}.')


def evaluar(modelo, i_evaluacion, dispositivo, num_clases):
    y_pre = []
    y_ver = []
    normalizador = nn.Softmax(dim=1)

    modelo.eval()
    with torch.no_grad():
        for (snps, etiquetas) in i_evaluacion:
            etiquetas = etiquetas.to(dispositivo)
            snps      = snps.to(dispositivo)

            prediccion = modelo(snps)
            prediccion = normalizador(prediccion)
            prediccion = np.argmax(prediccion.data.cpu(), axis=1)

            y_pre.extend(prediccion.tolist())
            y_ver.extend(etiquetas.tolist())

    # Graficar matriz de confusión
    clases      = [f'P{x}' for x in range(num_clases)]
    matriz_conf = confusion_matrix(y_ver, y_pre).astype('float')
    matriz_conf = matriz_conf / matriz_conf.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    print(matriz_conf)


def handle_args():
  parser = argparse.ArgumentParser(description='Evaluación de modelo.')
  parser.add_argument('-m','--modelo', required=True,
    help='Modelo para evaluar', type=str)
  parser.add_argument('-p','--modelo-perdida', required=False,
    help='Modelo del cual graficar pérdida', default='', type=str)
  parser.add_argument('-c','--clases',
    help='Número de clases a predecir', required=True, type=int)

  return vars(parser.parse_args())


if __name__ == '__main__':
  main()
