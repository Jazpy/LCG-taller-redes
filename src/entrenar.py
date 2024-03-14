import torch
import argparse
import time
import os
import manejo_de_datos
import torch.nn as nn
import torch.optim as optim


from modelo import MLP


def main():
    t_inicio = time.time()
    print(f'Corriendo con: {"cuda" if torch.cuda.is_available() else "cpu"}')
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Manejo de argumentos
    args       = lee_argumentos()
    p_validado = args['validado']
    num_clases = args['clases']
    tam_lote   = args['lote']
    tasa_apr   = args['aprendizaje']
    epocas     = args['epocas']

    # Cargadores de datos
    print('Creando cargadores de datos...')
    c_entrenamiento, c_validado, tam_ventana = manejo_de_datos.crea_cargadores_entrenamiento(
        '../datos/entrenamiento.csv', p_validado, tam_lote=tam_lote)


def entrena(modelo, optimizador, i_entrenamiento, i_validado, paso_eval, dispositivo,
            tam_ventana, epocas=20, criterio=nn.CrossEntropyLoss(),
            mejor_perdida_validado=float('Inf'), t_inicio=time.time(), id_modelo='model'):


def lee_argumentos():
  parser = argparse.ArgumentParser(description='Entrenamiento de modelo.')
  parser.add_argument('-v','--validado',
    help='Porcentaje de validación', default=0.10, type=float)
  parser.add_argument('-c','--clases',
    help='Número de clases a considerar', required=True, type=int)
  parser.add_argument('-l','--lote',
    help='Tamaño de lote', default=32, type=int)
  parser.add_argument('-e','--epocas',
    help='Número de épocas de entrenamiento', default=10, type=int)
  parser.add_argument('-a','--aprendizaje',
    help='Tasa de aprendizaje', default=1e-3, type=float)

  return vars(parser.parse_args())


if __name__ == '__main__':
  main()
