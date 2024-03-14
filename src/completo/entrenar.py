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

    # Identificador del modelo que entrenamos y creación del modelo
    id_modelo = f'MLP_{tasa_apr:.0E}'
    modelo = MLP(tam_ventana, num_clases)

    # Optimizador del modelo (Algoritmo de retropropagación)
    modelo = modelo.to(dispositivo)
    optimizador = optim.AdamW(modelo.parameters(), lr=tasa_apr)

    # Bucle de entrenamiento
    print(f'Entrenando... ({id_modelo})')
    entrena(modelo, optimizador, c_entrenamiento, c_validado, len(c_entrenamiento) // 2,
        dispositivo, tam_ventana, epocas, t_inicio=t_inicio, id_modelo=id_modelo)


def entrena(modelo, optimizador, i_entrenamiento, i_validado, paso_eval, dispositivo,
            tam_ventana, epocas=20, criterio=nn.CrossEntropyLoss(),
            mejor_perdida_validado=float('Inf'), t_inicio=time.time(), id_modelo='model'):
    perdida_global    = 0.0
    perdida_validado  = 0.0
    paso_global       = 0
    perdidas          = []
    perdidas_validado = []
    pasos_globales    = []

    modelo.train()
    for epoca in range(epocas):
        for (snps, etiquetas) in i_entrenamiento:
            snps      = snps.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            optimizador.zero_grad()
            prediccion = modelo(snps)
            perdida    = criterio(prediccion, etiquetas)
            perdida.backward()
            optimizador.step()

            perdida_global += perdida.item()
            paso_global    += 1

            # Validación
            if paso_global % paso_eval == 0:
                modelo.eval()
                with torch.no_grad():
                    # Recorrer datos de validación
                    for (snps, etiquetas) in i_validado:
                        etiquetas = etiquetas.to(dispositivo)
                        snps      = snps.to(dispositivo)

                        prediccion = modelo(snps)
                        perdida    = criterio(prediccion, etiquetas)
                        perdida_validado += perdida.item()

                    perdida_promedio = perdida_global / paso_eval
                    perdida_validado_promedio = perdida_validado / len(i_validado)
                    perdidas.append(perdida_promedio)
                    perdidas_validado.append(perdida_validado_promedio)
                    pasos_globales.append(paso_global)

                    perdida_global    = 0.0
                    perdida_validado = 0.0
                    modelo.train()

                    t_usado = time.strftime('%H:%M:%S',
                    time.gmtime(time.time() - t_inicio))
                    print((f'epoca [{epoca + 1}/{epocas}],\t'
                    f'paso [{paso_global}/{epocas * len(i_entrenamiento)}],\t'
                    f'pérdida de entrenamiento = {perdida_promedio:.3f},\t'
                    f'pérdida de validación = {perdida_validado_promedio:.3f},\t'
                    f'tiempo usado = {t_usado}'))

                    # Guardar siempre el mejor modelo
                    if perdida_validado_promedio < mejor_perdida_validado:
                        mejor_perdida_validado = perdida_validado_promedio
                        manejo_de_datos.guarda_modelo(f'../datos/{id_modelo}_modelo.pt',
                            modelo, optimizador, mejor_perdida_validado)
                        manejo_de_datos.guarda_metricas(f'../datos/{id_modelo}_metricas.pt',
                            perdidas, perdidas_validado, pasos_globales)


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
