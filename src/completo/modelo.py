import torch
import math
import torch.nn as nn


#########################
# Perceptrón Multi-Capa #
#########################


class MLP(nn.Module):
    '''
    Clase que representa un Perceptrón Multi-Capa (MLP).
    '''


    def __init__(self, longitud, num_clases=3):
        '''
        Inicialización del MLP. Se encarga de construir la estructura de la red, inicialmente con
        pesos y sesgos aleatorios.

        Parámetros:
            longitud - Longitud en términos de SNPs de la región asociada a este modelo.
            num_clases - Número de clases que el modelo puede predecir (i. e. cuántas poblaciones tenemos).
        '''
        super(MLP, self).__init__()

        self.num_clases = num_clases

        self.capas = nn.Sequential(
        nn.Linear(longitud, longitud), nn.ReLU(),
        nn.Linear(longitud, longitud), nn.ReLU(),
        nn.Linear(longitud, self.num_clases))


    def forward(self, snp):
        '''
        Función de propagación. Alimenta una matrix de SNPs a la capa de entrada, y recopila los resultados
        de la capa de salida.

        Parámetros:
            snp - La matrix de SNPs que se analizará.

        Valor de salida:
            Vector de probabilidades, una probabilidad para cada clase.
        '''
        predicciones = self.capas(snp)

        return predicciones
