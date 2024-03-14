import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def divide_entrenamiento_validado(fp, prop_validado=0.10):
    '''Divide los datos simulados en dos conjuntos: entrenamiento y validación.'''
    d_crudos = pd.read_csv(fp, header=None).to_numpy(dtype=np.float32)

    prop_entrenamiento = 1.0 - prop_validado
    d_entrenamiento, d_validado = train_test_split(d_crudos, train_size=prop_entrenamiento)

    return d_entrenamiento, d_validado


class DatosSNP(Dataset):
    '''
    Clase que representa un conjunto de datos. Por ejemplo datos de entrenamiento, datos de
    validación, o datos de evaluación.
    '''
    def __init__(self, d_snp, transform=None, target_transform=None):
        self.d_snp = d_snp
        self.l_ventana = len(self.d_snp[0]) - 1
        self.longitud  = len(self.d_snp)

        self.transform        = transform
        self.target_transform = target_transform


    def __len__(self):
        return self.longitud


    def __getitem__(self, indice):
        muestra = self.d_snp[indice]
        snps = muestra[1:]
        etiqueta = int(muestra[0])

        return snps, etiqueta


def crea_cargadores_entrenamiento(datos_fp, prop_validado, tam_lote=32):
    datos_entrenamiento, datos_validado = divide_entrenamiento_validado(datos_fp, prop_validado)

    snps_entrenamiento = DatosSNP(datos_entrenamiento)
    snps_validado      = DatosSNP(datos_validado)

    c_entrenamiento = DataLoader(snps_entrenamiento, batch_size=tam_lote, drop_last=True, shuffle=True)
    c_validado = DataLoader(snps_validado, batch_size=tam_lote, drop_last=True, shuffle=True)

    return c_entrenamiento, c_validado, snps_entrenamiento.l_ventana


def crea_cargador_evaluacion(datos_fp, tam_lote=32):
    datos_evaluacion = pd.read_csv(datos_fp, header=None).to_numpy(dtype=np.float32)
    snps_evaluacion = DatosSNP(datos_evaluacion)
    c_evaluacion = DataLoader(snps_evaluacion, batch_size=tam_lote, drop_last=True, shuffle=True)

    return c_evaluacion, snps_evaluacion.l_ventana


def guarda_modelo(fp, modelo, optimizador, perdida_validado):
    guardado = {'model_state_dict': modelo.state_dict(),
        'optimizer_state_dict': optimizador.state_dict(), 'valid_loss': perdida_validado}
    torch.save(guardado, fp)


def carga_modelo(fp, modelo, dispositivo, optimizador=None):
    guardado = torch.load(fp, map_location=dispositivo)
    modelo.load_state_dict(guardado['model_state_dict'])

    if optimizador:
        optimizador.load_state_dict(guardado['optimizer_state_dict'])

    return guardado['valid_loss']


def guarda_metricas(fp, lista_perdida_entrenamiento, lista_perdida_validado, lista_pasos_globales):
    guardado = {'train_loss_list': lista_perdida_entrenamiento,
        'valid_loss_list': lista_perdida_validado, 'global_steps_list': lista_pasos_globales}
    torch.save(guardado, fp)


def carga_metricas(fp, dispositivo):
    guardado = torch.load(fp, map_location=dispositivo)

    return (guardado['train_loss_list'], guardado['valid_loss_list'], guardado['global_steps_list'])
