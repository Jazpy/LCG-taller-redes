import msprime


##########################
# SIMULACION CON MSPRIME #
##########################


def divergencia_simple(n, l):
    '''
    Simulación de 600 muestras diploides de cada población, se consideran tres poblaciones "A" , "B" y "C" tal que:
        - Tienen N_e = 10,000 en el presente.
        - "A" y "B"  Se vuelven una sola población "AB"  1,000 generaciones en el pasado.
        - "C" y "AB" Se vuelven una sola población "ABC" 4,000 generaciones en el pasado.

    Parámetros:
        n - Número de muestras a simular (por población).
        l - Longitud de la región a simular.

    Valor de salida:
        Secuencia de árboles.

    Documentación relevante:
        msprime.Demography
        msprime.SampleSet
        msprime.sim_ancestry()
    '''

    # Especificación de poblaciones
    modelo_dem = msprime.Demography()
    modelo_dem.add_population(name='A',   initial_size=10_000)
    modelo_dem.add_population(name='B',   initial_size=10_000)
    modelo_dem.add_population(name='C',   initial_size=10_000)
    modelo_dem.add_population(name='AB',  initial_size=10_000)
    modelo_dem.add_population(name='ABC', initial_size=10_000)

    # Especificación de eventos demográficos
    modelo_dem.add_population_split(time=1_000, derived=['A', 'B'],  ancestral='AB')
    modelo_dem.add_population_split(time=4_000, derived=['C', 'AB'], ancestral='ABC')

    # Especificación de muestras a simular
    muestras_a = msprime.SampleSet(n, population='A', time=0, ploidy=2)
    muestras_b = msprime.SampleSet(n, population='B', time=0, ploidy=2)
    muestras_c = msprime.SampleSet(n, population='C', time=0, ploidy=2)

    arboles = msprime.sim_ancestry(
        samples=[muestras_a, muestras_b, muestras_c],
        demography=modelo_dem,
        recombination_rate=2e-8,
        sequence_length=l)

    return arboles


def mutacion_simple(arboles):
    '''
    Simulación de mutaciones dada una secuencia de árboles de msprime.

    Parámetros:
        arboles - Secuencia de árboles generada a través de msprime.sim_ancestry()

    Valor de salida:
        Secuencia de árboles (con mutaciones).

    Documentación relevante:
        msprime.sim_mutations()
    '''

    arboles_mutados = msprime.sim_mutations(arboles, rate=1.5e-8)

    return arboles_mutados


def guarda_genotipos(matriz, poblaciones, muestras_entrenamiento, muestras_eval, snps):
    '''
    Guarda la matriz de genotipos simulada en archivos de entrenamiento y evaoluación.

    Parámetros:
        matriz - La matriz de genotipos.
        poblaciones - Número de poblaciones en el modelo demográfico.
        muestras_entrenamiento - Número de muestras a usar para entrenar.
        muestras_eval - Número de muestras a usar para evaluar el modelo.
        snps - Número de SNPs a usar para la región a estudiar.
    '''

    with open('../datos/entrenamiento.csv', 'w') as en_f, open('../datos/evaluacion.csv', 'w') as ev_f:
        indice = 0
        for p in range(poblaciones):
            for _ in range(muestras_entrenamiento * 2):
                renglon = [p] + list(matriz[:snps, indice])
                en_f.write(','.join([str(x) for x in renglon]) + '\n')
                indice += 1
            for _ in range(muestras_eval * 2):
                renglon = [p] + list(matriz[:snps, indice])
                ev_f.write(','.join([str(x) for x in renglon]) + '\n')
                indice += 1


########
# MAIN #
########


def main():


if __name__ == '__main__':
    main()
