import numpy as np
### Método de eliminación de Gauss

def gaussElimin(a,b):
  n = len(b)
  # Fase de eliminacion
  for k in range(0,n-1):
    for i in range(k+1,n):
      if a[i,k] != 0.0:
        lam = a [i,k]/a[k,k]
        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
        b[i] = b[i] - lam*b[k]
  # Fase de sustitucion hacia atras
  for k in range(n-1,-1,-1):
    b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
  return b

"""Intersección de trayectorias.
Tres objetos se mueven de tal manera que sus trayectorias son:
# 2x - y + 3z = 24
# 2y - z = 14
# 7x - 5y = 6
Encontrar su punto de intersección."""

# Programa de eliminación Gaussiana mostrando los pasos

# Matriz de coeficientes
a = np.array([[2, -1, 3], [0, 2, -1], [7, -5, 0]])

# Vector de términos independientes
b = np.array([[24], [14], [6]])

# Obtener el tamaño del sistema
n = len(b)

print('La matriz de coeficientes es:\n', a)
print('La matriz constante es:\n', b)

# Crear la matriz aumentada correctamente
matriz_aumentada = np.concatenate((a, b), axis=1, dtype=float)
print('La matriz aumentada es:\n', matriz_aumentada)


def gaussElimin2(matriz_aumentada):
    n = len(matriz_aumentada)

    # Fase de eliminación
    for k in range(n-1):  
        for i in range(k+1, n):
            if matriz_aumentada[i, k] != 0.0:  # Evitar divisiones por cero
                lam = matriz_aumentada[i, k] / matriz_aumentada[k, k]
                matriz_aumentada[i, k:] -= lam * matriz_aumentada[k, k:]  # Restar en toda la fila

        print(f'Paso {k+1} de eliminación:\n{matriz_aumentada}')

    # Fase de sustitución hacia atrás
    x = np.zeros(n)  # Inicializar vector solución

    for k in range(n-1, -1, -1):  
        x[k] = (matriz_aumentada[k, -1] - np.dot(matriz_aumentada[k, k+1:n], x[k+1:n])) / matriz_aumentada[k, k]

    return x


# Llamar a la función con la matriz aumentada
solucion = gaussElimin2(matriz_aumentada)

print("\n Usando matrices se determino lo siguiente, el punto de intersección es: ",solucion)

"""Carga de los quarks
Los protones y neutrones están formados cada uno por tres quarks. Los protones 
poseen dos quarks up (u) y un quark down (d), los neutrones poseen un quark up
y dos quarks down. Si la carga de un protón es igual al positivo de la carga
del electrón(+e) y la carga de un neutrón es cero, determine las cargas de los
quarks up y down. (Tip: suponga que +e=1 .)"""

# Definimos una ecuacion para los protones
# 2u + d = 1
# Definimos una ecuacion para los neutrones
# u + 2d = 0

### Vector de términos independientes
b = np.array([[1], [0]])

### Matriz de coeficientes
a = np.array([[2, 1], [1, 2]])

# Obtener el tamaño del sistema
n = len(b)

print('\n La matriz de coeficientes es:\n', a)
print('La matriz constante es:\n', b)

# Crear la matriz aumentada correctamente
matriz_aumentada = np.concatenate((a, b), axis=1, dtype=float)
print('La matriz aumentada es:\n', matriz_aumentada)


def gaussElimin2(matriz_aumentada):
    n = len(matriz_aumentada)

    # Fase de eliminación
    for k in range(n-1):  
        for i in range(k+1, n):
            if matriz_aumentada[i, k] != 0.0:  # Evitar divisiones por cero
                lam = matriz_aumentada[i, k] / matriz_aumentada[k, k]
                matriz_aumentada[i, k:] -= lam * matriz_aumentada[k, k:]  # Restar en toda la fila

        print(f'Paso {k+1} de eliminación:\n{matriz_aumentada}')

    # Fase de sustitución hacia atrás
    x = np.zeros(n)  # Inicializar vector solución

    for k in range(n-1, -1, -1):  
        x[k] = (matriz_aumentada[k, -1] - np.dot(matriz_aumentada[k, k+1:n], x[k+1:n])) / matriz_aumentada[k, k]

    return x


# Llamar a la función con la matriz aumentada
solucion1 = gaussElimin2(matriz_aumentada)

print("La carga de los quarks up es de: ",solucion1[0]," y la carga de los quarks down es de ",solucion1[1])

"""Meteoros
El Centro de Investigación 1 examina la cantidad de meteoros que entran a la 
atmósfera. Con su equipo de recopilación de datos durante 8 horas captó 95kg de
meteoros, por fuentes externas sabemos que fueron de 4 distintas 
masas (1kg, 5kg, 10kg y 20kg). La cantidad total de meteoros fue de 26.
Otro centro de investigación captó que la cantidad de meteoros de 5kg 
es 4 veces la cantidad de meteoros de 10kg, y el número de meteoros de 1kg 
es 1 menos que el doble de la cantidad de meteoros de 5kg. Después use matrices
para encontrar el número asociado a cada masa de meteoros."""

# Comenzamos formando nuestro sistema de ecuaciones
# Para la cantidad de meteoros
# a + b + c + d = 26
# Para la masa total de los meteoros
# a + 5b +10c +20d = 95
# Para la cantidad de meteoros de 5kg
# b = 4c
# Para la cantidad de meteoros de 1kg
# a = 2b-1

### Vector de términos independientes
b = np.array([26, 95, 0, -1])

### Matriz de coeficientes
a = np.array([[1, 1, 1, 1], [1, 5, 10, 20], [0, 1, -4, 0], [1, -2, 0, 0]])

### Guarda una copia de la matriz y el vector original
aOrig = a.copy() 
bOrig = b.copy() 

### Encuentra la solucion con eliminacion de Gauss
x = gaussElimin(a,b)

### Calcula el determinante de la matriz de coeficientes
det = np.prod(np.diagonal(a))

### Imprime la matriz de coeficientes
print('\n Matriz de coeficientes: A= \n',aOrig)
### Imprime el vector solucion
print('\n Vector solucion: x =\n',x)
### Imprime la determinante de la matriz de coeficientes
print('\n determinante =',det)
### Imprime la verificacion del resultado
print('\nVerificacion del resultado: [a]{x} - b =\n',np.dot(aOrig,x) - bOrig)

print("\n Usando matrices se determino el número asociado a cada masa de meteoros, se registraron ",x[0]," meteoros de 1kg,",x[1]," meteoros de 5kg,",x[2]," meteoros de 10kg y ",x[3]," meteoros de 20kg")
