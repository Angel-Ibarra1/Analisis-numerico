# %%
#Sección 7.1 ejercicio 3
import numpy as np
import math

# Definimos la función diferencial
def F(x, y):
    return math.sin(y)

# Definimos la función eulerint con entradas: las condiciones iniciales x y y, 
# con el valor terminal de x que se llama xStop, con el incremento en x 
# denominado h y con la función que nos da el usuario F.

def eulerint(F, x, y, xStop, h):
    X = [] # Lista para guardar los valores de x
    Y = [] # Lista para guardar los valores de y
    X.append(x)
    Y.append(y)
    
    # Mientras no se alcance xStop, se avanza en pasos de h
    while x < xStop:
        h = min(h, xStop - x) # Ajuste del último paso si es necesario
        y = y + h * F(x, y) # Fórmula de Euler: y_{n+1} = y_n + h*f(x_n, y_n)
        x = x + h # Se avanza al siguiente x
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

# Runge-Kutta de segundo orden, este método es para comparar los resultados
# del ejemplo 7.3
def rungekutta2(F, x0, y0, xStop, h):
    X = [x0]
    Y = [y0]
    x = x0
    y = y0
    while x < xStop:
        h = min(h, xStop - x)
        K0 = h * F(x, y)
        K1 = h * F(x + h/2, y + K0/2)
        y = y + K1
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

# Imprimir solución
# Definimos la función imprimeSol para imprimir X y Y que se obtienen como 
# solución de la integración numérica.

def imprimeSol(X, Y, frec):

    def imprimeEncabezado(n):
        print("\n     x        ", end="")
        for i in range(n):
            print(f"      y[{i}]      ", end="")
        print()

    def imprimeLinea(x, y, n):
        print("{:13.4e}".format(x), end=" ")
        for i in range(n):
            print("{:13.4e}".format(y[i]), end=" ")
        print()

    m = len(Y)
    try:
        n = len(Y[0])
    except TypeError:
        n = 1

    imprimeEncabezado(n)
    for i in range(0, m, frec):
        if n == 1:
            imprimeLinea(X[i], [Y[i]], n)
        else:
            imprimeLinea(X[i], Y[i], n)

# Parámetros iniciales
x0 = 0 # Valor inicial de x
y0 = 1 # Valor inicial de y
xStop = 0.5 # Valor final de x
h = 0.1 # Tamaño de paso

# Soluciones
X_euler, Y_euler = eulerint(F, x0, y0, xStop, h)
X_rk2, Y_rk2 = rungekutta2(F, x0, y0, xStop, h)

# Comparar resultados

print("\nMétodo de Euler:")
imprimeSol(X_euler, Y_euler, frec=1)

print("\nMétodo de Runge-Kutta de 2do orden:")
imprimeSol(X_rk2, Y_rk2, frec=1)

# %%
#Sección 7.1 ejercicio 4
import numpy as np
import math

# Definimos la función de la EDO: y' = y^(1/3)

def F(x, y):
    return y ** (1/3) if y > 0 else 0  # para evitar errores cuando y=0

# Método de Euler

def eulerint(F, x, y, xStop, h):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h, xStop - x)
        y = y + h * F(x, y)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# Función para imprimir resultados

def imprimeSol(X, Y, frec):

    def imprimeEncabezado(n):
        print("\n     x        ", end="")
        for i in range(n):
            print(f"      y[{i}]      ", end="")
        print()

    def imprimeLinea(x, y, n):
        print("{:13.4e}".format(x), end=" ")
        for i in range(n):
            print("{:13.4e}".format(y[i]), end=" ")
        print()

    m = len(Y)
    try:
        n = len(Y[0])
    except TypeError:
        n = 1

    imprimeEncabezado(n)
    for i in range(0, m, frec):
        if n == 1:
            imprimeLinea(X[i], [Y[i]], n)
        else:
            imprimeLinea(X[i], Y[i], n)


# Configuración del problema

x0 = 0
xStop = 1
h = 0.05  # paso de integración

# Caso (a) y(0) = 0
print("\nCaso (a): y(0) = 0")
X0, Y0 = eulerint(F, x0, 0.0, xStop, h)
imprimeSol(X0, Y0, frec=1)

# Caso (b) y(0) = 1e-16 (valor pequeño)
print("\nCaso (b): y(0) = 1e-16")
X1, Y1 = eulerint(F, x0, 1e-16, xStop, h)
imprimeSol(X1, Y1, frec=1)

# %%
#Sección 8.1 ejercicio 3 a) y c)

import matplotlib.pyplot as plt
import numpy as np
from math import *

# Método de Runge-Kutta 4 para sistemas de EDO
def Run_Kut4(F, x, y, xStop, h):
    def run_kut4(F, x, y, h):
        K0 = h * F(x, y)
        K1 = h * F(x + h / 2.0, y + K0 / 2.0)
        K2 = h * F(x + h / 2.0, y + K1 / 2.0)
        K3 = h * F(x + h, y + K2)
        return (K0 + 2.0 * K1 + 2.0 * K2 + K3) / 6.0

    X = [x]
    Y = [y]
    while x < xStop:
        h = min(h, xStop - x)
        y = y + run_kut4(F, x, y, h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

# Impresión de resultados
def imprimeSol(X, Y, frec):
    def imprimeEncabezado(n):
        print("\n x ", end=" ")
        for i in range(n):
            print(f" y[{i}] ", end=" ")
        print()

    def imprimeLinea(x, y, n):
        print("{:13.4e}".format(x), end=" ")
        for i in range(n):
            print("{:13.4e}".format(y[i]), end=" ")
        print()

    m = len(Y)
    try: n = len(Y[0])
    except TypeError: n = 1

    if frec == 0: frec = m
    imprimeEncabezado(n)
    for i in range(0, m, frec):
        imprimeLinea(X[i], Y[i], n)
    if i != m - 1: imprimeLinea(X[m - 1], Y[m - 1], n)

# Método de Ridder para encontrar raíces
def Ridder(f, a, b, tol=1.0e-9):
    fa = f(a)
    fb = f(b)
    if fa == 0.0: return a
    if fb == 0.0: return b
    if np.sign(fa) != np.sign(fb): c = a; fc = fa
    for i in range(30):
        c = 0.5 * (a + b); fc = f(c)
        s = sqrt(fc**2 - fa * fb)
        if s == 0.0: return None
        dx = (c - a) * fc / s
        if (fa - fb) < 0.0: dx = -dx
        x = c + dx; fx = f(x)
        if i > 0:
            if abs(x - xOld) < tol * max(abs(x), 1.0): return x
        xOld = x
        if np.sign(fc) == np.sign(fx):
            if np.sign(fa) != np.sign(fx): b = x; fb = fx
            else: a = x; fa = fx
        else:
            a = c; b = x; fa = fc; fb = fx
    print("Demasiadas iteraciones")
    return None

# INICIO: Inciso (a)
# Definimos el sistema de 2 ecuaciones para y'' = -e^{-y}
def F1(x, y):
    F = np.zeros(2)
    F[0] = y[1]                 # y' = z
    F[1] = -exp(-y[0])          # z' = -e^{-y}
    return F

# Condiciones iniciales: y(0) = 1, y'(0) = u (estimado)
def initCond(u): return np.array([1.0, u])

# Función residual para encontrar u tal que y(1) = 0.5
def r(u):
    Y = Run_Kut4(F1, 0.0, initCond(u), 1.0, 0.1)[1]
    return Y[-1][0] - 0.5

# Método de Shooting: encontrar u tal que r(u) = 0
u = Ridder(r, -2.0, 0.0)

# Resolver con el valor óptimo de u
X, Y = Run_Kut4(F1, 0.0, initCond(u), 1.0, 0.1)
print("\nSolución del inciso (a):")
imprimeSol(X, Y, 2)
print(f"Valor estimado de y'(0) = {u:.4f}")

# El valor inicial de la gráfica es negativo por lo que se puede ver que inicia
# decreciendo, lo cual es razonable porque la aceleracción(y'') es negativa
# cumple con la condición de frontera.

# INICIO: Inciso (c)
# Definimos el sistema para y'' = cos(xy)
def F3(x, y):
    F = np.zeros(2)
    F[0] = y[1]
    F[1] = cos(x * y[0])
    return F

# Condiciones iniciales: y(0) = 0, y'(0) = u
def initCond3(u): return np.array([0.0, u])

# Residual para y(1) = 2
def r3(u):
    Y = Run_Kut4(F3, 0.0, initCond3(u), 1.0, 0.05)[1]
    return Y[-1][0] - 2.0

# Estimar u con Ridder
u3 = Ridder(r3, 2.0, 6.0)

# Resolver sistema
X3, Y3 = Run_Kut4(F3, 0.0, initCond3(u3), 1.0, 0.05)
print("\nSolución del inciso (c):")
imprimeSol(X3, Y3, 2)
print(f"Valor estimado de y'(0) = {u3:.4f}")

# El crecimiento al inicio es mayor debido a la función cos comienza cerca de 1
# que a medida que xy aumenta su influencia disminuye.

# Gráficas
plt.plot(X, Y[:, 0], label="Inciso (a): y'' = -e^{-y}")
plt.plot(X3, Y3[:, 0], label="Inciso (c): y'' = cos(xy)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Soluciones con Método de Disparo")
plt.legend()
plt.grid(True)
plt.show()
