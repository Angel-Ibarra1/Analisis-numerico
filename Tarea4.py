import numpy as np

#%%
"""Problema 11"""

## Modulo Newton-Raphson
## raiz = newtonRaphson(f,df,a,b,tol=1.0e-9).
## Encuentra la raiz de f(x) = 0 combinando Newton-Raphson
## con biseccion. La raiz debe estar en el intervalo (a,b).
## Los usuarios definen f(x) y su derivada df(x).
def err(string):
  print(string)
  input('Press return to exit')
  sys.exit()

def newtonRaphson(f,df,a,b,tol=1.0e-9):
  from numpy import sign
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if sign(fa) == sign(fb): err('La raiz no esta en el intervalo')
  x = 0.5*(a + b)
  for i in range(30):
    print(i)
    fx = f(x)
    if fx == 0.0: return x 
    if sign(fa) != sign(fx): b = x # Haz el intervalo mas pequeño
    else: a = x
    dfx = df(x)  
    try: dx = -fx/dfx # Trata un paso con la expresion de Delta x
    except ZeroDivisionError: dx = b - a # Si division diverge, intervalo afuera
    x = x + dx # avanza en x
    if (b - x)*(x - a) < 0.0: # Si el resultado esta fuera, usa biseccion
      dx = 0.5*(b - a)
      x = a + dx 
    if abs(dx) < tol*max(abs(b),1.0): return x # Revisa la convergencia y sal
  print('Too many iterations in Newton-Raphson')

def f(x): return x * np.sin(x) + 3 * np.cos(x) - x
def df(x): return x * np.cos(x) - 1 - 2 * np.sin(x)
root = newtonRaphson(f,df,-6,6)
print('Root =',root)

#%%
"""Problema 19"""

# Constantes
u = 2510  # m/s
M0 = 2.8e6  # kg
mpunto = 13.3e3  # kg/s
g = 9.81  # m/s^2
v = 335  # m/s (velocidad del sonido)

from scipy.optimize import fsolve

# Constants
u = 2510  # m/s
M0 = 2.8e6  # kg
mpunto = 13.3e3  # kg/s
g = 9.81  # m/s^2
v = 335  # m/s (speed of sound)

# Para despejar la t nos apoyaremos de una libreria
def velocidad(t):
    return u * np.log(M0 / (M0 - mpunto * t)) - g * t - v

# Initial guess
t = 10  # seconds

# Solve for t
t = fsolve(velocidad, t)[0]
print(t)
#%%
"""Problema 9"""
import numpy as np

# Datos de la tabla
x_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
f_vals = np.array([0.000000, 0.078348, 0.138910, 0.192916, 0.244981])

# Calcular f'(0.2) usando segunda derivada de f con aproximación central
# f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
# Aquí, x = 0.2, h = 0.1

h = 0.1
x = 0.2
f_prime_0_2 = (f_vals[3] - f_vals[1]) / (2 * h)

"""Problema 10"""
# Derivada de sin(x) en x=0.8 con cinco cifras significativas
# Parte (a): segunda derivada de f con aproximación forward
# f'(x) ≈ (f(x+h) - f(x)) / h

# Parte (b): usando segunda derivada de f con aproximación central
# f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

# Usaremos varios valores de h para ver cuál da el resultado más cercano 
# al valor real

from math import sin, cos

x = 0.8
true_derivative = cos(x)
hs = [10**-i for i in range(1, 10)]
forward_errors = []
central_errors = []

for h in hs:
    forward_diff = (sin(x + h) - sin(x)) / h
    central_diff = (sin(x + h) - sin(x - h)) / (2 * h)
    forward_errors.append(abs(forward_diff - true_derivative))
    central_errors.append(abs(central_diff - true_derivative))

# Encontrar el mejor h para cada método
best_forward_index = np.argmin(forward_errors)
best_central_index = np.argmin(central_errors)

best_forward_h = hs[best_forward_index]
best_central_h = hs[best_central_index]

best_forward_result = (sin(x + best_forward_h) - sin(x)) / best_forward_h
best_central_result = (sin(x + best_central_h) - sin(x - best_central_h)) / (2 * best_central_h)

f_prime_0_2, best_forward_h, best_forward_result, best_central_h, best_central_result, true_derivative

# Imprimir los resultados de forma ordenada y con cinco cifras significativas

print("Problema 9:")
print(f"f'(0.2) ≈ {f_prime_0_2:.5f} usando segunda derivada de f con aproximación central\n")

print("Problema 10:")
print(f"Valor verdadero de d(sin(x))/dx en x=0.8: {true_derivative:.10f}\n")

print("(a) segunda derivada de f con aproximación forward:")
print(f"  Mejor h: {best_forward_h}")
print(f"  Resultado: {best_forward_result:.10f}")
print(f"  Error absoluto: {abs(best_forward_result - true_derivative):.2e}\n")

print("(b) segunda derivada de f con aproximación central:")
print(f"  Mejor h: {best_central_h}")
print(f"  Resultado: {best_central_result:.10f}")
print(f"  Error absoluto: {abs(best_central_result - true_derivative):.2e}")

#%%
"""Problema 1"""
# Definimos la función
from math import tan, log, pi
import matplotlib.pyplot as plt

# Función que queremos integrar: ln(1 + tan(x))
def f(x):
    return log(1 + tan(x))

# Implementación de la regla trapezoidal recursiva
def trapecio_recursiva(f, a, b, Iold, k):
    if k == 1:
        # Primera aproximación usando una sola aplicación del trapecio
        Inew = (f(a) + f(b)) * (b - a) / 2.0
    else:
        n = 2**(k - 2)            # Número de nuevos puntos
        h = (b - a) / n           # Espaciamiento entre puntos
        x = a + h / 2.0           # Primer punto medio
        sum = 0.0
        for i in range(n):
            sum += f(x)
            x += h
        # Combinamos la aproximación anterior con los nuevos puntos
        Inew = (Iold + h * sum) / 2.0
    return Inew

# Evaluamos la integral ∫₀^{π/4} ln(1 + tan(x)) dx recursivamente
a = 0
b = pi / 4
k_max = 10
Iold = 0.0
resultados = []

for k in range(1, k_max + 1):
    Inew = trapecio_recursiva(f, a, b, Iold, k)
    resultados.append((k, Inew, abs(Inew - Iold)))
    Iold = Inew

resultados[-5:]  # Mostrar las últimas 5 aproximaciones para ver convergencia

# la regla trapezoidal recursiva tiene una alta eficacia en funciones suaves, 
# en este caso intervalos sin discontinuidades. 
