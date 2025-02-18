#%%
# Ejercicio 1
# Calculadora de temperatura(grados Celsius a partir de temperaturas conocidas en grados Farenheit)
# F seran los grados Farenheit y C seran los grados Celsius
F = float(input("Ingrese una temperatura en grados Fahrenheit: "))
C = (F-32)* 5/9
print(f"La temperatura en grados Celsius es {round(C,3)} °C")

#%%
# Ejercicio 2
import math
# Evaluar sinh de tres maneras distintas
x = float(input("Número a evaluar: "))
A = math.sinh(x)
print("El seno hiperbolico de ",x," es", A)
a = (math.exp(x)- math.exp(-x))/2
print("La ecuación es igual a ",a)
B = (math.e**x - math.e**-x)/2
print("La ecuación es igual a ",B)

#%%
# Ejercicio 3
import numpy as np

# Evaluar sin(ix) y sinh(x) para verificar la identidad
# Valores de x para probar
xs = np.linspace(-2, 2, 5)  # Generamos 5 valores entre -2 y 2

print("Verificación de sin(ix) = i sinh(x):")
for x in xs:
    sin_ix = np.sin(1j * x)  # sin(ix)
    i_sinh_x = 1j * np.sinh(x)  # i * sinh(x)
    print(f"x = {x:.2f}, sin(ix) = {sin_ix:.5f}, i*sinh(x) = {i_sinh_x:.5f}, Equal: {np.allclose(sin_ix, i_sinh_x)}")

print("\nVerificación de la identidad de Euler e^(ix) = cos(x) + i sin(x):")
for x in xs:
    e_ix = np.exp(1j * x)  # e^(ix)
    cos_x = np.cos(x)  # cos(x)
    i_sin_x = 1j * np.sin(x)  # i*sin(x)
    sum_cos_sin = cos_x + i_sin_x  # cos(x) + i*sin(x)
    print(f"x = {x:.2f}, e^(ix) = {e_ix:.5f}, cos(x) + i*sin(x) = {sum_cos_sin:.5f}, Equal: {np.allclose(e_ix, sum_cos_sin)}")

#%%
# Ejercicio 4
import numpy as np

# Función para calcular las raíces de una ecuación cuadrática
def calcular_raices(a, b, c):
    discriminante = np.sqrt(b**2 - 4*a*c + 0j)  # Se suma 0j para manejar números complejos
    raiz1 = (-b + discriminante) / (2 * a)
    raiz2 = (-b - discriminante) / (2 * a)
    return raiz1, raiz2

# Ejemplos de coeficientes para probar
ejemplos = [
    (1, -3, 2),   # Raíces reales distintas
    (1, 2, 1),    # Raíz real doble
    (1, 1, 1),    # Raíces complejas
    (2, -4, 2)    # Raíz real doble
]

print("\nCálculo de raíces de ecuaciones cuadráticas:")
for a, b, c in ejemplos:
    r1, r2 = calcular_raices(a, b, c)
    print(f"Para a={a}, b={b}, c={c}: raíces = {r1}, {r2}")

#%%
# Ejercicio 5
import numpy as np

# Parámetros de la trayectoria
v0 = 10  # m/s, velocidad inicial
theta = np.radians(45)  # grados a radianes
g = 9.81  # m/s^2, gravedad
y0 = 1  # m, posición inicial

# Función de trayectoria
def trayectoria(x, v0, theta, g, y0):
    return x * np.tan(theta) - (g / (2 * v0**2 * np.cos(theta)**2)) * x**2 + y0

# Valores de x para evaluar
x_trayectoria = np.linspace(0, 20, 10)  # 10 valores entre 0 y 20 metros

print("\nCálculo de la trayectoria de la pelota:")
for x in x_trayectoria:
    y = trayectoria(x, v0, theta, g, y0)
    print(f"Para x={x:.2f} m, y={y:.2f} m")
