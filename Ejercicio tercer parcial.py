import numpy as np
from math import *
import matplotlib.pyplot as plt
# Autores:
# Luis Mario Ruelas Heras
# Jose Angel Ibarra Corvera

# Parametros
theta = np.radians(30) # Angulo de lanzamiento en radianes
m = 0.25 #kg
v0 = 50 #m/s
C_D = 0.03 #kg/(m/s)^(1/2)
g = 9.80665 #m/s^2

# Valores iniciales
x0 = 0 # Posición inicial en  (m)
x1 = v0*cos(theta) # Velocidad inicial en x (m/s)
y0 = 0 # Posición inicial en y (m)
y1 = v0*sin(theta) # Velocidad inicial en y (m/s)

# Metodo de Runge-Kutta
def Run_Kut4(F,x,y,xStop,h):
      def run_kut4(F,x,y,h):
        K0 = h*F(x,y)
        K1 = h*F(x + h/2.0, y + K0/2.0)
        K2 = h*F(x + h/2.0, y + K1/2.0)
        K3 = h*F(x + h, y + K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0
      X = []
      Y = []
      X.append(x)
      Y.append(y)
      while x < xStop:
        h = min(h,xStop - x)
        y = y + run_kut4(F,x,y,h)
        x=x+h
        X.append(x)
        Y.append(y)
      return np.array(X),np.array(Y)

def imprimeSol(X,Y,frec):
      def imprimeEncabezado(n):
        print("\n x ",end=" ")
        for i in range (n):
          print(" y[",i,"] ",end=" ")
        print()
    
      def imprimeLinea(x,y,n):
        print("{:13.4e}".format(x),end=" ")
        for i in range (n):
          print("{:13.4e}".format(y[i]),end=" ")
        print() 
  
      m = len(Y)
      try: n = len(Y[0])
      except TypeError: n = 1
      if frec == 0: frec = m
      imprimeEncabezado(n)
      for i in range(0,m,frec):
       imprimeLinea(X[i],Y[i],n)
      if i != m - 1: imprimeLinea(X[m - 1],Y[m - 1],n)
      


def F(x,y):
      v = sqrt(y[1]**2 + y[3]**2)
      F = np.zeros(4)
      F[0] = y[1]
      F[1] = (-C_D / m) * y[1] * v**(1/2)
      F[2] = y[3]
      F[3] = (-C_D / m) * y[3] * v**(1/2) - g  
      return F

y = np.array([x0, x1, y0, y1])
X,Y=Run_Kut4(F, 0.0, y, 6.0, 0.05)
print("La solución es")
imprimeSol(X, Y, 4)

# Grafica de la trayectoria
plt.figure(figsize = (9, 4))
plt.plot(Y[:, 0], Y[:, 2], color = 'black')
plt.xlim(0, 70)
plt.ylim(0, 18)

plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.title('Trayectoria con método de Runge-Kutta 4')

plt.grid()
plt.show()