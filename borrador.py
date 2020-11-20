import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sigma = 1
W0 = 5

# Variables aleatorias A y Z, iguales para este caso
vaA = stats.norm(0, np.sqrt(sigma)) #Del enunciado la media = 0 y se eligió un valor de sigma = 1
vaZ = stats.norm(0, np.sqrt(sigma)) #Del enunciado la media = 0 y se eligió un valor de sigma = 1


#Se eligen valores arbitrarios para T y t_final
# Se crea el vector de tiempo
T = 90			# número de elementos
t_final = 12	# tiempo en segundos
t = np.linspace(0, t_final, T)

#Se elige un valor arbitrario de N
# Inicialización del proceso aleatorio X(t) con N realizaciones
N = 11
X_t = np.empty((N, len(t)))	# N funciones del tiempo x(t) con T puntos

# Crear muestras del proceso x(t) (A y Z independientes)
for i in range(N):
	A = vaA.rvs()
	Z = vaZ.rvs()
	x_t = A * np.cos(np.pi*t + Z)
	X_t[i,:] = x_t
	plt.plot(t, x_t)

# Mostrar las realizaciones, y su promedio calculado y teórico
plt.title('Realizaciones del proceso aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.show()

# Promedio de las N realizaciones en cada instante (cada punto en t)
P = [np.mean(X_t[:,i]) for i in range(len(t))]
plt.plot(t, P, lw=6)

# Se grafica el resultado teórico del valor esperado
E = 0*t
plt.plot(t, E, '-.', lw=4)

# Para mostrar las realizaciones, y su promedio calculado y teórico
plt.title('Resultado teorico del valor esperado')
plt.xlabel('$t$')
plt.ylabel('$E$')
plt.show()

# T valores de desplazamiento tau
desplazamiento = np.arange(T)
taus = desplazamiento/t_final

# Inicialización de matriz de valores de correlación para las N funciones
corr = np.empty((N, len(desplazamiento)))

# Nueva figura para la autocorrelación
plt.figure()

# Cálculo de correlación para cada valor de tau
for n in range(N):
	for i, tau in enumerate(desplazamiento):
		corr[n, i] = np.correlate(X_t[n,:], np.roll(X_t[n,:], tau))/T
	plt.plot(taus, corr[n,:])

# Valor teórico de correlación
Rxx = 1 * np.cos(W0*taus)

# Gráficas de correlación para cada realización
plt.plot(taus, Rxx, '-.', lw=4, label='Correlación teórica')
plt.title('Funciones de autocorrelación de las realizaciones del proceso')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$R_{XX}(\tau)$')
plt.legend()
plt.show()


