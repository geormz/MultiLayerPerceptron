import numpy as np
import matplotlib.pyplot as plt

# Función para calcular el gradiente de f(x1, x2)
def gradient(x1, x2):
    df_dx1 = 2 * x1 * np.exp(-(x1**2 + 3 * x2**2))
    df_dx2 = 6 * x2 * np.exp(-(x1**2 + 3 * x2**2))
    return np.array([df_dx1, df_dx2])

# Función f(x1, x2)
def func(x1, x2):
    return 10 - np.exp(-(x1**2 + 3 * x2**2))

# Algoritmo de descenso de gradiente
def gradient_descent(learning_rate, num_iterations):
    x = np.array([0.0, 0.0])  # Punto inicial
    history = [x]  # Para almacenar la historia de los puntos
    for _ in range(num_iterations):
        grad = gradient(x[0], x[1])
        x = x - learning_rate * grad
        history.append(x)
    return np.array(history)

# Parámetros del descenso de gradiente
learning_rate = 0.1
num_iterations = 100

# Ejecutar el descenso de gradiente
trajectory = gradient_descent(learning_rate, num_iterations)

# Graficar la función y la trayectoria del descenso de gradiente
x1 = np.linspace(-3, 3, 400)
x2 = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = func(X1, X2)





plt.contour(X1, X2, Z, levels=np.linspace(Z.min(), Z.max(), 100), cmap='viridis')
plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', c='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Descenso de Gradiente en f(x1, x2)')

# Obtener el valor mínimo encontrado
min_x1, min_x2 = trajectory[-1]
min_value = func(min_x1, min_x2)

# Mostrar el valor mínimo en la gráfica con una etiqueta
plt.annotate(f'Min: f({min_x1:.2f}, {min_x2:.2f}) = {min_value:.2f}', (min_x1, min_x2), color='black', fontsize=12)

plt.show()



