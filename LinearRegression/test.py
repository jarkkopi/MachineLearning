import numpy as np
import matplotlib.pyplot as plt

def my_linfit(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = len(x)

    a = (np.sum(x * y) - n * x_mean * y_mean) / (np.sum(x**2) - n * x_mean**2)
    b = y_mean - a * x_mean
    
    return a, b

x = np.random.uniform(-2, 5, 10)
y = np.random.uniform(0, 3, 10)

a, b = my_linfit(x, y)

plt.plot(x, y, 'kx', label='Data points')
xp = np.linspace(min(x) - 1, max(x) + 1, 100)
plt.plot(xp, a * xp + b, 'r-', label='Fitted line')

print(f"My fit: a = {a} and b = {b}")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Fit')
plt.legend()

plt.show()
