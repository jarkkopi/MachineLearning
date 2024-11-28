import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np

x_p = []
y_p = []
def my_linfit(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = len(x)

    a = (np.sum(x * y) - n * x_mean * y_mean) / (np.sum(x**2) - n * x_mean**2)
    b = y_mean - a * x_mean
    return a,b

def onclick(event):
    if event.button == 1:
        x_p.append(event.xdata)
        y_p.append(event.ydata)
        plt.plot(event.xdata, event.ydata, 'kx')
        plt.draw()

    elif event.button == 3:
        plt.gcf().canvas.mpl_disconnect(cid)
        a, b = my_linfit(np.array(x_p), np.array(y_p))
        xp = np.linspace(min(x_p), max(x_p), 100)
        plt.plot(xp, a*xp + b, 'r-', label='Fit line')
        plt.legend()
        plt.title('Linear Fit')
        plt.draw()
        print(f"My fit: a = {a} and b = {b}")
        

x = np.random.uniform(-2,5,10)
y = np.random.uniform(0,3,10)

answer = input("Random or input? (r/i)")

if answer == "r":
    a, b = my_linfit(x, y)
    plt.plot(x, y, 'kx')
    xp = np.arange(-2, 5, 0.1)
    plt.plot(xp, a*xp + b, 'r-')
    print(f"My fit: a = {a} and b = {b}")
    plt.show()
elif answer == 'i':
    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.draw()
else:
    print("not an option")
