# montecarlo assignment
# name: Xuening Tang

import random as ra
import math as ma
import matplotlib.pyplot as plt

# define a function that uses Monte Carlo stimulation to calculate the integral
def montecarlo(func, x1, y1, x2, y2):
    area = 0
    f_good_p = 0
    f_good_n = 0
    square = (x2-x1)*(y2-y1)

    for index in range(0,50000):
        dot_x = ra.random ()*(x2-x1) + x1
        dot_y = ra.random ()*(y2-y1) + y1
        if (func(dot_x) > dot_y) and (dot_y > 0) :
            f_good_p += 1
        if (func(dot_x) < dot_y) and (dot_y < 0) :
            f_good_n += 1
    f_good_p_p = f_good_p/50000
    f_good_n_p = f_good_n/50000

    area = square * f_good_p_p - square * f_good_n_p
    return area


# plot the Monte Carlo stimulation in a graph
def plot_montecarlo(func, x1, y1, x2, y2):
    X_values = []
    y_values = []
    x = x1
    while x < x2:
        X_values.append (x)
        x += 0.01
    for y in X_values:
        y_values.append (func(y))
    
# plot the random dots
    for index in range (0,50000):
        dot_x = ra.random ()*(x2-x1) + x1
        dot_y = ra.random ()*(y2-y1) + y1
        if ((func (dot_x) > dot_y) and (dot_y > 0)) or ((func(dot_x) < dot_y) and (dot_y < 0)) :
            plt.plot (dot_x, dot_y,'go')
        else:
            plt.plot (dot_x, dot_y,'ro')

# plot the functiom
    plt.plot (X_values, y_values, 'b-')
    
    plt.show ()

# testing of montecarlo calculations and ploting (This part needs to be set comments when using checkpy,
# otherwise checkpy may run out of time)
"""def func1(x):
		return ma.sin (x**2)
print (montecarlo (func1, 0, -1, ma.pi, 1))
plot_montecarlo (func1, 0, -1, ma.pi, 1)"""