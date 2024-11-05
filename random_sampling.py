# this file implement the orthgonoal space sampling method in python
from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

def pure_random_sampling(num_s):
    x_list = []
    y_list = []
    for n in range(num_s):
        x_c = rng.uniform(low = -1, high = 1)
        y_c = rng.uniform(low = -1, high = 1)
        x_list.append(x_c)
        y_list.append(y_c)
    return x_list, y_list


x,y = pure_random_sampling(100)
plt.scatter(x,y, color = "red")
plt.title("pure random sampling using np.random.uniform()")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.legend()
plt.show()

