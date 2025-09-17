import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FunctionVisualizer:
  def __init__(self, func_name, dimension, lower_bound, upper_bound):
    self.name = func_name
    self.dimension = dimension
    self.lB = lower_bound  # we will use the same bounds for all parameters
    self.uB = upper_bound
    
    
    self.parameters = np.zeros(self.dimension) #solution parameters
    self.f = np.inf  # objective function evaluation

  def sphere(self, x):
    return np.sum(np.square(x))

  def eval(self, x):
    return self.sphere(x)

  def visualise(self):
    
    #vytvoření navzájem si stejně vzdálených bodů na ose x a y s rozlišením 100 (100 bodů)
    x = np.linspace(self.lB, self.uB, 100)
    y = np.linspace(self.lB, self.uB, 100)
    #vytvoření mřížky všech kombinací x_i y_j z těchto hodnot 
    X, Y = np.meshgrid(x, y)
    
    print(X)
    print(Y)
    
    #generování random float od lb do ub
    print(np.random.uniform(self.lB, self.uB))    
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
      for j in range(Y.shape[1]):
        point = np.zeros(self.dimension)
        point[0] = X[i, j]
        point[1] = Y[i, j]
        Z[i, j] = self.eval(point)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.plot_surface(X, Y, Z, cmap='viridis')  # plocha
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.set_title(f'{self.name.capitalize()} function')
    
    plt.show()
    
    





print("Start")

visualizer = FunctionVisualizer("sphere", 3, lower_bound=-5.12, upper_bound=5.12)
visualizer.visualise()


