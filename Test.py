import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps

resolution = 100
graph_folder_name = "graphs_files"

class FunctionVisualizer:
  def __init__(self, func_name, dimensions, lower_bound, upper_bound, make_new_file=False):
    self.name = func_name
    self.d = dimensions
    self.lB = lower_bound  # we will use the same bounds for all parameters
    self.uB = upper_bound
    
    self.mk = make_new_file
    self.parameters = np.zeros(self.d) #solution parameters
    self.f = np.inf  # objective function evaluation


  #FUNCTIONS
  #každý bod má již počet dimenzí, takže můžu použít jen sum a nemusím specifikovat dimenze
  def sphere(self, x):
    return np.sum(x**2)

  def ackley(self, x, a=20, b=0.2, c=np.pi * 2):
    sum_sqrt = np.sum(np.square(x))
    sum_cos = np.sum(np.cos(c * x))
    
    first_term = -a * np.exp(-b * (np.sqrt(sum_sqrt / self.d)))
    second_term = -np.exp(sum_cos / self.d)
    
    return first_term + second_term + a + np.exp(1)

  def rastrigin(self, x):
    sum_bracket = np.sum(np.square(x) - 10*np.cos(2*np.pi*x))
    
    return 10+self.d + sum_bracket
  
  def rosenbrock(self, x):
    return np.sum(100 * np.square(x[1:] - np.square(x[:-1])) + (np.square(x[:-1] - 1)))
  
  def griewank(self, x):
    sum = np.sum(np.square(x)/4000)
    
    prod = 1
    for i in range(self.d):
      prod *= np.cos(x[i]/(np.sqrt(i + 1)))
      
    return sum - prod + 1
  
  def schwefel(self, x):
    return 418.9829 * self.d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
  
  def levy(self, x):
    w = 1 + ((x - 1) / 4.0)
    
    first_term = np.square(np.sin(np.pi * w[0]))
    if self.d > 1:
      sum_term = np.sum(np.square(w[:-1]-1) * (1 + 10 * np.square(np.sin(np.pi * w[:-1] + 1))))
    else:
      sum_term = 0.0
    
    second_term = np.square(w[-1] - 1) * (1 + np.square(np.sin(2 * np.pi * w[-1])))
    
    return first_term + sum_term + second_term
  
  def michalewicz(self, x, m=10):
    i = np.arange(1, self.d + 1)
    
    return -np.sum(np.sin(x) * np.sin((i * np.square(x)) / (np.pi))**(2*m))
  
  def zakharov(self, x):
      i = np.arange(1, self.d + 1)
      sum1 = np.sum(x**2)
      sum2 = np.sum(0.5 * i * x)
      return sum1 + sum2**2 + sum2**4
  #FUNCTIONS END


  #eval what functions we want to use in visualizing
  def function_type(self, x):
    if self.name == "sphere":
      return self.sphere(x)
    elif self.name == "ackley":
      return self.ackley(x) 
    elif self.name == "rastrigin":
      return self.rastrigin(x)
    elif self.name == "rosenbrock":
      return self.rosenbrock(x)
    elif self.name == "griewank":
      return self.griewank(x)
    elif self.name == "schwefel":
      return self.schwefel(x)
    elif self.name == "levy":
      return self.levy(x)
    elif self.name == "michalewicz":
      return self.michalewicz(x)
    elif self.name == "zakharov":
      return self.zakharov(x)

  def create_graph_file(self):
    #vytvoření navzájem si stejně vzdálených bodů na ose x a y s daným rozlišením (teda kolik bodů chci mít na jedné ose)
    #np.linspace využito protože vždy bude obsahovat koncový bod (vždy se dopočítávají ostatní body aby se vešel)
    x = np.linspace(self.lB, self.uB, resolution)
    y = np.linspace(self.lB, self.uB, resolution)
    #vytvoření mřížky všech kombinací x_i y_j z těchto hodnot 
    X, Y = np.meshgrid(x, y)
    
    #vytvoření pole pro Z souřadnice stejného tvaru jako X[d]
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        point = np.zeros(self.d)
        point[0] = X[i, j]
        point[1] = Y[i, j]
        #print(point)
        Z[i, j] = self.function_type(point)
    return X, Y, Z
  
  def visualise(self, X, Y, Z, label, highlight_points=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k', linewidth=0.5, alpha=0.9) 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x)')
    ax.set_title(f'{self.name.capitalize()} function')
    
    ax.set_xlim(self.lB, self.uB)
    ax.set_ylim(self.lB, self.uB)
    ax.set_zlim(Z.min(), Z.max())
    
    ax.view_init(elev= 25, azim=-120)
    
    if highlight_points is not None:
      ax.scatter(highlight_points[0], highlight_points[1], highlight_points[2], color = "red", s=50, label=label)
      ax.legend()
    
    plt.show()
    return
  
  def compute(self):
    os.makedirs(graph_folder_name, exist_ok=True)
    filepath = os.path.join(graph_folder_name, f"{self.name}_data.npz")
    
    if os.path.exists(filepath) and not self.mk:
      print(f"Loading cached data from existing file {self.name}_data.npz")
      data = np.load(filepath)
      X = data["X"]
      Y = data["Y"]
      Z = data["Z"]
    else:
      print(f"Data not found, computing them and saving to {self.name}_data.npz")
      X, Y, Z = self.create_graph_file()
      np.savez(filepath, X=X, Y=Y, Z=Z)
      print("Data saved!!!")
    
    return X, Y, Z
  
  
  #ALGORITHMS
  def blind_search(self, iterations=100):    
    X, Y, Z = self.compute()
    
    best_x = None
    best_f = np.inf
    for _ in range(iterations):
        candidate = np.random.uniform(self.lB, self.uB, self.d)
        f_val = self.function_type(candidate)
        if f_val < best_f:
            best_f = f_val
            best_x = candidate
    print(f"Best found point: {best_x}, value: {best_f}")
    coordinates = np.append(best_x, best_f)
    
    self.visualise(X, Y, Z, "Blind search algorithm", coordinates)
    
    return
  
  #ALGORITHMS END





print("Start")

sphere = FunctionVisualizer("sphere", 2, -5.12, 5.12)
sphere.blind_search()


ackley = FunctionVisualizer("ackley", 2, -32.768, 32.768)
ackley.blind_search()

rastrigin = FunctionVisualizer("rastrigin", 2, -5.12, 5.12)
rastrigin.blind_search()

"""
rosenbrock = FunctionVisualizer("rosenbrock", 2, -2.048, 2.048)
rosenbrock.compute()

#zoom, normal -600, 600
griewank = FunctionVisualizer("griewank", 2, -10, 10)
griewank.compute()

schwefel = FunctionVisualizer("schwefel", 2, -500, 500)
schwefel.compute()

levy = FunctionVisualizer("levy", 2, -10, 10)
levy.compute()

michalewicz= FunctionVisualizer("michalewicz", 2, 0, np.pi)
michalewicz.compute()

zakharov= FunctionVisualizer("zakharov", 2, -10, 10)
zakharov.compute()
"""