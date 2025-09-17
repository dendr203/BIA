import numpy as np
import matplotlib as mp

class Solution:
  def __init__(self, dimension, lower_bound, upper_bound):
    self.dimension = dimension
    self.lB = lower_bound  # we will use the same bounds for all parameters
    self.uB = upper_bound
    self.parameters = np.zeros(self.dimension) #solution parameters
    self.f = np.inf  # objective function evaluation
  
  def eval():
    
    print()



class Function:
  def __init__(self, name):
    self.name = name

  def sphere(self, dimensions):
    return np.sum(np.square(dimensions))




print("Start")
f = Function("sphere_function")
print(f.sphere([10,2]))




