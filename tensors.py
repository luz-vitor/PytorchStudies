import torch
import numpy as np

# Lists
print("Lists")
my_list = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
print(my_list)
print("\n")

# Numpy arrays
print("Numpy Arrays")
np1 = np.random.rand(3, 4)
print(np1)
print("\n")

# Tensors
print("Tensors")
tensor_2d = torch.randn(3, 4)
print(tensor_2d)
print("\n")

tensor_3d = torch.zeros(2, 3, 4)
print(tensor_3d)
print("\n")

## Create tensor out of numpy array
my_tensor = torch.tensor(np1)
print(my_tensor)
