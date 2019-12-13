from typing import NamedTuple  # >= Python.3.6.0
import torch
import numpy as np
# class Employee(NamedTuple):
#   name = 'test'
#   department = 'test'
#   salary = 111
#   is_remote = False  # >= Python.3.6.1
    
# bob = Employee()
# bob.name = 'wokule'
# print(bob.name)

print('Test for torch.from_numpy():')
a = np.array([213,21,3,23,543,52,4,234,123,21])
b = torch.from_numpy(a)
print(b)
print(type(b))
print('Test succeed')