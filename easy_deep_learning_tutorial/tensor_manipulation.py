import torch
import numpy as np

'''
torch function view

view : numpy -> reshape 

'''

t = np.array( [[[ 1, 2, 3],
               [ 4, 5, 6]], 
              [[ 7, 8, 9],
               [10, 11, 12]]])

print (t.shape)

float_tensor = torch.FloatTensor(t)

print (float_tensor.shape)
    