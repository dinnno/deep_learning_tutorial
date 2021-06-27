'''

####### logistic regression : binary classification #######

if class : 0, 1
    p(x=1 ;w) = 1 - p(x=0 ;w)
    
    we will use sigmoid function 
        sigmoid = 1 / (1 + e^(-x))

weight update via gradient descent

    w = w - a * gradient((cost(W))) a : learning_rate

######## Softmax Classification ##########
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [4, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#hypothesis = 1 /(1 + torch.exp(-(x_train.matmul(w) + b)))

hypothesis = torch.sigmoid(x_train.matmul(w) + b)

# loss = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
# cost = loss.mean

cost = F.binary_cross_entropy(hypothesis, y_train)

'''
higher implementation with class
'''

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        return self.sigmoid(self.linear(x))