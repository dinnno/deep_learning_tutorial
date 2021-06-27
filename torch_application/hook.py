'''
we can register a hook on a Tensor or a nn.Module

func hook : when the either forward or backward is called, hook is executed

1 : Forard Hook  >  foward pass 
2 : Backward Hook  >  backward pass


'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import matplotlib.pyplot as plt

from torch import nn, Tensor
from torchvision.models import resnet50
from torchvision import transforms
from torchvision import models
from PIL import Image

############### simple hook exemple ###################


# '''
# we can register hook on Module or Tensor
# '''

# a = torch.tensor(2.0, requires_grad=True)
# b = torch.tensor(3.0, requires_grad=True)

# c = a*b

# d = torch.tensor(5.0, requires_grad=True)

# e = c*d

# # e.backward()
# print(e)  #this case, we can't check all gredient

# # same ex using hook

# def c_hook(grad):
#     print(grad)
#     return grad + 2

# c.register_hook(c_hook)
# c.register_hook(lambda grad : print(grad))
# c.retain_grad()

# d.register_hook(lambda grad : grad + 100)

# e.retain_grad()
# e.register_hook(lambda grad : grad + 2)
# e.retain_grad()

# e.backward()

# h = c.register_hook(c_hook)

# h.remove()

############### visualize nth layer features #########

vgg = models.vgg16(pretrained=True).cuda()
#print(vgg)

class LayerResult:
    def __init__(self, hookers, layer_idx):
        self.hook = hookers[layer_idx].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output) :
        self.features = output.cpu().data.numpy()

    def remove_hook(self):
        self.hook.remove()

result = LayerResult(vgg.features, 15)

img = Image.open('./images/cat/download.jpg')
img = transforms.ToTensor()(img).unsqueeze(0)
vgg(img.cuda())

activations = result.features

fig, axes = plt.subplots(8,8)
for row in range(8):
    for column in range(8):
        axis = axes[row][column]
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.imshow(activations[0][row*8+column])
plt.show()

################# visualize weights ######################

print(vgg.state_dict().keys())
weights = vgg.state_dict()['features.0.weight'].cpu()

fig, axes = plt.subplots(8,8)
for row in range(8):
    for column in range(8):
        axis = axes[row][column]
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.imshow(activations[0][row*8+column])
plt.show()