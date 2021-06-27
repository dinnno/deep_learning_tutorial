import torch
import torch.nn as nn
from torch.nn.modules import adaptive
import torchvision

from torchvision.models import resnet50

# class PrintModel(nn.Module):
#     def __init__(self, model):
#         super(PrintModel, self).__init__()
#         self.model = model

#         for name, layer in self.model.named_children():
#             layer.__name__ = name
#             layer.register_forward_hook(
#                 lambda layer, _, output : print(f"{layer.__name__}: {output.shape}")
#             )

#     def forward(self, x):
#         return self.model(x)

# pre_model = resnet50()
# printed_resnet = PrintModel(pre_model)
# dummy_input = torch.ones(10, 3, 224, 224)

# _ = printed_resnet(dummy_input)


# #################################################################
# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

# pre_model.fc0.conv2.register_forward_hook(get_activation('fc0.conv2'))
# pre_model.fc1.conv2.register_forward_hook(get_activation('fc1.conv2'))

# output = pre_model(x)

# ################################################################
# '''
# insert module at a pretrained model
# '''

# # hard coding
# model = resnet50(pretrained=False)
# feats = list(model.features.children())

# feats.insert(8, nn.Identity())
# model.features = nn.Sequential(feats)

# #using hook

# model = resnet50()

###############################################################################################
'''
method to call all layers about model
'''

from base_resnet import resnet18

model = resnet18()

def insert_sigmoid(_, input, output):
    output = nn.Sigmoid()(output)
    return output

def get_param(model):
    for name, param in model._modules.items():
        if isinstance(param, nn.Sequential):
            get_param(param)
        else:
            for name2, param2 in param._modules.items():
                if name2 == 'conv2':
                    param2.register_forward_hook(insert_sigmoid)

get_param(model)

x = torch.ones([1,3,128, 128]).float()
output = model(x)
            
