import torch

cpt_path = './experiments/danet_d1/runs/nyud/danet_d/resnet50/default/model_best.pth.tar'
cpt = torch.load(cpt_path, map_location=torch.device('cpu'))

params = cpt['state_dict']
params['head.sa.lamb']
params['head.sa.gamma']


a = torch.tensor([-0.0, -1.8, -3.6, -5.5, -7.3])
b = torch.tensor([2, 2, 2, 5, 5])


import numpy as np
a = np.array([[1,2,3], [4, 5, 6]])
b = np.array([[0.1, 0.2, 0.1], [0.1, 0.3, 0.1]])
a*b[:,1:2]


ck = np.array([[[0.1,0.2,0.3,0.4], [0.25, 0.25, 0.25, 0.25]]])  # [B, c, L, kk]: by channel attention
#[1, 2, 4]   c=2

x_unfold = np.array([[ [[2, 2, 3, 3],[5, 5, 4, 4]], [[1, 1, 2, 2],[6, 6, 6, 6]],  ]])     # [B, m, C/m, L, kk]
# [ 1, 1, 2, 4] m =1

out = ck * x_unfold

import torch
from torch import nn

B =1
c =3
h =3
w =4
Q = torch.randint(0, 5, (1, 3, 3, 4))
print(Q)
Q = Q.view(B, c, -1)  #[B, c, hw]
Q = Q.permute((0, 2, 1)) #[B, hw, c]
Q = Q.view(1, 12, 1, 3)  #[1, 12, 1, 3] [B, hw , 1, c]
Q = Q.double()
Q2 = Q.view(B, -1, h*w)

unfold = nn.Unfold(kernel_size=3, padding=(1,1))
K = torch.randint(0, 5, (1, 3, 3, 4))
K = K.double()
output = unfold(K)  # [1, 27, 12]

output = output.permute(0, 2, 1)  #[1, 12, 27]
output = output.reshape((1, 12, 3, -1))  #[1, 12, 3, 9]  [B, hw, c, kk]


O = torch.matmul(Q, output)



Q = torch.randint(0, 5, (1, 1, 3, 4)).double()  #[1, 1, 3, 4]
Q = Q.view(1, 1, -1)
Q = Q.permute(0, 2, 1)   #[1, 12, 1]
K = torch.randint(0, 5, (1, 1, 3, 4)).double()
unfold = nn.Unfold(kernel_size=3, padding=(1,1))
Key = unfold(K)  # [1, 9, 12]
Key = Key.permute(0, 2, 1)


Q = torch.randint(0, 5, (1,  3, 4)).double()
P = torch.randint(0, 5, (2,  3, 4)).double()

Q*P

Q = torch.randint(0, 5, (1, 3, 1)).double()


import numpy as np
a = np.random.randint(6,size=(4,5,3))
idx = np.nonzero(a)



import torch

Q = torch.randint(0, 5, (1, 1, 12)).float()
K = torch.randint(0, 5, (1, 12, 1)).float()

torch.nn.Dropout

import torchvision.models.detection.faster_rcnn

import torchvision.models as models
models.resnet34()

models.resnet50()


import torch
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')

torch.utils.model_zoo.load_url()


import torch.nn as nn

modules = [nn.Linear(10, 10), nn.Linear(10, 10)]
modules.append(nn.Linear(10, 10))
modules.append(nn.Linear(10, 10))

sequential = nn.Sequential(*modules)



model = torchvision.models.resnet34(pretrained=True)
model = torchvision.models.mobilenet_v2(pretrained=True)



import torch.nn
import copy

l = torch.nn.Linear(3,1)
c = copy.deepcopy(l)
print([type(p) for p in l.parameters()])
print([type(p) for p in c.parameters()])

