import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from base import BaseModel
# from shifted_model import *
from model.modules import *

class MLPHead(BaseModel):
    def __init__(self, input_size=8, hidden_sizes=[4], output_size=2, reshape=None, freeze=False):
        super(MLPHead, self).__init__()
        layer_sizes = [input_size] + hidden_sizes
        self.layers = nn.ModuleList()
        self.reshape = reshape

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.last_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.output_size = output_size
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = nn.ReLU()(x)
        x = self.last_layer(x)
        if self.reshape == 'CVR':
            x = nn.functional.normalize(x, dim=1)
            x = x.reshape([-1, 4, self.output_size])
            x = (x[:,:,None,:] * x[:,None,:,:]).sum(3).sum(2)
            x = -x
        return x
    
class IdentityHead(BaseModel):
    def __init__(self, reshape=None, *args, **kwargs):
        super(IdentityHead, self).__init__(*args, **kwargs)
        self.reshape = reshape

    def forward(self, x):
        if self.reshape == 'CVR':
            x = nn.functional.normalize(x, dim=1)
            x = x.reshape([-1, 4, 8])
            x = (x[:,:,None,:] * x[:,None,:,:]).sum(3).sum(2)
            x = -x
        return x

class GPTHead(BaseModel):
    def __init__(self, input_size=8, hidden_sizes=[4], output_size=2, reshape=None, freeze=False):
        super(GPTHead, self).__init__()
        # self.fc1 = nn.Linear(58, 1)     # CVR
        # self.fc1 = nn.Linear(60, 1)     # SVRT
        self.fc1 = nn.Linear(input_size, 1)
        self.fc2 = nn.Linear(32001, output_size)
        self.input_size = input_size
        self.reshape = reshape
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = x.view(-1, 32001)
        x = self.fc2(x)
        if self.reshape == 'CVR':
            x = nn.functional.normalize(x, dim=1)
            x = x.reshape([-1, 4, 8])
            x = (x[:,:,None,:] * x[:,None,:,:]).sum(3).sum(2)
            x = -x
        return x

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class RavenHead(BaseModel):
    def __init__(self, **kwargs):
        super(RavenHead, self).__init__()
        self.classmlp = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))

    def forward(self, x):
        x = self.classmlp(x)
        return x


class CvrHead(BaseModel):
    def __init__(
        self,
        mlp_dim: int = 8,
        **kwargs
    ):
        super(CvrHead, self).__init__()
        self.hidden_size = mlp_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classmlp = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, self.hidden_size))

    def forward(self, x):
        x = self.classmlp(x)
        x = nn.functional.normalize(x, dim=1)
        x = x.reshape([-1, 4, self.hidden_size])
        x = (x[:,:,None,:] * x[:,None,:,:]).sum(3).sum(2)
        x = -x
        return x
    
    def shared_step(self, batch):
        x, task_idx = batch # B, 4, H, W
        # creates artificial label
        x_size = x.shape
        perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0)
        y = perms.argmax(1)
        perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4
        perms = perms.flatten()
        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])
        y_hat = self(x)
        return y_hat, y
    
class SvrtHead(BaseModel):
    def __init__(
        self,
        **kwargs
    ):
        super(SvrtHead, self).__init__()
        self.classmlp = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))

    def forward(self, x):
        x_size = x.shape
        x = self.classmlp(x)
        return x

class BongardHead(BaseModel):
    def __init__(self, **kwargs):
        super(BongardHead,self).__init__()
        self.classmlp = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4,2))
    
    def forward(self, x):
        x = self.classmlp(x)
        return x


class CFDecoder(BaseModel):
    def __init__(self, state_size, z_size, keep_dim=True, freeze=False):
        super(CFDecoder, self).__init__()
        self.gn_decoder = GCN(in_size=z_size, out_size=state_size, n_layers=0, hidden_size=32)
        self.rnn_decoder = nn.GRU(input_size=z_size, hidden_size=z_size, num_layers=1, batch_first=True)
        self.keep_dim = keep_dim
        self.z_size = z_size
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if not self.keep_dim:
            x = x.view(-1, 150, 5, self.z_size)
        B, T, K, S = x.shape
        # print("[]")
        # print(x.shape)
        x = x.transpose(1, 2).reshape(B * K, T, -1)
        # print(x.shape)
        z, _ = self.rnn_decoder(x)
        # print(z.shape)
        z = z.view(B, K, T, -1).transpose(1, 2)
        # print(z.shape)
        x = self.gn_decoder(z.reshape(B * T, K, -1)).reshape(B, T, K, -1)
        # print("en")
        x = x[:, :, :4, :]
        # print(x.shape)
        # print("outhead",x.shape)
        return x