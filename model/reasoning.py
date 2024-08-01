import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from base import BaseModel
import re
from PIL import Image
from typing import Tuple
from torchvision import transforms
import numpy as np
from minigpt4.models.mini_gpt4 import MiniGPT4


class MLP(BaseModel):
    def __init__(self, input_size=512, hidden_sizes=[128], output_size=8, freeze=False):
        super(MLP, self).__init__()
        layer_sizes = [input_size] + hidden_sizes
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.last_layer = nn.Linear(hidden_sizes[-1], output_size)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = nn.ReLU()(x)
        x = self.last_layer(x)
        return x

class ResNet18Reason(BaseModel):
    def __init__(self, num_channels=3, pretrained=False, freeze=False, dropout=0.0, keep_dim=False, state=0, reshape=None):
        super(ResNet18Reason, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained, progress=False, num_classes=1000)
        
        self.backbone.fc = nn.Identity()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.keep_dim = keep_dim
        self.state = state
        self.reshape = reshape

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.state < 4:
            x = self.dropout1(x)
            x = self.backbone.layer1(x)
        if self.state < 3:
            x = self.backbone.layer2(x)
        if self.state < 2:
            x = self.backbone.layer3(x)
        if self.state < 1:
            x = self.backbone.layer4(x)

        x = self.dropout2(x)

        x = self.backbone.avgpool(x)
        if not self.keep_dim:
            x = torch.flatten(x, 1)

        x = self.backbone.fc(x)
        return x
    

class CNN(BaseModel):
    def __init__(self, num_classes=8, pretrained=False, freeze=False):
        super(CNN, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained, progress=False, num_classes=1000).layer4
        self.linear = nn.Linear(512, num_classes)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class Transformer(BaseModel):
    def __init__(self, input_size, output_size, nhead=4, num_layers=6):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(input_size, output_size)
    
    def forward(self, src):
        output = self.encoder(src)
        output = self.decoder(output)
        return output
    
class Identity(BaseModel):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency_matrix, features):

        aggregate = torch.matmul(adjacency_matrix, features)
        aggregate = aggregate / (adjacency_matrix.sum(dim=1, keepdim=True) + 1e-8)  # 防止除零错误
        output = self.linear(aggregate)      
        return output

class GNN(BaseModel):
    def __init__(self, num_nodes, num_features=512, hidden_dim=128, num_classes=8):
        super(GNN, self).__init__()
        
        self.gc1 = GraphConvLayer(num_features, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, num_classes)
        self.adjacency_matrix = torch.randn(num_nodes, num_nodes)
        
    def forward(self, features):

        hidden = self.gc1(self.adjacency_matrix, features)
        hidden = torch.relu(hidden)  
        output = self.gc2(self.adjacency_matrix, hidden)
        
        return output
    
    def set_device(self, device):
        self.adjacency_matrix = self.adjacency_matrix.to(device)

class GCN_MLP(nn.Module):
    def __init__(self, in_size, h_size, out_size, n_layers, activation=nn.ReLU, batchnorm=False, n_channel=4):
        super(GCN_MLP, self).__init__()

        layer_list = []
        if batchnorm:
            layer_list.append(nn.BatchNorm1d(n_channel))

        layer_list.append(nn.Linear(in_size, h_size))
        layer_list.append(activation())
        for _ in range(n_layers):
            layer_list.append(nn.Linear(h_size, h_size))
            layer_list.append(activation())

        layer_list.append(nn.Linear(h_size, out_size))

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


class GCN(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, n_layers=0, activation=nn.ReLU, batchnorm=False):
        super(GCN, self).__init__()

        self.f = GCN_MLP(2 * input_size, hidden_size, hidden_size, n_layers, activation, batchnorm, n_channel=2 * input_size)
        self.g = GCN_MLP(2 * hidden_size + input_size, hidden_size, output_size, n_layers, activation, batchnorm, n_channel=4)

    def forward(self, x):
        x_size = x.shape 
        if (len(x_size) == 2):
            x = x.unsqueeze(1)
        if (len(x_size) == 4):
            B, C, H, W = x_size
            x = x.view(B, -1, C)
        B, K, S = x.shape

        x1 = x.unsqueeze(1).repeat(1, K, 1, 1)
        x2 = x.unsqueeze(2).repeat(1, 1, K, 1)

        x12 = torch.cat([x1, x2], dim=-1)
        interactions = self.f(x12.view(B * K * K, 2 * S)).view(B, K, K, -1)
        E = interactions * (1 - torch.eye(K).view(1, K, K, 1).repeat(B, 1, 1, interactions.shape[-1]).to(x.device))

        E = E.sum(2)
        E_sum = E.sum(1).unsqueeze(1).repeat(1, K, 1)

        out = self.g(torch.cat([x, E, E_sum], dim=-1))
        out = out.view(out.shape[0], -1)

        return out

class EBM(BaseModel):
    def __init__(self, input_size, hidden_size, repeat_size):
        super(EBM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, input_size)
        self.repeat_size = repeat_size

    def forward(self, data, target):
        target = target.unsqueeze(1).repeat(1, self.repeat_size)
        x = torch.cat((data, target), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output_vector = self.energy_layer(x)
        return torch.norm(output_vector, dim=1)  

def gumbel_softmax(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
    logits_with_noise = (logits + gumbel_noise) / temperature
    softmax_output = F.softmax(logits_with_noise, dim=-1)
    return softmax_output

class PCNN(BaseModel):
    def __init__(self, input_size, output_size, hidden_size=128, threshold=0, max_iterations=10):
        super(PCNN, self).__init__()

        self.threshold = threshold
        self.max_iterations = max_iterations
        self.fc_pre = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc_pre(x)
        row_means = torch.mean(x, dim=1, keepdim=True)
        row_stds = torch.std(x, dim=1, keepdim=True)
        normalized_x = (x - row_means) / row_stds
        input_layer = gumbel_softmax(normalized_x, 0.5)
        pulse_coupling_layer = input_layer
        output_layer = self.fc(pulse_coupling_layer)

        return output_layer