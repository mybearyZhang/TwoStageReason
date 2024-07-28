import torch
import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)

def mse_loss(output, target):
    return nn.MSELoss()(output, target)

def energy_loss(output, target):
    return nn.MSELoss()(output, torch.zeros_like(output))

def vqa_loss(output, target):
    return nn.CrossEntropyLoss()(output,target[0])