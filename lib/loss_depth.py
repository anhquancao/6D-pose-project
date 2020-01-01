from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random

class LogL2(_Loss):

    def __init__(self):
        super(LogL2, self).__init__(True)
    
    def forward(self, pred_log_depth, true_depth):
        mask_depth = true_depth.eq(0.).float()
        log_true_depth = torch.log(true_depth + mask_depth)
        
        diff = pred_log_depth - log_true_depth
        diff *= (1.0 - mask_depth)
        
        n = torch.sum(1 - mask_depth)
        
        return torch.sqrt(torch.sum(diff ** 2)) / n

class L2(_Loss):
    def __init__(self):
        super(L2, self).__init__(True)
        
    def forward(self, pred_log_depth, true_depth):
        mask_depth = true_depth.eq(0.).float()
        pred_depth = torch.exp(pred_log_depth)
        
        diff  = (pred_depth - true_depth) 
        diff *= (1.0 - mask_depth)
        
        n = torch.sum(1 - mask_depth)
        
        return torch.sqrt(torch.sum(diff ** 2)) / n
