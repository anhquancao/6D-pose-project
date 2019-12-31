from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random

def log_diff(log_yi, yi_true):
    return log_yi - torch.log(yi_true)

class LossDepth(_Loss):

    def __init__(self):
        super(LossDepth, self).__init__(True)
    
    def forward(self, pred_log_depth, true_depth):
        mask_depth = true_depth.eq(0.).float()
        
        di = log_diff(pred_log_depth, true_depth + mask_depth)
        
        di *= (1.0 - mask_depth)
        
        sum_di_square = torch.sum(di ** 2)
        square_sum_di = torch.sum(di) ** 2
        
        n = torch.sum(1 - mask_depth)
        
        return 1/n * sum_di_square - 0.5 / (n * n) * square_sum_di
