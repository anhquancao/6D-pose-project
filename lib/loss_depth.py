from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random

def im_grad(img):
    fx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3).float().cuda()
    fx_conv = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1)
    fx_conv.weight = nn.Parameter(fx)

    fy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3).float().cuda()
    fy_conv = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1)
    fy_conv.weight = nn.Parameter(fy)

    img_mean = torch.mean(img, 1, True)
    grad_x = fx_conv(img_mean)
    grad_y = fy_conv(img_mean)
    
    return grad_x, grad_y


class GradientLoss(_Loss):
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, pred_grad, true_grad):
        mask_depth = true_depth.eq(0.)
        pred_depth = torch.exp(pred_log_depth)
        true_depth[mask_depth] = pred_depth # use the predicted value to fill in the missing values
        
        grad_true_depth_x, grad_true_depth_y = im_grad(true_depth)
        grad_pred_depth_x, grad_pred_depth_y = im_grad(pred_depth)
        
        return torch.mean((grad_true_depth_x - grad_pred_depth_x) ** 2 + (grad_true_depth_y - grad_pred_depth_y) ** 2)
        
        
class NormalLoss(_Loss):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward(self):
        mask_depth = true_depth.eq(0.)
        pred_depth = torch.exp(pred_log_depth)
        true_depth[mask_depth] = pred_depth # use the predicted value to fill in the missing values
        
        grad_true_depth_x, grad_true_depth_y = im_grad(true_depth)
        grad_pred_depth_x, grad_pred_depth_y = im_grad(pred_depth)
        
        

class LogL2(_Loss):

    def __init__(self):
        super(LogL2, self).__init__()
    
    def forward(self, pred_log_depth, true_depth):
        bs = pred_log_depth.shape[0]

        mask_depth = true_depth.eq(0.).float()
        log_true_depth = torch.log(true_depth + mask_depth)
        
        diff = pred_log_depth - log_true_depth
        diff *= (1.0 - mask_depth)
        
        n = torch.sum(1 - mask_depth)
        
        res = torch.sqrt(torch.sum(diff ** 2)) / n
        
        return res / bs

class L2(_Loss):
    def __init__(self):
        super(L2, self).__init__()
        
    def forward(self, pred_log_depth, true_depth):
        bs = pred_log_depth.shape[0]
        
        mask_depth = true_depth.eq(0.).float()
        pred_depth = torch.exp(pred_log_depth)
       
        diff  = 10 * (pred_depth - true_depth) 

        diff *= (1.0 - mask_depth)
        
        n = torch.sum(1 - mask_depth)
        
        
        return torch.sqrt(torch.sum(diff ** 2)) / (n * bs)
