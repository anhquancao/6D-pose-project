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
    
    def forward(self, pred_log_depth, true_depth):
        pred_depth = torch.exp(pred_log_depth)
        
        grad_true_depth_x, grad_true_depth_y = im_grad(true_depth)
        grad_pred_depth_x, grad_pred_depth_y = im_grad(pred_depth)
        
        loss_dx = torch.log(torch.abs(grad_true_depth_x - grad_pred_depth_x) + 0.5).mean()
        loss_dy = torch.log(torch.abs(grad_true_depth_y - grad_pred_depth_y) + 0.5).mean()
        
        return loss_dx + loss_dy
        
        
class NormalLoss(_Loss):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward(self, pred_log_depth, true_depth):
        bs, c, h, w = true_depth.shape
        
        pred_depth = torch.exp(pred_log_depth)
        
        grad_true_depth_x, grad_true_depth_y = im_grad(true_depth)
        grad_pred_depth_x, grad_pred_depth_y = im_grad(pred_depth)
        
        ones = torch.ones(bs, 1, h, w).float().cuda()
        ones = torch.autograd.Variable(ones)
        
        true_depth_normal = torch.cat((-grad_true_depth_x, -grad_true_depth_y, ones), 1)
        pred_normal = torch.cat((-grad_pred_depth_x, -grad_pred_depth_y, ones), 1)
        
        cos = nn.CosineSimilarity(dim=1, eps=0)
        return torch.abs(1 - cos(pred_normal, true_depth_normal)).mean()
              

class LogL2(_Loss):

    def __init__(self):
        super(LogL2, self).__init__()
    
    def forward(self, pred_log_depth, true_depth):
        bs, c, h, w = true_depth.shape

        pred_depth = torch.exp(pred_log_depth)
        
        diff = pred_depth - true_depth
        
        return torch.log(torch.abs(diff) + 0.5).mean()

