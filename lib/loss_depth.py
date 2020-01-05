from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F

def im_grad(img):
    fx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3).float().cuda()
    fx_conv = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1)
    fx_conv.weight = nn.Parameter(fx)
    for param in fx_conv.parameters():
        param.requires_grad = False

    fy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3).float().cuda()
    fy_conv = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1)
    fy_conv.weight = nn.Parameter(fy)
    for param in fy_conv.parameters():
        param.requires_grad = False

    img_mean = torch.mean(img, 1, True)
    grad_x = fx_conv(img_mean)
    grad_y = fy_conv(img_mean)
    
    return grad_x, grad_y


class GradientLoss(_Loss):
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, pred_depth, true_depth):
        grad_true_depth_x, grad_true_depth_y = im_grad(true_depth)
        grad_pred_depth_x, grad_pred_depth_y = im_grad(pred_depth)
        
        loss_dx = torch.abs(grad_true_depth_x - grad_pred_depth_x).mean()
        loss_dy = torch.abs(grad_true_depth_y - grad_pred_depth_y).mean()
        
        return loss_dx + loss_dy
        
        
class NormalLoss(_Loss):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward(self, pred_depth, true_depth):
        bs, c, h, w = true_depth.shape
        
        grad_true_depth_x, grad_true_depth_y = im_grad(true_depth)
        grad_pred_depth_x, grad_pred_depth_y = im_grad(pred_depth)
        
        ones = torch.ones(bs, 1, h, w).float().cuda()
        ones = torch.autograd.Variable(ones)
        
        true_depth_normal = torch.cat((-grad_true_depth_x, -grad_true_depth_y, ones), 1)
        pred_normal = torch.cat((-grad_pred_depth_x, -grad_pred_depth_y, ones), 1)
        
        cos = nn.CosineSimilarity(dim=1, eps=0)
        return torch.abs(1 - cos(pred_normal, true_depth_normal)).mean()
              

class L2Loss(_Loss):

    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, pred_depth, true_depth):
        bs, c, h, w = true_depth.shape
        
        return torch.log(torch.abs(pred_depth - true_depth) + 0.5).mean()

class BerHu(_Loss):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold
    
    def forward(self, pred_depth, true_depth):
        diff = torch.abs(true_depth-pred_depth)

        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.mean(loss)
        return loss
    
class LogL2(_Loss):

    def __init__(self):
        super(LogL2, self).__init__()
    
    def forward(self, pred_log_depth, true_depth):
        bs, c, h, w = true_depth.shape
        
        log_true_depth = torch.log(true_depth)
        
        diff = pred_log_depth - log_true_depth
    
        
        res = torch.sqrt(torch.sum(diff ** 2)) / (h * w)
        
        return res