import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from torchvision import models
from collections import OrderedDict

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class UpProjBlock(nn.Module):
    """
    Deeper Depth Prediction with Fully Convolutional Residual Networks
    """
    # branch 1: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    # branch 2: 5*5 conv -> batchnorm

    def __init__(self, in_channels):
        super(UpProjBlock, self).__init__()
        out_channels = in_channels // 2
        self.unpool = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, groups=in_channels, stride=2)
        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)), 
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(out_channels))
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))
        
    def forward(self, x):
        x = self.unpool(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = x1 + x2
        return F.relu(x)

class DepthV2(nn.Module):
    def __init__(self, output_size, pretrained=True):
        super(DepthV2, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-2]
        decoder_out_channels = 2048
        
        self.conv = nn.Conv2d(decoder_out_channels, decoder_out_channels // 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(decoder_out_channels // 2)
        
        self.encoder = nn.Sequential(*modules)
        self.decoder = nn.Sequential(OrderedDict([
            ('up_proj_1', UpProjBlock(decoder_out_channels // 2)),
            ('up_proj_2', UpProjBlock(decoder_out_channels // 4)),
            ('up_proj_3', UpProjBlock(decoder_out_channels // 8)),
            ('up_proj_4', UpProjBlock(decoder_out_channels // 16))
        ]))
        
        self.predict_depth = nn.Conv2d(decoder_out_channels // 32, 1, kernel_size=3, padding=1)
        self.bilinear = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
        
        
    def forward(self, x):
        x = self.encoder(x)
        
        x = self.conv(x)
        x = self.bn(x)
        
        x = self.decoder(x)
        
        x = self.predict_depth(x)
        x = self.bilinear(x)
        
        return x
        

class DepthNetPSP(nn.Module):

    def __init__(self, usegpu=True):
        super(DepthNet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()

        self.model.final = nn.Conv2d(64, 1, 1)


    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetRGBOnly(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetRGBOnly, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.conv1 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)

        
        self.conv1_r = torch.nn.Conv1d(1024, 512, 1)
        self.conv1_t = torch.nn.Conv1d(1024, 512, 1)
        self.conv1_c = torch.nn.Conv1d(1024, 512, 1)

        self.conv2_r = torch.nn.Conv1d(512, 256, 1)
        self.conv2_t = torch.nn.Conv1d(512, 256, 1)
        self.conv2_c = torch.nn.Conv1d(512, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj
    
    def forward(self, img, choose, obj):
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()
        
        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        
        emb = self.conv1(emb)
        emb = self.conv2(emb)
        emb = self.conv3(emb)
        emb = self.conv4(emb)
        emb = self.conv5(emb)
        
        rx = F.relu(self.conv1_r(emb))
        tx = F.relu(self.conv1_t(emb))
        cx = F.relu(self.conv1_c(emb))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        return out_rx, out_tx, out_cx, emb.detach()
        

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
        
    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

    

        
    
class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        
        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        return out_rx, out_tx, out_cx, emb.detach()
 

class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
