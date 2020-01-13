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

    def __init__(self, in_channels, out_channels):
        super(UpProjBlock, self).__init__()
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

class UpProjBlockv2(nn.Module):
    """
    Deeper Depth Prediction with Fully Convolutional Residual Networks
    """
    # branch 1: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    # branch 2: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels):
        super(UpProjBlockv2, self).__init__()
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
        
    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = x1 + x2
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.pool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        block1 = self.layer1(x)
        block2 = self.layer2(block1)
        block3 = self.layer3(block2)
        block4 = self.layer4(block3)
        
        return block1, block2, block3, block4
    
class R(nn.Module):
    def __init__(self, output_channel=1, output_size=(480, 640)):
        super(R, self).__init__()
        num_features = 64 + 2048 // 32
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features)
        
        self.conv3 = nn.Conv2d(num_features, output_channel, kernel_size=5, padding=2, bias=True)
        self.bilinear = nn.Upsample(size=(480, 640), mode='bilinear', align_corners=True)
        
        
        
    def forward(self, x):
        x = self.bilinear(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        
        return x
    
class MFF(nn.Module):
    def __init__(self):
        super(MFF, self).__init__()
        self.up1 = UpProjBlockv2(in_channels=256, out_channels=16)
        self.up2 = UpProjBlockv2(in_channels=512, out_channels=16)
        self.up3 = UpProjBlockv2(in_channels=1024, out_channels=16)
        self.up4 = UpProjBlockv2(in_channels=2048, out_channels=16)
        
        self.conv = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        
    def forward(self, block1, block2, block3, block4, size):
        
        m1 = self.up1(block1, size)
        
        m2 = self.up2(block2, size)
        
        m3 = self.up3(block3, size)
        
        m4 = self.up4(block4, size)
        
        x = torch.cat([m1, m2, m3, m4], 1)
        
        
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_features=2048):
        super(Decoder, self).__init__()
        self.up1 = UpProjBlockv2(num_features // 2, num_features // 4)
        self.up2 = UpProjBlockv2(num_features // 4, num_features // 8)
        self.up3 = UpProjBlockv2(num_features // 8, num_features // 16)
        self.up4 = UpProjBlockv2(num_features // 16, num_features // 32)
        self.bn = nn.BatchNorm2d(num_features // 2)
        self.conv = nn.Conv2d(num_features, num_features // 2, kernel_size=1, bias=False)
        
    def forward(self, block1, block2, block3, block4):
        x =  F.relu(self.bn(self.conv(block4)))
        x = self.up1(x, [block3.shape[2], block3.shape[3]])
        
        x = self.up2(x, [block2.shape[2], block2.shape[3]])
        
        x = self.up3(x, [block1.shape[2], block1.shape[3]])
        
        x = self.up4(x, [block1.shape[2] * 2, block1.shape[3] * 2])
        
        return x
    
class DepthV3(nn.Module):
    def __init__(self, pretrained=True, output_channel=1, output_size=(480, 640)):
        super(DepthV3, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.encoder = Encoder()
        
        self.decoder = Decoder()
        
        self.MFF = MFF()
        self.R = R(output_channel=output_channel, output_size=output_size)
        
        self.up = UpProjBlockv2(32, 32)
        
        
    def forward(self, x):
        block1, block2, block3, block4 = self.encoder(x)
        x_decoded = self.decoder(block1, block2, block3, block4)
        x_mff = self.MFF(block1, block2, block3, block4, [x_decoded.shape[2], x_decoded.shape[3]])
        y = self.R(torch.cat([x_decoded, x_mff], 1))
        
        return y
    
class R2(nn.Module):
    def __init__(self, output_channel=1, output_size=(480, 640)):
        super(R2, self).__init__()
        num_features = 64 + 2048 // 32
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features)
        
        self.conv3 = nn.Conv2d(64 + 2048 // 32, 32, kernel_size=5, padding=2, bias=True)
        
        
    def forward(self, x):
#         x = self.bilinear(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        
        return x
    
class DepthV4(nn.Module):
    def __init__(self, pretrained=True, output_channel=1, output_size=(480, 640)):
        super(DepthV4, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.encoder = Encoder()
        
        self.decoder = Decoder()
        
        self.MFF = MFF()
        self.R = R(output_channel=output_channel, output_size=output_size)
        
        self.up = UpProjBlockv2(32, 32)
        
        
    def forward(self, x):
        block1, block2, block3, block4 = self.encoder(x)
        x_decoded = self.decoder(block1, block2, block3, block4)
        x_mff = self.MFF(block1, block2, block3, block4, [x_decoded.shape[2], x_decoded.shape[3]])
        y = self.R(torch.cat([x_decoded, x_mff], 1))
        
        return self.up(y, [x.shape[2], x.shape[3]])

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
            ('up_proj_1', UpProjBlock(decoder_out_channels // 2, decoder_out_channels // 4)),
            ('up_proj_2', UpProjBlock(decoder_out_channels // 4, decoder_out_channels // 8)),
            ('up_proj_3', UpProjBlock(decoder_out_channels // 8, decoder_out_channels // 16)),
            ('up_proj_4', UpProjBlock(decoder_out_channels // 16, decoder_out_channels // 32))
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
        super(DepthNetPSP, self).__init__()

        self.model = psp_models['resnet18'.lower()]()

        self.model.final = nn.Conv2d(64, 1, 1)


    def forward(self, x):
        x = self.model(x)
        return x

class ConfNet(nn.Module):
    def __init__(self, num_obj):
        super(ConfNet, self).__init__()
       
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 32, 1)
        self.conv5 = torch.nn.Conv1d(32, 8, 1)
        
        self.fc1 = nn.Linear(4000, 1024)
        self.fc2 = nn.Linear(1024, 500)
        self.dropout = nn.Dropout()
        
    def forward(self, emb):
        emb = F.relu(self.conv1(emb))
        
        emb = F.relu(self.conv2(emb))
        emb = F.relu(self.conv3(emb))
        emb = F.relu(self.conv4(emb))
        emb = F.relu(self.conv5(emb))

        emb = emb.view(emb.size(0), -1)

        emb = F.relu(self.fc1(emb))
        emb = self.dropout(emb)
        emb = F.relu(self.fc2(emb))
        return emb
        
class PoseNetRGBOnlyV2(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetRGBOnlyV2, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.cnn = DepthV4(output_channel=32)
        
        self.conv1 = torch.nn.Conv1d(32, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 1024, 1)
        
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

        
    
    def forward(self, img, choose, obj):

        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)

        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        emb = F.relu(self.conv1(emb))
        
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

    
class PoseNetRGBOnly(nn.Module):
    def __init__(self, num_points, num_obj, model="psp"):
        super(PoseNetRGBOnly, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        if model == 'psp':
            self.cnn = ModifiedResnet()
        if model == 'depthv4':
            self.cnn = DepthV4(output_channel=32)

            

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

        
    
    def forward(self, img, choose, obj):
#         print(img.shape)
        out_img = self.cnn(img)
#         print(out_img.shape)
        bs, di, _, _ = out_img.size()
#         print('===========')
#         print(out_img.shape)
        emb = out_img.view(bs, di, -1)
#         print(emb.shape)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
#         print(emb.shape)
        emb = F.relu(self.conv1(emb))
        emb = F.relu(self.conv2(emb))
        emb = F.relu(self.conv3(emb))
        emb = F.relu(self.conv4(emb))
        emb = F.relu(self.conv5(emb))
        
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
