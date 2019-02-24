###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'MRATT_Module']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        
        self.out_conv = Conv2d(in_channels=in_dim//4, out_channels=in_dim, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        #print('Channels {} Height {}, Width {} '.format(C, height, width))
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key).view(m_batchsize, -1) # flattening for softmax N x ((HxW)x(HxW))

        #print('Energy Size after1 {}'.format(energy.size()))
        
        attention = self.softmax(energy).view(m_batchsize, -1, width*height) # reshaping back to N x (HxW)x(HxW)

        #print('Energy Size after2 {}'.format(attention.size()))

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        #print('Proj. value size {}'.format(proj_value.size()))

        #out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        out = torch.bmm(proj_value, attention)

        _, C_in, _, = out.size()
        
        out = out.view(m_batchsize, C_in, height, width)

        out = self.out_conv(out)

        #print('Gamma_value--{}'.format(self.gamma))

        out = (self.gamma.expand_as(out) * out) + x
        
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class MRATT_Module(Module):
    """ MRATT Module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(MRATT_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.upsample = nn.UpsamplingNearest2d(size=(4096, 4096))
      

        self.softmax = Softmax()
    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        #print('Channels {} Height {}, Width {} '.format(C, height, width))
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key).view(m_batchsize, -1) # flattening for softmax N x ((HxW)x(HxW))

        #print('Energy Size after1 {}'.format(energy.size()))
        
        attention = self.softmax(energy).view(m_batchsize, -1, width*height) # reshaping back to N x (HxW)x(HxW)

        if height < 64:

            attention = attention.unsqueeze(1)

            attention = self.upsample(attention)

            attention = attention.squeeze(1)
                        
        return attention

