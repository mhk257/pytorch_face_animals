'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from att_modules import PAM_Module
from CGNet import ConvBNPReLU, ChannelWiseConv, ChannelWiseDilatedConv, BNPReLU, FGlo
from deform_conv import DeformConv2D

# from .preresnet import BasicBlock, Bottleneck

__all__ = ['HourglassNet', 'hg']


# # -----------------HPM 
# class Bottleneck(nn.Module):

#     expansion = 2
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
        
#         """
#         args:
#            nIn: number of input channels : 256
#            nOut: number of output channels, : 256
#            add: if true, residual learning
#         """
#         super(Bottleneck, self).__init__()
#         #print('Revised bottleneck...')
#         self.relu = nn.ReLU(inplace=True)
        
        
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes // 2, kernel_size=3, stride=stride, padding=1, bias=True)

#         self.bn3 = nn.BatchNorm2d(planes // 2)
#         self.conv3 = nn.Conv2d(planes // 2, planes // 2, kernel_size=3, stride=stride, padding=1, bias=True)
        
#         self.downsample = downsample
#         self.stride = stride
        
#     def forward(self, input):
#         residual = input

#         output1 = self.bn1(input)
#         output1 = self.relu(output1)
#         output1 = self.conv1(output1)

#         output2 = self.bn2(output1)
#         output2 = self.relu(output2)
#         output2 = self.conv2(output2)

#         output3 = self.bn3(output2)
#         output3 = self.relu(output3)
#         output3 = self.conv3(output3)

#         output = torch.cat([output1, output2, output3], 1)

#         if self.downsample is not None:
            
#             residual = self.downsample(input)
#             #print('downsample...{}'.format(residual.size()))  
        
#         output  = output + residual

#         return output



# #-------------------CG based
# class Bottleneck(nn.Module):

#     expansion = 2
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
        
#         """
#         args:
#            nIn: number of input channels : 256
#            nOut: number of output channels, : 256
#            add: if true, residual learning
#         """
#         super(Bottleneck, self).__init__()
#         #print('Revised bottleneck...')
#         self.relu = nn.ReLU(inplace=True)
        
        
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True) # reduce computations

#         self.bnloc = nn.BatchNorm2d(planes)
#         self.F_loc = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

#         self.bnsur = nn.BatchNorm2d(planes)
#         self.F_sur = ChannelWiseDilatedConv(planes, planes, 3, 1, 2) # surrounding context
        
#         self.downsample = downsample
#         self.stride = stride
#         self.F_glo= FGlo(planes*2, 16)

#     def forward(self, input):
#         residual = input

#         output = self.bn1(input)
#         output = self.relu(output)
#         output = self.conv1(output)

#         loc = self.bnloc(output)
#         loc = self.relu(loc)
#         loc = self.F_loc(loc)

#         sur = self.bnsur(output)
#         sur = self.relu(sur)
#         sur = self.F_sur(sur)

#         output = torch.cat([loc, sur], 1)

#         # print(residual.size())
#         # print(output.size())
#         output = self.F_glo(output)  #F_glo is employed to refine the joint feature
#         # if residual version
#         if self.downsample is not None:
            
#             residual = self.downsample(input)
#             #print('downsample...{}'.format(residual.size()))  
        
#         output  = output + residual

#         return output
       


##-------------------Original one
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

       

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        
        if self.downsample is not None:
            
            residual = self.downsample(x)
            #print('downsample...{}'.format(residual.size()))
            

        #print(x.size())    
        

        out += residual

        return out

##-------------------Deformable one
# class Bottleneck(nn.Module):
#     expansion = 2

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()

#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.offsets = nn.Conv2d(planes, 18, kernel_size=3, padding=1, bias=True)
#         self.conv2 = DeformConv2D(planes, planes, kernel_size=3, padding=1, bias=True)
        
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

       

#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)

#         out = self.bn2(out)
#         out = self.relu(out)
        
#         offsets = self.offsets(out)
#         out = self.conv2(out, offsets)

#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.conv3(out)

        
#         if self.downsample is not None:
            
#             residual = self.downsample(x)
#             #print('downsample...{}'.format(residual.size()))
            

#         #print(x.size())    
        

#         out += residual

#         return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=68):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, fc2_, score_ = [], [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4)) # change this to 3 for no. of downsampling steps..
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc2_.append(self._make_fc2(num_classes, ch))

            
        #self.sa = PAM_Module(ch)

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
        self.fc2_ = nn.ModuleList(fc2_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def _make_fc2(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(outplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    # def _make_fc2(self, inplanes, outplanes):
    #     bn = nn.BatchNorm2d(inplanes)
    #     conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
    #     return nn.Sequential(
    #             conv,
    #             bn,
    #             self.relu,
    #         )

    def forward(self, x):
        out = []
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)

            

            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)

            # if i == 0:

            #     y = self.fc[i](y)
            #     score = self.score[i](y)
            #     out.append(score)
            # else:
            #     prev_prediction = out[i-1] # this is 68 C
            #     prev_prediction = self.fc2_[i-1](prev_prediction) # 68C -> 256C
            #     y = y + prev_prediction
                
            #     y = self.fc[i](y)
            #     score = self.score[i](y)
            #     out.append(score)


            
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
