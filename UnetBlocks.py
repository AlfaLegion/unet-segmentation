import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchsummary import summary
import Loss
import DataLoader
from torch.autograd import Variable
import torchvision.utils as v_utils
from PIL import Image



class ConvBlock(nn.Module):
    def __init__(self,input_channels,out_channels,padding):
        super(ConvBlock,self).__init__()

        self.conv_1=nn.Conv2d(input_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn_1=nn.BatchNorm2d(out_channels)
        self.relu_1=nn.ReLU(inplace=True)

        self.conv_2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn_2=nn.BatchNorm2d(out_channels)

    def forward(self, input):

        input=self.conv_1(input)
        input=self.bn_1(input)
        input=self.relu_1(input)
        input=self.conv_2(input)
        input=self.bn_2(input)

        return input

class Bottleneck(nn.Module):
    def __init__(self,input_channels,out_channels,padding):
        super(Bottleneck,self).__init__()

        self.conv_1=nn.Conv2d(input_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn_1=nn.BatchNorm2d(out_channels)
        self.relu_1=nn.ReLU(inplace=True)

        self.conv_2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn_2=nn.BatchNorm2d(out_channels)

    def forward(self,input):
        input=self.conv_1(input)
        input=self.bn_1(input)
        input=self.relu_1(input)
        input=self.conv_2(input)
        input=self.bn_2(input)
        
        return input

class DeconvBnRelu(nn.Module):
    def __init__(self,input_channels,out_channels,padding,out_padding):
        super(DeconvBnRelu,self).__init__()

        self.deconv1=nn.ConvTranspose2d(input_channels,out_channels,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu1=nn.ReLU(inplace=True)
    
    def forward(self,input):

        input=self.deconv1(input)
        input=self.bn1(input)
        input=self.relu1(input)
        return input
