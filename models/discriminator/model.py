from re import X
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

# torchsummary and torchvision
# from torchsummary import summary
from torchvision.utils import save_image

# matplotlib stuff
import matplotlib.pyplot as plt
import matplotlib.image as img

# numpy and pandas
import numpy as np
import pandas as pd

# Common python packages
import datetime
import os
import sys
import time

## math
import math


CONCAT_FLAG = True  #False add   True concat
################## Discriminator ###########################

def discriminator_block(in_filters, out_filters):
    """Return downsampling layers of each discriminator block"""
    layers = [nn.Conv3d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers



def discriminator_block_new(in_filters, out_filters):
    """Return downsampling layers of each discriminator block"""
    layers = [nn.Conv3d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)
    return layers


class Discriminator64(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator64, self).__init__()
        layers = []
        layers.extend(discriminator_block(in_channels * 2, 32))
        layers.extend(discriminator_block(32, 64))
        layers.extend(discriminator_block(64, 128))
        layers.append(nn.Conv3d(128, 1, 4, padding=0))
        layers.append(nn.AvgPool3d(5))
        self.model = nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        layers = []
        layers.extend(discriminator_block(in_channels * 2, 64))
        layers.extend(discriminator_block(64, 128))
        layers.extend(discriminator_block(128, 256))
        layers.extend(discriminator_block(256, 512))
        layers.append(nn.Conv3d(512, 1, 4, padding=0))
        layers.append(nn.AvgPool3d(5))
        self.model = nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class DiscriminatorCycle(nn.Module):
    def __init__(self, in_channels=3):
        super(DiscriminatorCycle, self).__init__()

        layers = []
        layers.extend(discriminator_block(in_channels, 64))
        layers.extend(discriminator_block(64, 128))
        layers.extend(discriminator_block(128, 256))
        layers.extend(discriminator_block(256, 512))
        layers.append(nn.Conv3d(512, 1, 4, padding=0))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc,nhidden):
        super().__init__()
      
        self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
  

        ks = 5
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        # print("segmap:",segmap.shape)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        # print("gamma",gamma.shape)
        beta = self.mlp_beta(actv)

        # gamma = F.interpolate(gamma, size=x.size()[2:], mode='nearest')
        # beta = F.interpolate(beta, size=x.size()[2:], mode='nearest')

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class Spade_pp(nn.Module):
    def __init__(self, norm_nc, label_nc,nhidden):
        super().__init__()
      
        self.convs = nn.Sequential(
            nn.Conv3d(norm_nc,norm_nc,kernel_size=1+1*2, stride=1, padding=1),
            nn.BatchNorm3d(norm_nc),
            nn.ReLU(inplace=False)
        )
        self.convs_attention = nn.Sequential(
            nn.Conv3d(norm_nc,norm_nc,kernel_size=1+1*2, stride=1, padding=1),
        )

        self.softmax = nn.Softmax(dim=1)

        self.convs_ = nn.Sequential(
            nn.Conv3d(norm_nc,norm_nc,kernel_size=1+1*2, stride=1, padding=1),
            nn.BatchNorm3d(norm_nc),
            nn.ReLU(inplace=False)
        )
        self.convs_attention_ = nn.Sequential(
            nn.Conv3d(norm_nc,norm_nc,kernel_size=1+1*2, stride=1, padding=1),
        )

        self.softmax_ = nn.Softmax(dim=1)
        # self.spade = SPADE(norm_nc, label_nc,nhidden)
        self.spade = SPADE(norm_nc, label_nc,nhidden)
  


    def forward(self, input1, input2):

        add_wise = input1 + input2

        mul_wise = input1 * input2
        # add_wise = input1
        # mul_wise = input2

        # if CONCAT_FLAG:
        #     add_wise = torch.cat([input1, input2], 1)
        #     mul_wise = torch.cat([input1, input2], 1)


        fea = self.convs(add_wise)
        atten = self.softmax(self.convs_attention(add_wise))
        res = fea*atten

        fea_ = self.convs_(mul_wise)
        atten_ = self.softmax_(self.convs_attention_(mul_wise)) 
        res_ = fea_ * atten_

        # out = self.spade(res,res_)  
        out = res+res_  #消融实验测试注意力模块

        return out


class DiscriminatorCycle_new(nn.Module):
    def __init__(self, in_channels=1):
        super(DiscriminatorCycle_new, self).__init__()


        if CONCAT_FLAG:
            layers = []
            layers.append(discriminator_block_new(in_channels, 64))
            layers.append(discriminator_block_new(64*2, 128))
            layers.append(discriminator_block_new(128*2, 256))
            layers.append(discriminator_block_new(256*2, 512))
            self.model = nn.ModuleList(layers)

            layers_ = []
            layers_.append(discriminator_block_new(in_channels, 64))
            layers_.append(discriminator_block_new(64*2, 128))
            layers_.append(discriminator_block_new(128*2, 256))
            layers_.append(discriminator_block_new(256*2, 512))
            self.model_ = nn.ModuleList(layers_)
            self.final = nn.Conv3d(512*4, 1, 4, padding=0)

        else:
            layers = []
            layers.append(discriminator_block_new(in_channels, 64))
            layers.append(discriminator_block_new(64, 128))
            layers.append(discriminator_block_new(128, 256))
            layers.append(discriminator_block_new(256, 512))
            self.model = nn.ModuleList(layers)

            layers_ = []
            layers_.append(discriminator_block_new(in_channels, 64))
            layers_.append(discriminator_block_new(64, 128))
            layers_.append(discriminator_block_new(128, 256))
            layers_.append(discriminator_block_new(256, 512))
            self.model_ = nn.ModuleList(layers_)

            self.final = nn.Conv3d(512, 1, 4, padding=0)

        
        fusion = []
        fusion.append(Spade_pp(64,64,64))
        fusion.append(Spade_pp(128,128,128))
        fusion.append(Spade_pp(256,256,256))
        fusion.append(Spade_pp(512,512,512))
        # fusion.append(SPADE(64,64,64))
        # fusion.append(SPADE(128,128,128))
        # fusion.append(SPADE(256,256,256))
        # fusion.append(SPADE(512,512,512))



        self.fusion = nn.ModuleList(fusion)




    def forward(self, img1, img2):
        for i,(layer1,layer2, fusion_atten)  in enumerate(zip(self.model,self.model_, self.fusion)):
            if i ==0:
                temp1 = img1
                temp2 = img2
            temp1 = layer1(temp1)
            temp2 = layer2(temp2)
            fusion_res = fusion_atten(temp1,temp2)           
            # fusion_res = temp1+temp2

            # fusion_res = torch.cat([temp1, temp2],1)
            # nn.Conv3d(fusion_res,fusion_res,kernel_size=3, stride=1, padding=1),


            if CONCAT_FLAG:
                temp1 = torch.cat([temp1, fusion_res],1)
                temp2 = torch.cat([temp2, fusion_res],1)
            else:
                temp1 = temp1 + fusion_res
                temp2 = temp2 + fusion_res

        if CONCAT_FLAG:
            out = self.final(torch.cat([temp1,temp2],1))
        else:
            out = self.final(temp1+temp2)
        return out

class test(nn.Module):
    def __init__(self, in_channels=3):
        super(test, self).__init__()


        if CONCAT_FLAG:
            layers = []
            layers.append(discriminator_block_new(in_channels, 64))
            layers.append(discriminator_block_new(64*2, 128))
            layers.append(discriminator_block_new(128*2, 256))
            layers.append(discriminator_block_new(256*2, 512))
            self.model = nn.ModuleList(layers)

            layers_ = []
            layers_.append(discriminator_block_new(in_channels, 64))
            layers_.append(discriminator_block_new(64*2, 128))
            layers_.append(discriminator_block_new(128*2, 256))
            layers_.append(discriminator_block_new(256*2, 512))
            self.model_ = nn.ModuleList(layers_)
            self.final = nn.Conv3d(512*4, 1, 4, padding=0)

        else:
            layers = []
            layers.append(discriminator_block_new(in_channels, 64))
            layers.append(discriminator_block_new(64, 128))
            layers.append(discriminator_block_new(128, 256))
            layers.append(discriminator_block_new(256, 512))
            self.model = nn.ModuleList(layers)

            layers_ = []
            layers_.append(discriminator_block_new(in_channels, 64))
            layers_.append(discriminator_block_new(64, 128))
            layers_.append(discriminator_block_new(128, 256))
            layers_.append(discriminator_block_new(256, 512))
            self.model_ = nn.ModuleList(layers_)

            self.final = nn.Conv3d(512, 1, 4, padding=0)

        
        fusion = []
        fusion.append(Spade_pp(64,64,64))
        fusion.append(Spade_pp(128,128,128))
        fusion.append(Spade_pp(256,256,256))
        fusion.append(Spade_pp(512,512,512))
        # fusion.append(SPADE(64,64,64))
        # fusion.append(SPADE(128,128,128))
        # fusion.append(SPADE(256,256,256))
        # fusion.append(SPADE(512,512,512))



        self.fusion = nn.ModuleList(fusion)




    def forward(self, img):
        img1 = img
        img2 = img
        for i,(layer1,layer2, fusion_atten)  in enumerate(zip(self.model,self.model_, self.fusion)):
            if i ==0:
                temp1 = img1
                temp2 = img2
            temp1 = layer1(temp1)
            temp2 = layer2(temp2)
            fusion_res = fusion_atten(temp1,temp2)           
            # fusion_res = temp1+temp2

            # fusion_res = torch.cat([temp1, temp2],1)
            # nn.Conv3d(fusion_res,fusion_res,kernel_size=3, stride=1, padding=1),


            if CONCAT_FLAG:
                temp1 = torch.cat([temp1, fusion_res],1)
                temp2 = torch.cat([temp2, fusion_res],1)
            else:
                temp1 = temp1 + fusion_res
                temp2 = temp2 + fusion_res

        if CONCAT_FLAG:
            out = self.final(torch.cat([temp1,temp2],1))
        else:
            out = self.final(temp1+temp2)
        return out


    # def forward(self, img1, img2):
    #     for i,(layer1,layer2, fusion_atten)  in enumerate(zip(self.model,self.model_, self.fusion)):
    #         if i ==0:
    #             temp1 = img1
    #             temp2 = img2
    #         temp1 = layer1(temp1)
    #         temp2 = layer2(temp2)
    #         # fusion_res = fusion_atten(temp1,temp2)

    #         # fusion_res = temp1+temp2
    #         # temp1 = temp1 + fusion_res
    #         # temp2 = temp2 + fusion_res
       
    #     out = self.final(temp2+temp1)
    #     return out
