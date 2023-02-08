import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck_1 = UNet3D._block(features * 8, features * 16, name="bottleneck")
        self.bottleneck_2 = UNet3D._block(features * 8, features * 16, name="bottleneck")
        self.bottleneck_3 = UNet3D._block(features * 8, features * 16, name="bottleneck")

        ## xy
        self.upconv4_xy = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4_xy = UNet3D._block_2d((features * 8)*2  , features * 8, name="dec4")
        self.upconv3_xy = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3_xy = UNet3D._block_2d((features * 4)*2 , features * 4, name="dec3")
        self.upconv2_xy = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2_xy = UNet3D._block_2d((features * 2)*2 , features * 2, name="dec2")
        self.upconv1_xy = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1_xy = UNet3D._block_2d(features*2 , features, name="dec1")

        ## yz
        self.upconv4_yz = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4_yz = UNet3D._block_2d((features * 8)*2  , features * 8, name="dec4")
        self.upconv3_yz = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3_yz = UNet3D._block_2d((features * 4)*2 , features * 4, name="dec3")
        self.upconv2_yz = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2_yz = UNet3D._block_2d((features * 2)*2 , features * 2, name="dec2")
        self.upconv1_yz = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1_yz = UNet3D._block_2d(features*2 , features, name="dec1")

        ## xz
        self.upconv4_xz = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4_xz = UNet3D._block_2d((features * 8)*2  , features * 8, name="dec4")
        self.upconv3_xz = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3_xz = UNet3D._block_2d((features * 4)*2 , features * 4, name="dec3")
        self.upconv2_xz = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2_xz = UNet3D._block_2d((features * 2) *2, features * 2, name="dec2")
        self.upconv1_xz = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1_xz = UNet3D._block_2d(features*2, features, name="dec1")

        self.bn = nn.BatchNorm3d(features)
        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        enc4_tmp = self.pool4(enc4)
        bottleneck_1 = self.bottleneck_1(enc4_tmp)
        bottleneck_1 = torch.sum(bottleneck_1,dim=-1)
        bottleneck_2 = self.bottleneck_2(enc4_tmp)
        bottleneck_2 = torch.sum(bottleneck_2,dim=-2)
        bottleneck_3 = self.bottleneck_3(enc4_tmp)
        bottleneck_3 = torch.sum(bottleneck_3,dim=-3)

        # bottleneck_1 = bottleneck[:,:,:,:,0] + bottleneck[:,:,:,:,3]
        # bottleneck_2 = bottleneck[:,:,:,:,1] + bottleneck[:,:,:,:,3]
        # bottleneck_3 = bottleneck[:,:,:,:,2] + bottleneck[:,:,:,:,3]

        dec4_xy = self.upconv4_xy(bottleneck_1)
        dec4_xy = torch.cat((dec4_xy, torch.sum(enc4,dim=-1)), dim=1) 
        dec4_xy = self.decoder4_xy(dec4_xy)
        dec3_xy = self.upconv3_xy(dec4_xy)
        dec3_xy = torch.cat((dec3_xy, torch.sum(enc3,dim=-1)), dim=1) 
        dec3_xy = self.decoder3_xy(dec3_xy)
        dec2_xy = self.upconv2_xy(dec3_xy)
        dec2_xy = torch.cat((dec2_xy, torch.sum(enc2,dim=-1)), dim=1) 
        dec2_xy = self.decoder2_xy(dec2_xy)
        dec1_xy = self.upconv1_xy(dec2_xy)
        dec1_xy = torch.cat((dec1_xy, torch.sum(enc1,dim=-1)), dim=1) 
        dec1_xy = self.decoder1_xy(dec1_xy)

        dec4_yz = self.upconv4_yz(bottleneck_2)
        dec4_yz = torch.cat((dec4_yz, torch.sum(enc4,dim=-2)), dim=1) 
        dec4_yz = self.decoder4_yz(dec4_yz)
        dec3_yz = self.upconv3_yz(dec4_yz)
        dec3_yz = torch.cat((dec3_yz, torch.sum(enc3,dim=-2)), dim=1) 
        dec3_yz = self.decoder3_yz(dec3_yz)
        dec2_yz = self.upconv2_yz(dec3_yz)
        dec2_yz = torch.cat((dec2_yz, torch.sum(enc2,dim=-2)), dim=1) 
        dec2_yz = self.decoder2_yz(dec2_yz)
        dec1_yz = self.upconv1_yz(dec2_yz)
        dec1_yz = torch.cat((dec1_yz, torch.sum(enc1,dim=-2)), dim=1) 
        dec1_yz = self.decoder1_yz(dec1_yz)

        dec4_xz = self.upconv4_xz(bottleneck_3)
        dec4_xz = torch.cat((dec4_xz, torch.sum(enc4,dim=-3)), dim=1) 
        dec4_xz = self.decoder4_xz(dec4_xz)
        dec3_xz = self.upconv3_xz(dec4_xz)
        dec3_xz = torch.cat((dec3_xz, torch.sum(enc3,dim=-3)), dim=1) 
        dec3_xz = self.decoder3_xz(dec3_xz)
        dec2_xz = self.upconv2_xz(dec3_xz)
        dec2_xz = torch.cat((dec2_xz, torch.sum(enc2,dim=-3)), dim=1) 
        dec2_xz = self.decoder2_xz(dec2_xz)
        dec1_xz = self.upconv1_xz(dec2_xz)
        dec1_xz = torch.cat((dec1_xz, torch.sum(enc1,dim=-3)), dim=1) 
        dec1_xz = self.decoder1_xz(dec1_xz)

        order_list = [(0,1,2,3,4),(0,1,4,2,3),(0,1,2,4,3)]
        feature_list = [dec1_xy, dec1_yz, dec1_xz]
        feature_list_1 = [dec2_xy, dec2_yz, dec2_xz]
        feature_list_2 = [dec3_xy, dec3_yz, dec3_xz]
        feature_list_3 = [dec4_xy, dec4_yz, dec4_xz]
        plane_3d_origin = [] 
        plane_3d_1 = [] 
        plane_3d_2 = [] 
        plane_3d_3 = [] 
        for j in range(3):
            # tmp = feature_list[j]
            tmp_sdf = feature_list[j].unsqueeze(-1).repeat(1,1,1,1,dec1_xz.shape[2]).permute(order_list[j])
            plane_3d_origin.append(tmp_sdf)

            tmp_sdf = feature_list_1[j].unsqueeze(-1).repeat(1,1,1,1,dec2_xz.shape[2]).permute(order_list[j])
            plane_3d_1.append(tmp_sdf)

            tmp_sdf = feature_list_2[j].unsqueeze(-1).repeat(1,1,1,1,dec3_xz.shape[2]).permute(order_list[j])
            plane_3d_2.append(tmp_sdf)

            tmp_sdf = feature_list_3[j].unsqueeze(-1).repeat(1,1,1,1,dec4_xz.shape[2]).permute(order_list[j])
            plane_3d_3.append(tmp_sdf)
        res = sum(plane_3d_origin)
        res_1 = sum(plane_3d_1) #dec2
        res_2 = sum(plane_3d_2) #dec3
        res_3 = sum(plane_3d_3) #dec4

        
        # res = self.bn(res)
        # res = nn.ReLU()(res)



        outputs = self.conv(res)
        return outputs, res_1,res_2,res_3,enc1,enc2,enc3,enc4

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


    @staticmethod
    def _block_2d(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 64
    x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))
    
    model = UNet3D(in_channels=1, out_channels=1)
    
    out = model(x)
    print("out size: {}".format(out.size()))