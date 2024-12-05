import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from utils import *

class boundarypad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.pad(F.pad(input,(0,0,1,1),'reflect'),(1,1,0,0),'circular')


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(p=0.1)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        x_mod = F.pad(F.pad(x,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        # Second convolution layer
        h = F.pad(F.pad(h,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        h = self.drop(h)
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class Self_attn_conv_reg(nn.Module):
    
    def __init__(self, in_channels,out_channels):
        super(Self_attn_conv_reg, self).__init__()
        self.query = self._conv(in_channels,in_channels//8,stride=1)
        self.key = self.key_conv(in_channels,in_channels//8,stride=2)
        self.value = self.key_conv(in_channels,out_channels,stride=2)
        self.post_map = nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=(1,1),stride=1,padding=0))
        self.out_ch = out_channels

    def _conv(self,n_in,n_out,stride):
        return nn.Sequential(boundarypad(),nn.Conv2d(n_in,n_in//2,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_in//2,n_out,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_out,n_out,kernel_size=(3,3),stride=stride,padding=0))
    
    def key_conv(self,n_in,n_out,stride):
        return nn.Sequential(boundarypad(),nn.Conv2d(n_in,n_in//2,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_in//2,n_out,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_out,n_out,kernel_size=(3,3),stride=1,padding=0))
    
    def forward(self, x):
        size = x.size()
        x = x.float()
        q,k,v = self.query(x).flatten(-2,-1),self.key(x).flatten(-2,-1),self.value(x).flatten(-2,-1)
        beta = F.softmax(torch.bmm(q.transpose(1,2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1,2))
        o = self.post_map(o.view(-1,self.out_ch,size[-2],size[-1]).contiguous())
        return o
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        Squeeze-and-Excitation Layer
        
        Args:
            channel (int): Number of input channels
            reduction (int): Reduction ratio for channel compression
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of SE Layer
        
        Args:
            x (torch.Tensor): Input feature map
        
        Returns:
            torch.Tensor: Channel-wise recalibrated feature map
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)

class SEResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        reduction: int = 16
    ):
        """
        Squeeze-and-Excitation ResNet Block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            activation (str): Activation function type
            norm (bool): Whether to use group normalization
            n_groups (int): Number of groups for group normalization
            reduction (int): Reduction ratio for SE layer
        """
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se_layer = SELayer(out_channels, reduction)
        
        self.drop = nn.Dropout(p=0.2)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()
        
        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of SE-ResNet Block
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after SE-ResNet block processing
        """
        x_mod = F.pad(F.pad(x, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        
        h = F.pad(F.pad(h, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        
        h = self.se_layer(h)
        
        h = self.drop(h)
        
        return h + self.shortcut(x)
    
class GlobalContextBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction_ratio: float = 0.5,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        # Compute the reduced channel dimension
        self.reduced_channels = max(1, int(in_channels * reduction_ratio))
        
        # Context modeling components
        # self.channel_attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
        #     nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1),
        #     nn.LeakyReLU(0.3),
        #     nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1),
        #     nn.Sigmoid()
        # )
        
        # Spatial context modeling
        self.spatial_context = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(self.reduced_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Main transformation paths
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Dropout and activation
        self.activation = nn.LeakyReLU(0.3)
        self.drop = nn.Dropout(p=0.2)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Optional normalization
        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # Preserve input for residual connection
        residual = x
        
        # Channel-wise global context
        # channel_att = self.channel_attention(x)
        # x_channel_weighted = x * channel_att
        
        # Spatial global context
        spatial_att = self.spatial_context(x)
        x_spatial_weighted = x * spatial_att
        
        # Combine global contexts
        # x_global_context = x_channel_weighted + x_spatial_weighted
        x_global_context = x_spatial_weighted

        # First convolution layer with padding
        x_mod = F.pad(F.pad(x_global_context,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        
        # Second convolution layer with padding
        h = F.pad(F.pad(h,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        h = self.drop(h)
        
        # Residual connection
        return h + self.shortcut(residual)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        reduction: int = 16
    ):
        """
        Mobile Inverted Bottleneck Block with Squeeze-and-Excitation
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            activation (str): Activation function type
            norm (bool): Whether to use group normalization
            n_groups (int): Number of groups for group normalization
            reduction (int): Reduction ratio for SE layer
        """
        super().__init__()
        
        # Matching activation style from SEResNetBlock
        self.activation = nn.LeakyReLU(0.3)
        
        # Expansion convolution with padding matching SEResNetBlock
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution matching SEResNetBlock
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation Layer
        self.se_layer = SELayer(out_channels, reduction)
        
        # Dropout matching SEResNetBlock
        self.drop = nn.Dropout(p=0.2)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Normalization handling
        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of MBConv Block
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor after MBConv block processing
        """
        # Circular and reflect padding matching SEResNetBlock
        x_mod = F.pad(F.pad(x, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
        
        # First convolution path
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        
        # Second convolution path with similar padding
        h = F.pad(F.pad(h, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        
        # Squeeze-and-Excitation
        h = self.se_layer(h)
        
        # Dropout
        h = self.drop(h)
        
        # Residual connection
        return h + self.shortcut(x)

class EfficientNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 10,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        reduction: int = 16
    ):
        """
        EfficientNet-like Architecture
        
        Args:
            in_channels (int): Number of input image channels
            num_classes (int): Number of output classes
            activation (str): Activation function type
            norm (bool): Whether to use group normalization
            n_groups (int): Number of groups for group normalization
            reduction (int): Reduction ratio for SE layer
        """
        super().__init__()
        
        # Initial convolution matching block-level padding and conv style
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3)
        )
        
        # Network configuration
        stages_config = [
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 512)
        ]
        
        # Create stages using MBConvBlock
        self.stages = nn.ModuleList()
        for in_ch, out_ch in stages_config:
            self.stages.append(
                MBConvBlock(
                    in_channels=in_ch, 
                    out_channels=out_ch,
                    activation=activation,
                    norm=norm,
                    n_groups=n_groups,
                    reduction=reduction
                )
            )
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512, out_channels)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of EfficientNet-like model
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Class logits
        """
        # Matching circular and reflect padding from block-level implementation
        x = F.pad(F.pad(x, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
        
        # Initial convolution
        x = self.stem(x)
        
        # Pass through stages
        for stage in self.stages:
            x = stage(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.flatten(1)
        
        # Classification
        x = self.classifier(x)
        
        return x




class Self_attn_conv(nn.Module):
    
    def __init__(self, in_channels,out_channels):
        super(Self_attn_conv, self).__init__()
        self.query = self._conv(in_channels,in_channels//8,stride=1)
        self.key = self.key_conv(in_channels,in_channels//8,stride=2)
        self.value = self.key_conv(in_channels,out_channels,stride=2)
        self.post_map = nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=(1,1),stride=1,padding=0))
        self.out_ch = out_channels

    def _conv(self,n_in,n_out,stride):
        return nn.Sequential(boundarypad(),nn.Conv2d(n_in,n_in//2,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_in//2,n_out,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_out,n_out,kernel_size=(3,3),stride=stride,padding=0))
    
    def key_conv(self,n_in,n_out,stride):
        return nn.Sequential(nn.Conv2d(n_in,n_in//2,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),nn.Conv2d(n_in//2,n_out,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),nn.Conv2d(n_out,n_out,kernel_size=(3,3),stride=1,padding=0))
    
    def forward(self, x):
        size = x.size()
        x = x.float()
        q,k,v = self.query(x).flatten(-2,-1),self.key(x).flatten(-2,-1),self.value(x).flatten(-2,-1)
        beta = F.softmax(torch.bmm(q.transpose(1,2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1,2))
        o = self.post_map(o.view(-1,self.out_ch,size[-2],size[-1]).contiguous())
        return o