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


import math

class WindowPartition(nn.Module):
    def __init__(self, window_size=7):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        window_size = self.window_size
        
        # Pad if needed to ensure divisibility
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Reshape into windows
        x = x.view(B, C, H + pad_h // window_size, window_size, W + pad_w // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
        
        return windows, (H, W)

class WindowReverse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, windows, original_size, window_size):
        H, W = original_size
        B = int(windows.shape[0] / ((H + window_size - 1) // window_size * 
                                    (W + window_size - 1) // window_size))
        
        windows = windows.view(B, (H + window_size - 1) // window_size, 
                                (W + window_size - 1) // window_size, 
                                windows.shape[1], window_size, window_size)
        
        windows = windows.permute(0, 3, 1, 4, 2, 5).contiguous()
        windows = windows.view(B, windows.shape[1], 
                               (H + window_size - 1) // window_size * window_size, 
                               (W + window_size - 1) // window_size * window_size)
        
        # Remove padding if needed
        windows = windows[:, :, :H, :W]
        
        return windows

class SwinTransformerAttention(nn.Module):
    def __init__(self, in_channels, out_channels, window_size=7, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Window partition and reverse operations
        self.window_partition = WindowPartition(window_size)
        self.window_reverse = WindowReverse()

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), out_channels)
        )
        self.register_buffer("relative_position_index", self._get_relative_position_index())

        # Final projection
        self.post_map = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def _get_relative_position_index(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Cyclic shift if shift_size is specified
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # Partition into windows
        windows, (orig_h, orig_w) = self.window_partition(shifted_x)
        
        # Compute attention within windows
        query = self.query(windows)
        key = self.key(windows)
        value = self.value(windows)

        # Reshape for attention computation
        query = query.view(query.size(0), query.size(1), -1).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1).transpose(1, 2)

        # Compute attention
        attn = torch.bmm(query, key)
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, 
            self.window_size * self.window_size, 
            -1
        )
        attn = attn + relative_position_bias.mean(dim=-1)
        
        # Softmax and apply attention
        attn = F.softmax(attn, dim=-1)
        output = torch.bmm(attn, value)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view_as(windows)
        
        # Reverse window partition
        output = self.window_reverse(output, (orig_h, orig_w), self.window_size)
        
        # Reverse cyclic shift if applicable
        if self.shift_size > 0:
            output = torch.roll(output, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # Final projection
        output = self.post_map(output)

        return output