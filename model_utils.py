# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional, Tuple, Union
# from utils import *

# class boundarypad(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input):
#         return F.pad(F.pad(input,(0,0,1,1),'reflect'),(1,1,0,0),'circular')


# class ResidualBlock(nn.Module):

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         activation: str = "gelu",
#         norm: bool = False,
#         n_groups: int = 1,
#     ):
#         super().__init__()
#         self.activation = nn.LeakyReLU(0.3)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.drop = nn.Dropout(p=0.1)
#         # If the number of input channels is not equal to the number of output channels we have to
#         # project the shortcut connection
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
#         else:
#             self.shortcut = nn.Identity()

#         if norm:
#             self.norm1 = nn.GroupNorm(n_groups, in_channels)
#             self.norm2 = nn.GroupNorm(n_groups, out_channels)
#         else:
#             self.norm1 = nn.Identity()
#             self.norm2 = nn.Identity()

#     def forward(self, x: torch.Tensor):
#         # First convolution layer
#         x_mod = F.pad(F.pad(x,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
#         h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
#         # Second convolution layer
#         h = F.pad(F.pad(h,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
#         h = self.activation(self.bn2(self.conv2(self.norm2(h))))
#         h = self.drop(h)
#         # Add the shortcut connection and return
#         return h + self.shortcut(x)


# class Self_attn_conv_reg(nn.Module):
    
#     def __init__(self, in_channels,out_channels):
#         super(Self_attn_conv_reg, self).__init__()
#         self.query = self._conv(in_channels,in_channels//8,stride=1)
#         self.key = self.key_conv(in_channels,in_channels//8,stride=2)
#         self.value = self.key_conv(in_channels,out_channels,stride=2)
#         self.post_map = nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=(1,1),stride=1,padding=0))
#         self.out_ch = out_channels

#     def _conv(self,n_in,n_out,stride):
#         return nn.Sequential(boundarypad(),nn.Conv2d(n_in,n_in//2,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_in//2,n_out,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_out,n_out,kernel_size=(3,3),stride=stride,padding=0))
    
#     def key_conv(self,n_in,n_out,stride):
#         return nn.Sequential(boundarypad(),nn.Conv2d(n_in,n_in//2,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_in//2,n_out,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_out,n_out,kernel_size=(3,3),stride=1,padding=0))
    
#     def forward(self, x):
#         size = x.size()
#         x = x.float()
#         q,k,v = self.query(x).flatten(-2,-1),self.key(x).flatten(-2,-1),self.value(x).flatten(-2,-1)
#         beta = F.softmax(torch.bmm(q.transpose(1,2), k), dim=1)
#         o = torch.bmm(v, beta.transpose(1,2))
#         o = self.post_map(o.view(-1,self.out_ch,size[-2],size[-1]).contiguous())
#         return o
    

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         """
#         Squeeze-and-Excitation Layer
        
#         Args:
#             channel (int): Number of input channels
#             reduction (int): Reduction ratio for channel compression
#         """
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         """
#         Forward pass of SE Layer
        
#         Args:
#             x (torch.Tensor): Input feature map
        
#         Returns:
#             torch.Tensor: Channel-wise recalibrated feature map
#         """
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
        
#         y = self.fc(y).view(b, c, 1, 1)
        
#         return x * y.expand_as(x)

# class SEResNetBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         activation: str = "gelu",
#         norm: bool = False,
#         n_groups: int = 1,
#         reduction: int = 16
#     ):
#         """
#         Squeeze-and-Excitation ResNet Block
        
#         Args:
#             in_channels (int): Number of input channels
#             out_channels (int): Number of output channels
#             activation (str): Activation function type
#             norm (bool): Whether to use group normalization
#             n_groups (int): Number of groups for group normalization
#             reduction (int): Reduction ratio for SE layer
#         """
#         super().__init__()
#         self.activation = nn.LeakyReLU(0.3)
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.se_layer = SELayer(out_channels, reduction)
        
#         self.drop = nn.Dropout(p=0.2)
        
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
#                 nn.BatchNorm2d(out_channels)
#                 )
#         else:
#             self.shortcut = nn.Identity()
        
#         if norm:
#             self.norm1 = nn.GroupNorm(n_groups, in_channels)
#             self.norm2 = nn.GroupNorm(n_groups, out_channels)
#         else:
#             self.norm1 = nn.Identity()
#             self.norm2 = nn.Identity()

#     def forward(self, x: torch.Tensor):
#         """
#         Forward pass of SE-ResNet Block
        
#         Args:
#             x (torch.Tensor): Input tensor
        
#         Returns:
#             torch.Tensor: Output tensor after SE-ResNet block processing
#         """
#         x_mod = F.pad(F.pad(x, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
#         h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        
#         h = F.pad(F.pad(h, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
#         h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        
#         h = self.se_layer(h)
        
#         h = self.drop(h)
        
#         return h + self.shortcut(x)
    
# class GlobalContextBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         reduction_ratio: float = 0.5,
#         activation: str = "gelu",
#         norm: bool = False,
#         n_groups: int = 1,
#     ):
#         super().__init__()
#         # Compute the reduced channel dimension
#         self.reduced_channels = max(1, int(in_channels * reduction_ratio))
        
#         # Context modeling components
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
#             nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1),
#             nn.LeakyReLU(0.3),
#             nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         # Spatial context modeling
#         self.spatial_context = nn.Sequential(
#             nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1),
#             nn.LeakyReLU(0.3),
#             nn.Conv2d(self.reduced_channels, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         # Main transformation paths
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         # Dropout and activation
#         self.activation = nn.LeakyReLU(0.3)
#         self.drop = nn.Dropout(p=0.2)
        
#         # Shortcut connection
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.shortcut = nn.Identity()
        
#         # Optional normalization
#         if norm:
#             self.norm1 = nn.GroupNorm(n_groups, in_channels)
#             self.norm2 = nn.GroupNorm(n_groups, out_channels)
#         else:
#             self.norm1 = nn.Identity()
#             self.norm2 = nn.Identity()

#     def forward(self, x: torch.Tensor):
#         # Preserve input for residual connection
#         residual = x
        
#         # Channel-wise global context
#         channel_att = self.channel_attention(x)
#         x_channel_weighted = x * channel_att
        
#         # Spatial global context
#         spatial_att = self.spatial_context(x)
#         x_spatial_weighted = x * spatial_att
        
#         # Combine global contexts
#         x_global_context = x_channel_weighted + x_spatial_weighted
        
#         # First convolution layer with padding
#         x_mod = F.pad(F.pad(x_global_context,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
#         h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        
#         # Second convolution layer with padding
#         h = F.pad(F.pad(h,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
#         h = self.activation(self.bn2(self.conv2(self.norm2(h))))
#         h = self.drop(h)
        
#         # Residual connection
#         return h + self.shortcut(residual)



# class Self_attn_conv(nn.Module):
    
#     def __init__(self, in_channels,out_channels):
#         super(Self_attn_conv, self).__init__()
#         self.query = self._conv(in_channels,in_channels//8,stride=1)
#         self.key = self.key_conv(in_channels,in_channels//8,stride=2)
#         self.value = self.key_conv(in_channels,out_channels,stride=2)
#         self.post_map = nn.Sequential(nn.Conv2d(out_channels,out_channels,kernel_size=(1,1),stride=1,padding=0))
#         self.out_ch = out_channels

#     def _conv(self,n_in,n_out,stride):
#         return nn.Sequential(boundarypad(),nn.Conv2d(n_in,n_in//2,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_in//2,n_out,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),boundarypad(),nn.Conv2d(n_out,n_out,kernel_size=(3,3),stride=stride,padding=0))
    
#     def key_conv(self,n_in,n_out,stride):
#         return nn.Sequential(nn.Conv2d(n_in,n_in//2,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),nn.Conv2d(n_in//2,n_out,kernel_size=(3,3),stride=stride,padding=0),nn.LeakyReLU(0.3),nn.Conv2d(n_out,n_out,kernel_size=(3,3),stride=1,padding=0))
    
#     def forward(self, x):
#         size = x.size()
#         x = x.float()
#         q,k,v = self.query(x).flatten(-2,-1),self.key(x).flatten(-2,-1),self.value(x).flatten(-2,-1)
#         beta = F.softmax(torch.bmm(q.transpose(1,2), k), dim=1)
#         o = torch.bmm(v, beta.transpose(1,2))
#         o = self.post_map(o.view(-1,self.out_ch,size[-2],size[-1]).contiguous())
#         return o



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
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
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
        channel_att = self.channel_attention(x)
        x_channel_weighted = x * channel_att
        
        # Spatial global context
        spatial_att = self.spatial_context(x)
        x_spatial_weighted = x * spatial_att
        
        # Combine global contexts
        x_global_context = x_channel_weighted + x_spatial_weighted
        
        # First convolution layer with padding
        x_mod = F.pad(F.pad(x_global_context,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        
        # Second convolution layer with padding
        h = F.pad(F.pad(h,(0,0,1,1),'reflect'),(1,1,0,0),'circular')
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        h = self.drop(h)
        
        # Residual connection
        return h + self.shortcut(residual)
    

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.reduction = reduction
        reduced_channels = max(1, in_channels // reduction)  # Ensure reduced channels are at least 1
        
        # Channel attention for height
        self.fc_h = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention for width
        self.fc_w = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        att_h = self.fc_h(x)
        att_w = self.fc_w(x)
        return x * att_h * att_w
    

class CoordinateAttentionResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        reduction: int = 16
    ):
        super().__init__()

        # Ensure out_channels is an integer
        # if isinstance(out_channels, list):
        #     out_channels = out_channels[0]
        out_channels = int(out_channels)
        
        self.activation = nn.LeakyReLU(0.3)
        
        # Force float conversion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0, dtype=torch.float32)
        self.bn1 = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0, dtype=torch.float32)
        self.bn2 = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        
        self.coord_attention = CoordinateAttention(out_channels, reduction)
        
        self.drop = nn.Dropout(p=0.2)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), dtype=torch.float32),
                nn.BatchNorm2d(out_channels, dtype=torch.float32)
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
        # Ensure input is float32
        x = x.float()
        
        # First convolution layer with padding
        x_mod = F.pad(F.pad(x, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'circular')
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        
        # Second convolution layer with padding
        h = F.pad(F.pad(h, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'circular')
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        
        # Apply coordinate attention
        h = self.coord_attention(h)
        
        h = self.drop(h)
        
        return h + self.shortcut(x)
    

class CNNLSTMResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_lstm_layers: int = 1,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        # Convolutional layers
        self.activation = nn.LeakyReLU(0.3)
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Dropout
        self.drop = nn.Dropout(p=0.2)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=out_channels,  # Use out_channels as input size for LSTM
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Projection for shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Normalization
        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # CNN Path
        # First convolution layer with padding
        x_mod = F.pad(F.pad(x, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        
        # Second convolution layer with padding
        h = F.pad(F.pad(h, (0,0,1,1), 'reflect'), (1,1,0,0), 'circular')
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        h = self.drop(h)
        
        # Prepare for LSTM: reshape and transpose
        # Assuming input shape is [batch_size, channels, height, width]
        batch_size, channels, height, width = h.shape
        
        # Reshape to [batch_size, sequence_length, features]
        # Here, we'll use height as sequence length and flatten width and channels
        h_reshaped = h.permute(0, 2, 3, 1).contiguous()  # [batch_size, height, width, channels]
        h_reshaped = h_reshaped.view(batch_size, height, -1)  # [batch_size, height, width*channels]
        
        # LSTM processing
        lstm_out, _ = self.lstm(h_reshaped)
        
        # Reshape LSTM output back to match original tensor shape
        lstm_out = lstm_out.view(batch_size, height, width, -1)
        lstm_out = lstm_out.permute(0, 3, 1, 2).contiguous()
        
        # Residual connection
        shortcut = self.shortcut(x)
        
        return lstm_out + shortcut


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