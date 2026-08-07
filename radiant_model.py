import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from scipy.signal import convolve2d

from radiant_config import RADIANTConfig as args
try:
    from timm.layers import trunc_normal_
except ImportError:  # timm < 0.9
    from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out

from radiant_modules import (
    MultiGranularityInteraction,
    NonInterferingDecomposition,
    SobelGradient,
    TaskMemory,
)
EPSILON = 1e-4
MAX = 1e2


class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias

#SwinTransformer - Window patition
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows

#SwinTransformer - Window reverse
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#SwinTransformer - get relative position
def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

#SwinTransformer - Window attention
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift_size):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.shift_size = shift_size

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, qkv_mfif, trainingTag):
        #MFIF training
        if (trainingTag == 2):
            B_, N, _ = qkv.shape        
            qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
            qkv_mfif = qkv_mfif.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q_mfif, k_mfif, v_mfif = qkv_mfif[0], qkv_mfif[1], qkv_mfif[2]  # make torchscript happy (cannot use tensor as tuple)

            #text modality -> vision
            with torch.no_grad():            
                q = q * self.scale
            q_mfif = q_mfif * self.scale
            
            with torch.no_grad():            
                if (self.shift_size == 0):
                    attn = (q_mfif @ k.transpose(-2, -1))            
                else:
                    attn = (q @ k.transpose(-2, -1))            
            attn_mfif = (q_mfif @ k_mfif.transpose(-2, -1))


            relative_position_bias = self.meta(self.relative_positions)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            with torch.no_grad():
                attn = attn + relative_position_bias.unsqueeze(0)        
            attn_mfif = attn_mfif + relative_position_bias.unsqueeze(0)

            with torch.no_grad():
                attn = self.softmax(attn)
            attn_mfif = self.softmax(attn_mfif)

            with torch.no_grad():        
                x_ivif = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
            x_mfif = (attn_mfif @ v_mfif).transpose(1, 2).reshape(B_, N, self.dim)
        elif (trainingTag == 1):
            B_, N, _ = qkv.shape        
            qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
            qkv_mfif = qkv_mfif.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q_mfif, k_mfif, v_mfif = qkv_mfif[0], qkv_mfif[1], qkv_mfif[2]  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            with torch.no_grad():
                q_mfif = q_mfif * self.scale
            
            attn = (q @ k.transpose(-2, -1))
            
            with torch.no_grad():
                if (self.shift_size == 0):
                    attn_mfif = (q @ k_mfif.transpose(-2, -1))
                else:
                    attn_mfif = (q_mfif @ k_mfif.transpose(-2, -1))

            relative_position_bias = self.meta(self.relative_positions)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            attn = attn + relative_position_bias.unsqueeze(0)        
            with torch.no_grad():
                attn_mfif = attn_mfif + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            with torch.no_grad():
                attn_mfif = self.softmax(attn_mfif)

            x_ivif = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
            with torch.no_grad():        
                x_mfif = (attn_mfif @ v_mfif).transpose(1, 2).reshape(B_, N, self.dim)

        return x_ivif, x_mfif

spe_transformer_cur_depth = 0

def save_feature_maps_as_images(feature_maps, alias, numFeatures = 3, output_folder="visualization"):
    global spe_transformer_cur_depth
    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Assuming feature_maps is a PyTorch tensor with dimensions BxCxHxW
    batch_size, num_channels, height, width = feature_maps.size()    
    for i in range(batch_size):
        for j in range(numFeatures):
            # Get the j-th channel of the i-th feature map
            channel_data = feature_maps[i, j, :, :].cpu().detach().numpy()

            # Normalize values to be in the range [0, 255]
            channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min()) * 255

            # Convert to uint8
            channel_data = channel_data.astype('uint8')

            # Create a PIL Image from the numpy array
            image = Image.fromarray(channel_data)

            # Save the image
            image_path = os.path.join(output_folder, f"content_{alias}_spe_transformer_cur_depth_{spe_transformer_cur_depth}_feature_map_{0}_channel_{j}.jpg")
            image.save(image_path)

#SwinTransformer - Attention
class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
            self.conv_mfif = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.V_mfif = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)
            self.proj_mfif = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, 2*dim, 1)
            self.QK_mfif = nn.Conv2d(dim, 2*dim, 1)
            self.attn = WindowAttention(dim, window_size, num_heads, shift_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:    # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1/4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x_ivif, x_mfif, trainingTag):
        B, C, H, W = x_ivif.shape
        
        #MFIF task
        if (trainingTag == 2):
        
            #print(x_ivif.shape);
            if self.conv_type == 'DWConv' or self.use_attn:
                with torch.no_grad():            
                    V = self.V(x_ivif)
                V_mfif = self.V_mfif(x_mfif)

            #save_feature_maps_as_images(V_mfif,"v_mfif");
            
            #print("V.shape:");
            #print(V.shape);

            if self.use_attn:
                with torch.no_grad():            
                    QK = self.QK(x_ivif)
                QK_mfif = self.QK_mfif(x_mfif)

                #save_feature_maps_as_images(QK_mfif[:,:self.dim,:,:],"q_mfif");
                #save_feature_maps_as_images(QK_mfif[:,self.dim:,:,:],"k_mfif");

                with torch.no_grad():
                    QKV = torch.cat([QK, V], dim=1)
                QKV_mfif = torch.cat([QK_mfif, V_mfif], dim=1)



                with torch.no_grad():                
                    shifted_QKV = self.check_size(QKV, self.shift_size > 0)
                shifted_QKV_mfif = self.check_size(QKV_mfif, self.shift_size > 0)
                
                
                Ht, Wt = shifted_QKV.shape[2:]

                # partition windows
                with torch.no_grad():                
                    shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
                shifted_QKV_mfif = shifted_QKV_mfif.permute(0, 2, 3, 1)

                qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
                qkv_mfif = window_partition(shifted_QKV_mfif, self.window_size)  # nW*B, window_size**2, C

                #attn_windows = self.attn(qkv)
                #attn_windows_mfif = self.attn(qkv_mfif)
                attn_windows, attn_windows_mfif = self.attn(qkv, qkv_mfif, trainingTag)
                

                # merge windows
                shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
                shifted_out_mfif = window_reverse(attn_windows_mfif, self.window_size, Ht, Wt)  # B H' W' C

                # reverse cyclic shift
                out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
                out_mfif = shifted_out_mfif[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]

                attn_out = out.permute(0, 3, 1, 2)
                attn_out_mfif = out_mfif.permute(0, 3, 1, 2)

                if self.conv_type in ['Conv', 'DWConv']:
                    with torch.no_grad():                
                        conv_out = self.conv(V)
                    conv_out_mfif = self.conv_mfif(V_mfif)
                    
                    with torch.no_grad():                    
                        out = self.proj(conv_out + attn_out)
                    out_mfif = self.proj_mfif(conv_out_mfif + attn_out_mfif)
                else:
                    with torch.no_grad():                                    
                        out = self.proj(attn_out)
                    out_mfif = self.proj_mfif(attn_out_mfif)

            else:
                if self.conv_type == 'Conv':
                    out = self.conv(x_ivif)                # no attention and use conv, no projection
                    out_mfif = self.conv_mfif(x_mfif)                # no attention and use conv, no projection
                elif self.conv_type == 'DWConv':
                    out = self.proj(self.conv(V))
                    out_mfif = self.proj_mfif(self.conv_mfif(V_mfif))
        elif (trainingTag == 1):
            #print(x_ivif.shape);
            if self.conv_type == 'DWConv' or self.use_attn:
                V = self.V(x_ivif)
                with torch.no_grad():
                    V_mfif = self.V_mfif(x_mfif)
            #print("V.shape:");
            #print(V.shape);
            if self.use_attn:
                QK = self.QK(x_ivif)
                with torch.no_grad():
                    QK_mfif = self.QK_mfif(x_mfif)

                QKV = torch.cat([QK, V], dim=1)
                
                #print("QKV.shape:");
                #print(QKV.shape);                
                #
                #print(V.shape);                
                with torch.no_grad():
                    QKV_mfif = torch.cat([QK_mfif, V_mfif], dim=1)

                # shift
                shifted_QKV = self.check_size(QKV, self.shift_size > 0)
                with torch.no_grad():
                    shifted_QKV_mfif = self.check_size(QKV_mfif, self.shift_size > 0)
                Ht, Wt = shifted_QKV.shape[2:]

                # partition windows
                shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
                with torch.no_grad():
                    shifted_QKV_mfif = shifted_QKV_mfif.permute(0, 2, 3, 1)

                qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
                qkv_mfif = window_partition(shifted_QKV_mfif, self.window_size)  # nW*B, window_size**2, C

                #attn_windows = self.attn(qkv)
                #attn_windows_mfif = self.attn(qkv_mfif)
                attn_windows, attn_windows_mfif = self.attn(qkv, qkv_mfif, trainingTag)
                

                # merge windows
                shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
                shifted_out_mfif = window_reverse(attn_windows_mfif, self.window_size, Ht, Wt)  # B H' W' C

                # reverse cyclic shift
                out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
                out_mfif = shifted_out_mfif[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]

                attn_out = out.permute(0, 3, 1, 2)
                attn_out_mfif = out_mfif.permute(0, 3, 1, 2)

                if self.conv_type in ['Conv', 'DWConv']:
                    conv_out = self.conv(V)
                    with torch.no_grad():
                        conv_out_mfif = self.conv_mfif(V)
                    
                    out = self.proj(conv_out + attn_out)
                    with torch.no_grad():
                        out_mfif = self.proj_mfif(conv_out_mfif + attn_out_mfif)
                else:
                    out = self.proj(attn_out)
                    with torch.no_grad():                    
                        out_mfif = self.proj_mfif(attn_out_mfif)

            else:
                if self.conv_type == 'Conv':
                    out = self.conv(x_ivif)                # no attention and use conv, no projection
                    out_mfif = self.conv_mfif(x_mfif)                # no attention and use conv, no projection
                elif self.conv_type == 'DWConv':
                    out = self.proj(self.conv(V))
                    out_mfif = self.proj_mfif(self.conv_mfif(V_mfif))
        
        return out, out_mfif


class Mlp(nn.Module):
	def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.network_depth = network_depth

        
		self.mlp = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1),
			nn.ReLU(True),
			nn.Conv2d(hidden_features, out_features, 1)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.network_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.mlp(x)

#SwinTransformer - TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.norm1_mfif = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.norm2_mfif = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()

        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        self.mlp_mfif = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x_ivif, x_mfif, trainingTag):
    
        #train IVIF task;
        if (trainingTag == 1):
            identity = x_ivif
            identity_mfif = x_mfif
            if self.use_attn: x_ivif, rescale, rebias = self.norm1(x_ivif)
            if self.use_attn: x_mfif, rescale_mfif, rebias_mfif = self.norm1_mfif(x_mfif)

            x_ivif, x_mfif = self.attn(x_ivif, x_mfif, trainingTag)

            if self.use_attn: x_ivif = x_ivif * rescale + rebias
            if self.use_attn: x_mfif = x_mfif * rescale_mfif + rebias_mfif

            x_ivif = identity + x_ivif
            x_mfif = identity_mfif + x_mfif

            identity = x_ivif
            identity_mfif = x_mfif

            if self.use_attn and self.mlp_norm: x_ivif, rescale, rebias = self.norm2(x_ivif)
            
            if self.use_attn and self.mlp_norm: x_mfif, rescale_mfif, rebias_mfif = self.norm2_mfif(x_mfif)

            x_ivif = self.mlp(x_ivif)
            x_mfif = self.mlp_mfif(x_mfif)

            if self.use_attn and self.mlp_norm: x_ivif = x_ivif * rescale + rebias
            if self.use_attn and self.mlp_norm: x_mfif = x_mfif * rescale_mfif + rebias_mfif


            x_ivif = identity + x_ivif
            x_mfif = identity_mfif + x_mfif
        elif (trainingTag == 2):
            #trainMFIF task;
            identity = x_ivif
            identity_mfif = x_mfif
            if self.use_attn: x_ivif, rescale, rebias = self.norm1(x_ivif)
            if self.use_attn: x_mfif, rescale_mfif, rebias_mfif = self.norm1_mfif(x_mfif)

            x_ivif, x_mfif = self.attn(x_ivif, x_mfif, trainingTag)

            if self.use_attn: x_ivif = x_ivif * rescale + rebias
            if self.use_attn: x_mfif = x_mfif * rescale_mfif + rebias_mfif

            x_ivif = identity + x_ivif
            x_mfif = identity_mfif + x_mfif

            identity = x_ivif
            identity_mfif = x_mfif

            if self.use_attn and self.mlp_norm: x_ivif, rescale, rebias = self.norm2(x_ivif)
            
            if self.use_attn and self.mlp_norm: x_mfif, rescale_mfif, rebias_mfif = self.norm2_mfif(x_mfif)

            x_ivif = self.mlp(x_ivif)
            x_mfif = self.mlp_mfif(x_mfif)

            if self.use_attn and self.mlp_norm: x_ivif = x_ivif * rescale + rebias
            if self.use_attn and self.mlp_norm: x_mfif = x_mfif * rescale_mfif + rebias_mfif


            x_ivif = identity + x_ivif
            x_mfif = identity_mfif + x_mfif
        return x_ivif, x_mfif

#Swintransformer - Basic
class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth-attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])
        self.weights = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(depth)])            
        self.weights_mfif = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(depth)])            

    def forward(self, x_ivif, x_mfif, trainingTag):
        global spe_transformer_cur_depth
        # IVIF train
        if (trainingTag == 1):
            for i, blk in enumerate(self.blocks):
                #identity_ivif = x_ivif;
                x_ivif, x_mfif  = blk(x_ivif, x_mfif, trainingTag)
                weight_i = self.weights[i];          

                if (i % 2 == 0):
                    x_ivif = x_ivif + weight_i*x_mfif;

            return x_ivif;
        elif (trainingTag == 2):
        # MFIF train
            for i, blk in enumerate(self.blocks):
                spe_transformer_cur_depth = i;
                x_ivif, x_mfif  = blk(x_ivif, x_mfif, trainingTag)
                weight_i = self.weights_mfif[i];          
                if (i % 2 == 0):
                    x_mfif = x_mfif + weight_i*x_ivif;                
            return x_mfif;

#SwinTransformer - Embed
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        #1x1 conv
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=patch_size,
            padding=(kernel_size - patch_size + 1) // 2,
            padding_mode='reflect',
        )

    def forward(self, x):
        x = self.proj(x)
        return x

#SwinTransformer - UnEmbed
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


#SwinTransformer - Main
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # n = 128  # number of filters
        # s = 3  # filter size
        # num_block = 4  # number of layers
        # Channel = 3
    
        self.patch_size = 4
        embed_dims=[32+32+32+2,48]
        depths = [2]
        num_heads = [2]
        attn_ratio = [1] #all layers use attention
        conv_type = ['DWConv']
        mlp_ratios=[2.]
        window_size=8
        in_chans = embed_dims[0]
        norm_layer=[RLN, RLN, RLN, RLN, RLN]
        # backbone
        self.patch_embed_ivif = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)            
            
        self.patch_embed_mfif = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)                        
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])     
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=embed_dims[0], embed_dim=embed_dims[0], kernel_size=3)                                 
                
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, mod_pad_w, mod_pad_h

    def forward(self, x_ivif, x_mfif, trainingTag):

        #save_feature_maps_as_images(x_ivif,"cnnFeatures_ivif",20);

        x_ivif, mod_pad_w_ivif, mod_pad_h_ivif = self.check_image_size(x_ivif);     
        x_mfif, mod_pad_w_mfif, mod_pad_h_mfif = self.check_image_size(x_mfif);     

        x_ivif = self.patch_embed_ivif(x_ivif);        
        x_mfif = self.patch_embed_mfif(x_mfif);        
        
        x = self.layer1(x_ivif, x_mfif, trainingTag);        
        
        #save_feature_maps_as_images(x_mfif,"afterAttentionMF",20);
        #save_feature_maps_as_images(x_ivif,"afterAttentionIVIF",20);
        
        x = self.patch_unembed(x);        
        _,_,h,w = x.size()
        
        x = x[:, :, :h - mod_pad_h_ivif, :w - mod_pad_w_ivif]        
        
        return x

#Shared feature fusion module
class ComplementFeatureFusionModule(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(ComplementFeatureFusionModule, self).__init__()

        self.height = height
        d = (32+32+32+2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d((32+32+32+2)*2, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)

        attn = self.mlp(in_feats) # mlp(B*C*1*1)->B*(C*2)*1*1       

        return attn

#Specific Feature Extraction & decoder
class TransformerSpecificExtractor(nn.Module):
    def __init__(self):
        super(TransformerSpecificExtractor, self).__init__()

        self.SwinTransformerSpecific = TransformerNet()

    def forward(self, x_ivif, x_mfif, trainingTag):
        
        x = self.SwinTransformerSpecific(x_ivif, x_mfif, trainingTag);

        return x

class CNNspecificDecoder(nn.Module):
    def __init__(self, embed_size, num_decoder_layers):
        super(CNNspecificDecoder, self).__init__()
        
        self.fuseComplementFeatures = ComplementFeatureFusionModule(embed_size*2)        
        # Decoder
        layers = []
        channels = [embed_size,embed_size//2,embed_size//4,1];
        lastOut = embed_size*2
        cur_depth = 0;
        for _ in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if (_==num_decoder_layers-1):
                    layers.append(nn.Tanh());
            else:
                layers.append(nn.ReLU(True))
            lastOut = channels[cur_depth];
            cur_depth += 1;
        self.decoder = nn.Sequential(*layers)        
    
    def forward(self,fea_com_fused):
        x = self.fuseComplementFeatures(fea_com_fused);
        x = self.decoder(x);
        x = x / 2 + 0.5;        
        return x;
        

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, isLast):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if (isLast == True):
            self.ac = nn.Tanh()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.ac(out)
        return out

#Shared Feature Extraction
class SharedFeatureExtractor(nn.Module):
    def __init__(self, s, n, channel, stride):
        super(SharedFeatureExtractor, self).__init__()
        # n = 64  # number of filters
        # s = 3  # filter size
        # Channel = 3
        self.n = n
        self.conv_1 = ConvLayer(channel*2, 32, s, stride, isLast = False)
        self.conv_2 = ConvLayer(32+2, 32, s, stride, isLast = False)
        self.conv_3 = ConvLayer(32+32+2, 32, s, stride, isLast = False)

    def forward(self, x):
        x_1 = self.conv_1(x)  # Z_0
        x_2 = self.conv_2(torch.cat((x,x_1),1))  # Z_0
        x_3 = self.conv_3(torch.cat((x,x_1,x_2),1))  # Z_0        
        return torch.cat((x,x_1,x_2,x_3),1)

#Shared Feature Extraction
class ReconstructionDecoder(nn.Module):
    def __init__(self, embed_size, num_decoder_layers):
        super(ReconstructionDecoder, self).__init__()
        # Decoder
        layers = []
        channels = [embed_size,embed_size//2,embed_size//4,1];
        lastOut = 32+32+32+2
        cur_depth = 0;
        for _ in range(num_decoder_layers):
            layers.append(nn.ReflectionPad2d(1))
            layers.append(nn.Conv2d(lastOut, channels[cur_depth], kernel_size=3, padding=0))
            if (_==num_decoder_layers-1):
                layers.append(nn.Tanh());
            else:
                layers.append(nn.ReLU(True))
            lastOut = channels[cur_depth];
            cur_depth += 1;
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x);
        return x


class TwoBranchesFusionNet(nn.Module):
    def __init__(self, s, n, channel, stride):
        super(TwoBranchesFusionNet, self).__init__()
        
        self.getSharedFeatures = SharedFeatureExtractor(s, n, channel, stride)        
        num_decoder_layers = 4;
        self.decoder_rec = ReconstructionDecoder(n,num_decoder_layers);                
        
        embed_size = int(args.n)
        heads = 4
        num_transformer_blocks = 2
        num_decoder_layers = 4

        self.extractor_multask = TransformerSpecificExtractor()
        self.cnnDecoder = CNNspecificDecoder(embed_size, num_decoder_layers);                        
        
    def forward_encoder(self, x, y):
        x = torch.cat((x,y),1)
        fea_x = self.getSharedFeatures(x)
        
        fea_com = fea_x;
        
        return fea_com;    
    
    #trainingTag = 1, IVIF task; trainingTag = 2, MFIF task;
    def forward_MultiTask_branch(self, fea_com_ivif, fea_com_mfif, trainingTag = 2):
        x = self.extractor_multask(fea_com_ivif, fea_com_mfif, trainingTag);
        return x;
    
    def forward_mixed_decoder(self, fea_com, fea_fused):
        x = self.cnnDecoder([fea_com,fea_fused]);
        return x;
        
    def forward_rec_decoder(self, fea_com):
        return self.decoder_rec(fea_com);                
        
    def forward(self, x, y):
        fea_com = self.forward_encoder(x, y);
        output = self.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com, trainingTag = 2)   
        return output
class StateSpaceModule(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D实数初始化
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D参数
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape

        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)

        # 卷积
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, "b d l -> b l d")

        # SSM
        x = F.silu(x)
        y = self.ssm(x)

        # 门控
        y = y * F.silu(z)

        # 输出投影
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """状态空间模型核心计算"""
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()

        B, L, D = x.shape
        deltaBC = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)

        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        # 选择性扫描算法
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        """选择性扫描算法的高效实现"""
        batch_size, L, inner_dim = u.shape
        N = A.shape[1]

        # 离散化
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

        # 递归计算
        x = torch.zeros((batch_size, inner_dim, N), device=u.device, dtype=u.dtype)
        y = []

        for i in range(L):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
            y_i = torch.sum(x * C[:, i].unsqueeze(1), dim=-1)
            y.append(y_i)

        y = torch.stack(y, dim=1)  # (B, L, D)
        y = y + u * D

        return y


class VisionMambaEncoder(nn.Module):
    """基于Vision Mamba的视觉特征编码器"""

    def __init__(self, dim, depth=4, patch_size=4):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, dim) * 0.02)

        # Mamba layers
        self.layers = nn.ModuleList([
            StateSpaceModule(dim) for _ in range(depth)
        ])

        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(depth)
        ])

        # 残差连接权重
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(depth)
        ])

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Patch embedding below has no padding. Pad explicitly so rectangular
        # and non-divisible inference sizes retain a valid 2-D token grid.
        pad_h = (-H) % self.patch_size
        pad_w = (-W) % self.patch_size
        if pad_h or pad_w:
            pad_mode = "reflect" if H > 1 and W > 1 else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        patch_h = (H + pad_h) // self.patch_size
        patch_w = (W + pad_w) // self.patch_size

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, C)
        N = x.shape[1]

        # Position embedding
        if N <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :N, :]
        else:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N,
                mode='linear'
            ).transpose(1, 2)

        x = x + pos_embed

        # Mamba layers with residual connections
        for layer, norm, weight in zip(self.layers, self.norm_layers, self.residual_weights):
            residual = x
            x = norm(x)
            x = layer(x)
            x = residual + weight * x

        # Reshape back to 2D
        x = rearrange(x, 'b (h w) c -> b c h w', h=patch_h, w=patch_w)

        # Interpolate to original spatial size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x


class AdvancedOrthogonalProjection(nn.Module):
    """基于变分推断和几何约束的正交投影"""

    def __init__(self, input_dim, content_dim, modality_dim, task_dim):
        super().__init__()
        self.input_dim = input_dim
        self.content_dim = content_dim
        self.modality_dim = modality_dim
        self.task_dim = task_dim

        # 变分编码器用于特征分解
        self.content_encoder = VisionMambaEncoder(input_dim, depth=3)
        self.modality_encoder = VisionMambaEncoder(input_dim, depth=3)
        self.task_encoder = VisionMambaEncoder(input_dim, depth=3)

        # 变分参数网络
        self.content_mu_net = nn.Sequential(
            nn.Conv2d(input_dim, content_dim * 2, 3, padding=1),
            nn.GroupNorm(8, content_dim * 2),
            nn.GELU(),
            nn.Conv2d(content_dim * 2, content_dim, 1)
        )

        self.content_logvar_net = nn.Sequential(
            nn.Conv2d(input_dim, content_dim * 2, 3, padding=1),
            nn.GroupNorm(8, content_dim * 2),
            nn.GELU(),
            nn.Conv2d(content_dim * 2, content_dim, 1)
        )

        self.modality_mu_net = nn.Sequential(
            nn.Conv2d(input_dim, modality_dim * 2, 3, padding=1),
            nn.GroupNorm(8, modality_dim * 2),
            nn.GELU(),
            nn.Conv2d(modality_dim * 2, modality_dim, 1)
        )

        self.modality_logvar_net = nn.Sequential(
            nn.Conv2d(input_dim, modality_dim * 2, 3, padding=1),
            nn.GroupNorm(8, modality_dim * 2),
            nn.GELU(),
            nn.Conv2d(modality_dim * 2, modality_dim, 1)
        )

        self.task_mu_net = nn.Sequential(
            nn.Conv2d(input_dim, task_dim * 2, 3, padding=1),
            nn.GroupNorm(8, task_dim * 2),
            nn.GELU(),
            nn.Conv2d(task_dim * 2, task_dim, 1)
        )

        self.task_logvar_net = nn.Sequential(
            nn.Conv2d(input_dim, task_dim * 2, 3, padding=1),
            nn.GroupNorm(8, task_dim * 2),
            nn.GELU(),
            nn.Conv2d(task_dim * 2, task_dim, 1)
        )

        # 几何约束网络
        self.geometric_constraint_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(content_dim + modality_dim + task_dim, 512, 1),
            nn.GELU(),
            nn.Conv2d(512, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, 9, 1)  # 3x3矩阵的上三角元素
        )

        # 正交性优化器
        self.orthogonal_optimizer = OrthogonalConstraintOptimizer()

        # 互信息估计网络
        self.mutual_info_estimator = MutualInformationEstimator(
            content_dim, modality_dim, task_dim
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl_divergence(self, mu, logvar):
        """计算KL散度"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def apply_geometric_constraints(self, content, modality, task):
        """应用几何约束确保正交性"""
        B, _, H, W = content.shape

        # 展平特征用于矩阵运算
        content_flat = content.view(B, self.content_dim, -1)
        modality_flat = modality.view(B, self.modality_dim, -1)
        task_flat = task.view(B, self.task_dim, -1)

        # 计算Gram矩阵
        gram_cm = torch.bmm(content_flat, modality_flat.transpose(1, 2))
        gram_ct = torch.bmm(content_flat, task_flat.transpose(1, 2))
        gram_mt = torch.bmm(modality_flat, task_flat.transpose(1, 2))

        # 几何约束网络
        combined = torch.cat([content, modality, task], dim=1)
        constraint_params = self.geometric_constraint_net(combined).view(B, 9)

        # 构建约束矩阵
        constraint_matrix = torch.zeros(B, 3, 3, device=content.device)
        triu_indices = torch.triu_indices(3, 3, offset=1)
        constraint_matrix[:, triu_indices[0], triu_indices[1]] = constraint_params

        # 应用约束优化
        content_opt, modality_opt, task_opt = self.orthogonal_optimizer(
            content_flat, modality_flat, task_flat, constraint_matrix
        )

        # 重塑回原始形状
        content_opt = content_opt.view(B, self.content_dim, H, W)
        modality_opt = modality_opt.view(B, self.modality_dim, H, W)
        task_opt = task_opt.view(B, self.task_dim, H, W)

        return content_opt, modality_opt, task_opt

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        """
        # 多尺度特征提取
        content_features = self.content_encoder(x)
        modality_features = self.modality_encoder(x)
        task_features = self.task_encoder(x)

        # 变分编码
        content_mu = self.content_mu_net(content_features)
        content_logvar = self.content_logvar_net(content_features)
        content = self.reparameterize(content_mu, content_logvar)

        modality_mu = self.modality_mu_net(modality_features)
        modality_logvar = self.modality_logvar_net(modality_features)
        modality = self.reparameterize(modality_mu, modality_logvar)

        task_mu = self.task_mu_net(task_features)
        task_logvar = self.task_logvar_net(task_features)
        task = self.reparameterize(task_mu, task_logvar)

        # 应用几何约束
        content_ortho, modality_ortho, task_ortho = self.apply_geometric_constraints(
            content, modality, task
        )

        # 计算各种损失
        content_kl = self.compute_kl_divergence(content_mu, content_logvar)
        modality_kl = self.compute_kl_divergence(modality_mu, modality_logvar)
        task_kl = self.compute_kl_divergence(task_mu, task_logvar)

        # 互信息损失
        mi_loss = self.mutual_info_estimator(content_ortho, modality_ortho, task_ortho)

        # 正交性约束损失
        orthogonal_loss = self.compute_advanced_orthogonal_loss(
            content_ortho, modality_ortho, task_ortho
        )

        losses = {
            'content_kl': content_kl,
            'modality_kl': modality_kl,
            'task_kl': task_kl,
            'mutual_info_loss': mi_loss,
            'orthogonal_loss': orthogonal_loss
        }

        return content_ortho, modality_ortho, task_ortho, losses

    def compute_advanced_orthogonal_loss(self, content, modality, task):
        """计算高级正交性损失"""
        B, _, H, W = content.shape

        # 展平特征
        content_flat = content.view(B, self.content_dim, -1)
        modality_flat = modality.view(B, self.modality_dim, -1)
        task_flat = task.view(B, self.task_dim, -1)

        # 归一化
        content_norm = F.normalize(content_flat, p=2, dim=1)
        modality_norm = F.normalize(modality_flat, p=2, dim=1)
        task_norm = F.normalize(task_flat, p=2, dim=1)

        # 计算余弦相似度矩阵
        sim_cm = torch.bmm(content_norm.transpose(1, 2), modality_norm)
        sim_ct = torch.bmm(content_norm.transpose(1, 2), task_norm)
        sim_mt = torch.bmm(modality_norm.transpose(1, 2), task_norm)

        # 正交性损失（期望余弦相似度为0）
        orthogonal_loss = (
                torch.mean(sim_cm ** 2) +
                torch.mean(sim_ct ** 2) +
                torch.mean(sim_mt ** 2)
        )

        return orthogonal_loss


class OrthogonalConstraintOptimizer(nn.Module):
    """正交约束优化器"""

    def __init__(self, max_iterations=5, tolerance=1e-6):
        super().__init__()
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def forward(self, content, modality, task, constraint_matrix):
        """应用迭代优化算法实现正交约束"""
        B, C_c, N = content.shape
        B, C_m, N = modality.shape
        B, C_t, N = task.shape

        # 初始化
        content_opt = content.clone()
        modality_opt = modality.clone()
        task_opt = task.clone()

        for iteration in range(self.max_iterations):
            # 保存上一次迭代的结果
            prev_content = content_opt.clone()
            prev_modality = modality_opt.clone()
            prev_task = task_opt.clone()

            # Gram-Schmidt正交化过程
            # 1. 正交化content相对于modality
            proj_cm = torch.bmm(
                torch.bmm(content_opt.transpose(1, 2), modality_opt),
                modality_opt.transpose(1, 2)
            ).transpose(1, 2)
            content_opt = content_opt - proj_cm

            # 2. 正交化content相对于task
            proj_ct = torch.bmm(
                torch.bmm(content_opt.transpose(1, 2), task_opt),
                task_opt.transpose(1, 2)
            ).transpose(1, 2)
            content_opt = content_opt - proj_ct

            # 3. 正交化modality相对于task
            proj_mt = torch.bmm(
                torch.bmm(modality_opt.transpose(1, 2), task_opt),
                task_opt.transpose(1, 2)
            ).transpose(1, 2)
            modality_opt = modality_opt - proj_mt

            # 归一化
            content_opt = F.normalize(content_opt, p=2, dim=1)
            modality_opt = F.normalize(modality_opt, p=2, dim=1)
            task_opt = F.normalize(task_opt, p=2, dim=1)

            # 检查收敛性
            content_diff = torch.norm(content_opt - prev_content, p='fro')
            modality_diff = torch.norm(modality_opt - prev_modality, p='fro')
            task_diff = torch.norm(task_opt - prev_task, p='fro')

            if (content_diff + modality_diff + task_diff) < self.tolerance:
                break

        return content_opt, modality_opt, task_opt


class MutualInformationEstimator(nn.Module):
    """互信息估计网络"""

    def __init__(self, content_dim, modality_dim, task_dim):
        super().__init__()

        # MINE网络用于估计互信息
        self.mine_cm = MINENetwork(content_dim, modality_dim)
        self.mine_ct = MINENetwork(content_dim, task_dim)
        self.mine_mt = MINENetwork(modality_dim, task_dim)

    def forward(self, content, modality, task):
        """计算互信息损失（最小化互信息以实现解耦）"""
        # 将2D特征展平
        B, C_c, H, W = content.shape
        content_flat = content.view(B, C_c, -1).mean(dim=2)  # Global average pooling
        modality_flat = modality.view(B, -1, H * W).mean(dim=2)
        task_flat = task.view(B, -1, H * W).mean(dim=2)

        # 计算互信息
        mi_cm = self.mine_cm(content_flat, modality_flat)
        mi_ct = self.mine_ct(content_flat, task_flat)
        mi_mt = self.mine_mt(modality_flat, task_flat)

        # 返回负互信息作为损失（最小化互信息）
        return -(mi_cm + mi_ct + mi_mt)


class MINENetwork(nn.Module):
    """MINE网络用于互信息估计"""

    def __init__(self, x_dim, y_dim, hidden_dim=512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        """
        估计x和y之间的互信息
        """
        batch_size = x.shape[0]

        # 正样本：配对的(x, y)
        pos_xy = torch.cat([x, y], dim=1)
        pos_scores = self.network(pos_xy)

        # 负样本：随机排列的y
        y_shuffle = y[torch.randperm(batch_size)]
        neg_xy = torch.cat([x, y_shuffle], dim=1)
        neg_scores = self.network(neg_xy)

        # MINE损失
        mi_estimate = torch.mean(pos_scores) - torch.log(torch.mean(torch.exp(neg_scores)))

        return mi_estimate


class GraphNeuralPrototypeEvolution(nn.Module):
    """基于图神经网络的高级原型演化机制"""

    def __init__(self, num_tasks, prototype_dim, num_prototypes_per_task=5, gnn_layers=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.prototype_dim = prototype_dim
        self.num_prototypes_per_task = num_prototypes_per_task
        self.total_prototypes = num_tasks * num_prototypes_per_task

        # 可学习的原型图
        self.register_buffer('prototype_features',
                             torch.randn(self.total_prototypes, prototype_dim))
        self.register_buffer('prototype_graph',
                             torch.eye(self.total_prototypes))

        # 图神经网络层
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(prototype_dim, prototype_dim // 2, dropout=0.1)
            for _ in range(gnn_layers)
        ])

        # 原型更新网络
        self.prototype_updater = MetaLearningPrototypeUpdater(prototype_dim)

        # 难样本挖掘网络
        self.hard_sample_miner = HardSampleMiner(prototype_dim)

        # 动态图学习
        self.graph_learner = DynamicGraphLearner(prototype_dim)

        # 原型聚类网络
        self.prototype_clusterer = PrototypeClustering(prototype_dim, num_tasks)

    def evolve_prototypes(self, task_features, task_id, hard_samples_mask):
        """使用图神经网络演化原型"""
        # 获取当前任务的原型索引
        start_idx = task_id * self.num_prototypes_per_task
        end_idx = start_idx + self.num_prototypes_per_task

        # 提取当前原型
        current_prototypes = self.prototype_features[start_idx:end_idx]

        # 动态更新图结构
        updated_graph = self.graph_learner(self.prototype_features, self.prototype_graph)

        # 通过GNN传播信息
        evolved_features = self.prototype_features.clone()
        for gnn_layer in self.gnn_layers:
            evolved_features = gnn_layer(evolved_features, updated_graph)

        # 提取演化后的原型
        evolved_prototypes = evolved_features[start_idx:end_idx]

        # 使用元学习更新原型
        if hard_samples_mask.any():
            hard_samples = task_features[hard_samples_mask]
            updated_prototypes = self.prototype_updater(
                evolved_prototypes, hard_samples, current_prototypes
            )
        else:
            updated_prototypes = evolved_prototypes

        # 更新原型特征
        with torch.no_grad():
            self.prototype_features[start_idx:end_idx] = updated_prototypes
            self.prototype_graph = updated_graph

        return updated_prototypes

    def forward(self, task_features, task_id):
        """
        Args:
            task_features: [B, D] 任务特征
            task_id: 任务ID
        """
        B, D = task_features.shape

        # 难样本挖掘
        hard_samples_mask, difficulty_scores = self.hard_sample_miner(
            task_features, task_id, self.prototype_features
        )

        # 演化原型
        evolved_prototypes = self.evolve_prototypes(
            task_features, task_id, hard_samples_mask
        )

        # 计算原型对比损失
        prototype_loss = self.compute_prototype_contrastive_loss(
            task_features, task_id, evolved_prototypes
        )

        # 原型聚类损失
        clustering_loss = self.prototype_clusterer(self.prototype_features, task_id)

        return {
            'prototype_loss': prototype_loss,
            'clustering_loss': clustering_loss,
            'hard_samples_mask': hard_samples_mask,
            'difficulty_scores': difficulty_scores,
            'evolved_prototypes': evolved_prototypes
        }

    def compute_prototype_contrastive_loss(self, features, task_id, prototypes):
        """计算原型对比损失"""
        # 获取所有任务的原型作为负样本
        all_prototypes = self.prototype_features  # [total_prototypes, D]

        # 当前任务原型作为正样本
        start_idx = task_id * self.num_prototypes_per_task
        end_idx = start_idx + self.num_prototypes_per_task
        positive_prototypes = prototypes  # [num_prototypes_per_task, D]

        # 计算特征与所有原型的相似度
        features_norm = F.normalize(features, p=2, dim=1)
        all_prototypes_norm = F.normalize(all_prototypes, p=2, dim=1)

        similarities = torch.mm(features_norm, all_prototypes_norm.T)  # [B, total_prototypes]

        # 构建标签（正样本对应当前任务的原型）
        labels = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)

        # 使用多标签对比损失
        pos_similarities = similarities[:, start_idx:end_idx]  # [B, num_prototypes_per_task]
        neg_similarities = torch.cat([
            similarities[:, :start_idx],
            similarities[:, end_idx:]
        ], dim=1)  # [B, total_prototypes - num_prototypes_per_task]

        # 计算InfoNCE损失
        temperature = 0.07
        pos_exp = torch.exp(pos_similarities / temperature)
        neg_exp = torch.exp(neg_similarities / temperature)

        pos_sum = torch.sum(pos_exp, dim=1)
        neg_sum = torch.sum(neg_exp, dim=1)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum)).mean()

        return loss


class GraphAttentionLayer(nn.Module):
    """图注意力层"""

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
        Args:
            h: 节点特征 [N, in_features]
            adj: 邻接矩阵 [N, N]
        """
        Wh = torch.mm(h, self.W)  # [N, out_features]
        N = Wh.size()[0]

        # 计算注意力系数
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), negative_slope=self.alpha)

        # 应用掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 应用注意力
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class MetaLearningPrototypeUpdater(nn.Module):
    """基于元学习的原型更新器"""

    def __init__(self, prototype_dim, meta_lr=0.01):
        super().__init__()
        self.prototype_dim = prototype_dim
        self.meta_lr = meta_lr

        # 元学习网络
        self.meta_network = nn.Sequential(
            nn.Linear(prototype_dim * 3, prototype_dim * 2),
            nn.ReLU(True),
            nn.Linear(prototype_dim * 2, prototype_dim),
            nn.Tanh()
        )

        # LSTM用于序列更新
        self.lstm = nn.LSTM(prototype_dim, prototype_dim, batch_first=True)

        # 注意力机制
        self.attention = nn.MultiheadAttention(prototype_dim, num_heads=8, batch_first=True)

    def forward(self, current_prototypes, hard_samples, old_prototypes):
        """
        Args:
            current_prototypes: [num_prototypes, D]
            hard_samples: [num_hard_samples, D]
            old_prototypes: [num_prototypes, D]
        """
        num_prototypes = current_prototypes.shape[0]

        if hard_samples.shape[0] == 0:
            return current_prototypes

        # 计算样本到原型的相似度
        similarities = torch.mm(
            F.normalize(hard_samples, p=2, dim=1),
            F.normalize(current_prototypes, p=2, dim=1).T
        )

        # 为每个原型分配最相似的样本
        _, closest_samples_idx = torch.max(similarities, dim=0)

        updated_prototypes = []
        for i in range(num_prototypes):
            sample_idx = closest_samples_idx[i]
            closest_sample = hard_samples[sample_idx:sample_idx + 1]

            # 准备元学习输入
            meta_input = torch.cat([
                current_prototypes[i:i + 1],
                closest_sample,
                old_prototypes[i:i + 1]
            ], dim=1)

            # 元学习更新
            update_direction = self.meta_network(meta_input)
            updated_prototype = current_prototypes[i:i + 1] + self.meta_lr * update_direction

            updated_prototypes.append(updated_prototype)

        updated_prototypes = torch.cat(updated_prototypes, dim=0)

        # 使用LSTM进行序列更新
        prototype_sequence = torch.stack([old_prototypes, current_prototypes, updated_prototypes], dim=1)
        lstm_out, _ = self.lstm(prototype_sequence.unsqueeze(0))
        lstm_prototypes = lstm_out.squeeze(0)[-1]  # 取最后一个时间步

        # 使用注意力机制融合信息
        attn_out, _ = self.attention(
            updated_prototypes.unsqueeze(0),
            lstm_prototypes.unsqueeze(0),
            lstm_prototypes.unsqueeze(0)
        )
        final_prototypes = attn_out.squeeze(0)

        return final_prototypes


class HardSampleMiner(nn.Module):
    """高级难样本挖掘器"""

    def __init__(self, feature_dim, uncertainty_threshold=0.7):
        super().__init__()
        self.feature_dim = feature_dim
        self.uncertainty_threshold = uncertainty_threshold

        # 不确定性估计网络
        self.uncertainty_estimator = UncertaintyEstimator(feature_dim)

        # 难度评分网络
        self.difficulty_scorer = DifficultyScorer(feature_dim)

        # 自适应阈值学习
        self.threshold_learner = AdaptiveThresholdLearner(feature_dim)

    def forward(self, features, task_id, prototypes):
        """
        Args:
            features: [B, D] 样本特征
            task_id: 任务ID
            prototypes: [total_prototypes, D] 所有原型
        """
        B, D = features.shape

        # 估计不确定性
        uncertainty_scores = self.uncertainty_estimator(features)

        # 计算难度分数
        difficulty_scores = self.difficulty_scorer(features, prototypes, task_id)

        # 学习自适应阈值
        adaptive_threshold = self.threshold_learner(features, uncertainty_scores, difficulty_scores)

        # 综合评分
        combined_scores = uncertainty_scores * 0.5 + difficulty_scores * 0.5

        # 确定难样本
        hard_samples_mask = combined_scores > adaptive_threshold

        return hard_samples_mask, combined_scores


class UncertaintyEstimator(nn.Module):
    """不确定性估计器"""

    def __init__(self, feature_dim, num_monte_carlo=10):
        super().__init__()
        self.num_monte_carlo = num_monte_carlo

        # 贝叶斯神经网络
        self.bayesian_net = nn.Sequential(
            BayesianLinear(feature_dim, feature_dim // 2),
            nn.ReLU(True),
            BayesianLinear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(True),
            BayesianLinear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """使用Monte Carlo Dropout估计不确定性"""
        self.train()  # 启用dropout

        predictions = []
        for _ in range(self.num_monte_carlo):
            pred = self.bayesian_net(features)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [num_mc, B, 1]

        # 计算预测方差作为不确定性
        uncertainty = torch.var(predictions, dim=0).squeeze(1)  # [B]

        return uncertainty


class BayesianLinear(nn.Module):
    """贝叶斯线性层"""

    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var

        # 权重参数
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # 偏置参数
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)

    def forward(self, x):
        # 重参数化
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

        bias_std = torch.log1p(torch.exp(self.bias_rho))
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        return F.linear(x, weight, bias)


class DifficultyScorer(nn.Module):
    """难度评分器"""

    def __init__(self, feature_dim):
        super().__init__()

        # 原型距离计算网络
        self.distance_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

        # 特征复杂度评估网络
        self.complexity_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features, prototypes, task_id):
        """
        计算样本的难度分数
        """
        B, D = features.shape

        # 找到最近的原型
        similarities = torch.mm(
            F.normalize(features, p=2, dim=1),
            F.normalize(prototypes, p=2, dim=1).T
        )
        max_similarities, closest_prototype_idx = torch.max(similarities, dim=1)

        # 获取最近原型
        closest_prototypes = prototypes[closest_prototype_idx]  # [B, D]

        # 计算原型距离分数
        distance_input = torch.cat([features, closest_prototypes], dim=1)
        distance_scores = self.distance_net(distance_input).squeeze(1)

        # 计算特征复杂度分数
        complexity_scores = self.complexity_net(features).squeeze(1)

        # 综合难度分数
        difficulty_scores = 0.6 * (1 - max_similarities) + 0.4 * complexity_scores

        return difficulty_scores


class AdaptiveThresholdLearner(nn.Module):
    """自适应阈值学习器"""

    def __init__(self, feature_dim):
        super().__init__()

        self.threshold_net = nn.Sequential(
            nn.Linear(feature_dim + 2, feature_dim // 2),  # +2 for uncertainty and difficulty
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

        # 历史统计信息
        self.register_buffer('running_mean_uncertainty', torch.tensor(0.5))
        self.register_buffer('running_mean_difficulty', torch.tensor(0.5))
        self.momentum = 0.9

    def forward(self, features, uncertainty_scores, difficulty_scores):
        """学习自适应阈值"""
        B = features.shape[0]

        # 更新运行统计
        if self.training:
            with torch.no_grad():
                current_mean_uncertainty = torch.mean(uncertainty_scores)
                current_mean_difficulty = torch.mean(difficulty_scores)

                self.running_mean_uncertainty = (
                        self.momentum * self.running_mean_uncertainty +
                        (1 - self.momentum) * current_mean_uncertainty
                )
                self.running_mean_difficulty = (
                        self.momentum * self.running_mean_difficulty +
                        (1 - self.momentum) * current_mean_difficulty
                )

        # 构建输入特征
        global_uncertainty = self.running_mean_uncertainty.expand(B, 1)
        global_difficulty = self.running_mean_difficulty.expand(B, 1)

        threshold_input = torch.cat([
            features,
            global_uncertainty,
            global_difficulty
        ], dim=1)

        # 预测自适应阈值
        adaptive_thresholds = self.threshold_net(threshold_input).squeeze(1)

        return adaptive_thresholds


class DynamicGraphLearner(nn.Module):
    """动态图学习器"""

    def __init__(self, feature_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        # 图学习网络
        self.graph_learner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(True),
            nn.Linear(feature_dim // 4, feature_dim // 4)
        )

        # 注意力权重网络
        self.attention_net = nn.MultiheadAttention(
            embed_dim=feature_dim // 4,
            num_heads=4,
            batch_first=True
        )

    def forward(self, prototype_features, current_graph):
        """
        学习动态图结构
        Args:
            prototype_features: [N, D] 原型特征
            current_graph: [N, N] 当前图邻接矩阵
        """
        N, D = prototype_features.shape

        # 学习节点表示
        node_representations = self.graph_learner(prototype_features)  # [N, D//4]

        # 使用注意力机制学习边权重
        attn_out, attn_weights = self.attention_net(
            node_representations.unsqueeze(0),
            node_representations.unsqueeze(0),
            node_representations.unsqueeze(0)
        )

        # 提取注意力权重作为新的邻接矩阵
        new_adjacency = attn_weights.squeeze(0)  # [N, N]

        # 应用温度缩放
        new_adjacency = new_adjacency / self.temperature
        new_adjacency = F.softmax(new_adjacency, dim=1)

        # 与当前图结合
        momentum = 0.8
        updated_graph = momentum * current_graph + (1 - momentum) * new_adjacency

        # 确保对称性
        updated_graph = (updated_graph + updated_graph.T) / 2

        return updated_graph


class PrototypeClustering(nn.Module):
    """原型聚类器"""

    def __init__(self, prototype_dim, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

        # 聚类中心
        self.cluster_centers = nn.Parameter(torch.randn(num_tasks, prototype_dim))

        # 聚类网络
        self.clustering_net = nn.Sequential(
            nn.Linear(prototype_dim, prototype_dim // 2),
            nn.ReLU(True),
            nn.Linear(prototype_dim // 2, num_tasks),
            nn.Softmax(dim=1)
        )

    def forward(self, prototype_features, task_id):
        """
        执行原型聚类
        """
        # 预测聚类分配
        cluster_probs = self.clustering_net(prototype_features)  # [N, num_tasks]

        # 计算聚类损失
        target_cluster = torch.zeros_like(cluster_probs)

        # 为每个任务的原型分配到对应的聚类
        num_prototypes_per_task = prototype_features.shape[0] // self.num_tasks
        for i in range(self.num_tasks):
            start_idx = i * num_prototypes_per_task
            end_idx = start_idx + num_prototypes_per_task
            target_cluster[start_idx:end_idx, i] = 1.0

        # 交叉熵损失
        clustering_loss = F.kl_div(
            torch.log(cluster_probs + 1e-8),
            target_cluster,
            reduction='batchmean'
        )

        return clustering_loss


class AdvancedMultiGrainContrastiveLearning(nn.Module):
    """基于2024年最新技术的高级多粒度对比学习"""

    def __init__(self, feature_dim, num_scales=5, temperature=0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.temperature = temperature

        # 多尺度特征提取器
        self.multi_scale_extractors = nn.ModuleList([
            VisionMambaEncoder(feature_dim, depth=2, patch_size=2 ** i)
            for i in range(num_scales)
        ])

        # 跨尺度注意力融合
        self.cross_scale_attention = CrossScaleAttentionFusion(feature_dim, num_scales)

        # 自监督对比学习
        self.self_supervised_contrastive = SelfSupervisedContrastiveLearning(feature_dim)

        # 负样本挖掘器
        self.hard_negative_miner = HardNegativeMiner(feature_dim)

        # 温度自适应学习
        self.temperature_learner = TemperatureLearner(feature_dim)

    def forward(self, features1, features2):
        """
        高级多粒度对比学习
        Args:
            features1: 第一模态特征 [B, C, H, W]
            features2: 第二模态特征 [B, C, H, W]
        """
        B, C, H, W = features1.shape

        # 多尺度特征提取
        multi_scale_features1 = []
        multi_scale_features2 = []

        for extractor in self.multi_scale_extractors:
            ms_feat1 = extractor(features1)
            ms_feat2 = extractor(features2)
            multi_scale_features1.append(ms_feat1)
            multi_scale_features2.append(ms_feat2)

        # 跨尺度注意力融合
        fused_features1 = self.cross_scale_attention(multi_scale_features1)
        fused_features2 = self.cross_scale_attention(multi_scale_features2)

        # 自监督对比学习
        ssl_loss1 = self.self_supervised_contrastive(fused_features1)
        ssl_loss2 = self.self_supervised_contrastive(fused_features2)

        # 学习自适应温度
        adaptive_temperature = self.temperature_learner(fused_features1, fused_features2)

        # 难负样本挖掘
        hard_negatives1, hard_negatives2 = self.hard_negative_miner(
            fused_features1, fused_features2
        )

        # 计算多粒度对比损失
        contrastive_losses = self.compute_multi_grain_contrastive_loss(
            fused_features1, fused_features2,
            hard_negatives1, hard_negatives2,
            adaptive_temperature
        )

        total_loss = (
                contrastive_losses['global_loss'] +
                contrastive_losses['local_loss'] +
                contrastive_losses['cross_modal_loss'] +
                ssl_loss1 + ssl_loss2
        )

        return total_loss, {
            'contrastive_losses': contrastive_losses,
            'ssl_loss': ssl_loss1 + ssl_loss2,
            'adaptive_temperature': adaptive_temperature,
            'fused_features1': fused_features1,
            'fused_features2': fused_features2
        }

    def compute_multi_grain_contrastive_loss(self, features1, features2, hard_neg1, hard_neg2, temperature):
        """计算多粒度对比损失"""
        losses = {}

        # 全局对比损失
        global_feat1 = F.adaptive_avg_pool2d(features1, 1).flatten(1)
        global_feat2 = F.adaptive_avg_pool2d(features2, 1).flatten(1)
        losses['global_loss'] = self.infonce_loss(global_feat1, global_feat2, temperature)

        # 局部对比损失
        local_feat1 = features1.flatten(2).transpose(1, 2)  # [B, HW, C]
        local_feat2 = features2.flatten(2).transpose(1, 2)  # [B, HW, C]
        losses['local_loss'] = self.dense_contrastive_loss(local_feat1, local_feat2, temperature)

        # 跨模态对比损失
        losses['cross_modal_loss'] = self.cross_modal_contrastive_loss(
            features1, features2, hard_neg1, hard_neg2, temperature
        )

        return losses

    def infonce_loss(self, anchor, positive, temperature):
        """InfoNCE损失"""
        batch_size = anchor.shape[0]

        # 归一化
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        # 计算相似度
        pos_sim = torch.sum(anchor * positive, dim=1) / temperature

        # 负样本相似度
        neg_sim = torch.mm(anchor, positive.T) / temperature

        # 构建logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def dense_contrastive_loss(self, features1, features2, temperature):
        """密集对比损失"""
        B, N, C = features1.shape

        # 重塑为[B*N, C]
        feat1_flat = features1.view(-1, C)
        feat2_flat = features2.view(-1, C)

        # 归一化
        feat1_norm = F.normalize(feat1_flat, p=2, dim=1)
        feat2_norm = F.normalize(feat2_flat, p=2, dim=1)

        # 正样本对：相同位置的特征
        pos_sim = torch.sum(feat1_norm * feat2_norm, dim=1) / temperature

        # 负样本：不同位置的特征
        neg_sim = torch.mm(feat1_norm, feat2_norm.T) / temperature

        # 移除对角线元素（正样本）
        mask = torch.eye(feat1_norm.shape[0], device=feat1_norm.device).bool()
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))

        # 计算损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(feat1_norm.shape[0], dtype=torch.long, device=feat1_norm.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def cross_modal_contrastive_loss(self, features1, features2, hard_neg1, hard_neg2, temperature):
        """跨模态对比损失"""
        # 全局特征
        global1 = F.adaptive_avg_pool2d(features1, 1).flatten(1)
        global2 = F.adaptive_avg_pool2d(features2, 1).flatten(1)

        # 归一化
        global1_norm = F.normalize(global1, p=2, dim=1)
        global2_norm = F.normalize(global2, p=2, dim=1)
        hard_neg1_norm = F.normalize(hard_neg1, p=2, dim=1)
        hard_neg2_norm = F.normalize(hard_neg2, p=2, dim=1)

        # 正样本相似度
        pos_sim = torch.sum(global1_norm * global2_norm, dim=1) / temperature

        # 难负样本相似度
        neg_sim1 = torch.sum(global1_norm * hard_neg1_norm, dim=1) / temperature
        neg_sim2 = torch.sum(global2_norm * hard_neg2_norm, dim=1) / temperature

        # InfoNCE损失
        logits = torch.stack([pos_sim, neg_sim1, neg_sim2], dim=1)
        labels = torch.zeros(global1.shape[0], dtype=torch.long, device=global1.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class CrossScaleAttentionFusion(nn.Module):
    """跨尺度注意力融合"""

    def __init__(self, feature_dim, num_scales):
        super().__init__()
        self.num_scales = num_scales

        # 尺度权重学习
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim * num_scales, feature_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(feature_dim, num_scales, 1),
            nn.Softmax(dim=1)
        )

        # 特征融合网络
        self.fusion_net = nn.Sequential(
            nn.Conv2d(feature_dim * num_scales, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 1)
        )

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: List of [B, C, H, W] features at different scales
        """
        # 调整所有特征到相同尺寸
        target_size = multi_scale_features[0].shape[2:]
        aligned_features = []

        for feat in multi_scale_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        # 连接所有尺度的特征
        concatenated = torch.cat(aligned_features, dim=1)

        # 学习尺度注意力权重
        scale_weights = self.scale_attention(concatenated)  # [B, num_scales, 1, 1]

        # 加权融合
        weighted_features = []
        for i, feat in enumerate(aligned_features):
            weight = scale_weights[:, i:i + 1, :, :]
            weighted_features.append(feat * weight)

        weighted_sum = torch.stack(weighted_features, dim=1).sum(dim=1)

        # 特征融合
        fused_features = self.fusion_net(concatenated) + weighted_sum

        return fused_features


class SelfSupervisedContrastiveLearning(nn.Module):
    """自监督对比学习"""

    def __init__(self, feature_dim):
        super().__init__()

        # 数据增强网络
        self.augmentation_net = DataAugmentationNetwork(feature_dim)

        # 投影头
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(True),
            nn.Linear(feature_dim // 4, 128)
        )

    def forward(self, features):
        """
        自监督对比学习
        Args:
            features: [B, C, H, W]
        """
        # 生成增强视图
        aug_features1 = self.augmentation_net(features, augmentation_type='strong')
        aug_features2 = self.augmentation_net(features, augmentation_type='weak')

        # 投影到表示空间
        proj1 = self.projection_head(aug_features1)
        proj2 = self.projection_head(aug_features2)
        proj_orig = self.projection_head(features)

        # 计算SimCLR损失
        ssl_loss = self.simclr_loss(proj_orig, proj1, proj2)

        return ssl_loss

    def simclr_loss(self, anchor, pos1, pos2, temperature=0.5):
        """SimCLR损失"""
        batch_size = anchor.shape[0]

        # 归一化
        anchor = F.normalize(anchor, p=2, dim=1)
        pos1 = F.normalize(pos1, p=2, dim=1)
        pos2 = F.normalize(pos2, p=2, dim=1)

        # 正样本相似度
        pos_sim1 = torch.sum(anchor * pos1, dim=1) / temperature
        pos_sim2 = torch.sum(anchor * pos2, dim=1) / temperature

        # 负样本相似度（同批次其他样本）
        neg_sim1 = torch.mm(anchor, pos1.T) / temperature
        neg_sim2 = torch.mm(anchor, pos2.T) / temperature

        # 移除对角线（正样本）
        mask = torch.eye(batch_size, device=anchor.device).bool()
        neg_sim1 = neg_sim1.masked_fill(mask, float('-inf'))
        neg_sim2 = neg_sim2.masked_fill(mask, float('-inf'))

        # 计算损失
        logits1 = torch.cat([pos_sim1.unsqueeze(1), neg_sim1], dim=1)
        logits2 = torch.cat([pos_sim2.unsqueeze(1), neg_sim2], dim=1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        loss1 = F.cross_entropy(logits1, labels)
        loss2 = F.cross_entropy(logits2, labels)

        return (loss1 + loss2) / 2


class DataAugmentationNetwork(nn.Module):
    """可学习的数据增强网络"""

    def __init__(self, feature_dim):
        super().__init__()

        # 空间变换网络
        self.spatial_transformer = SpatialTransformerNetwork(feature_dim)

        # 特征变换网络
        self.feature_transformer = FeatureTransformerNetwork(feature_dim)

        # 噪声注入网络
        self.noise_injector = NoiseInjectionNetwork(feature_dim)

    def forward(self, features, augmentation_type='weak'):
        """
        可学习的数据增强
        """
        if augmentation_type == 'weak':
            # 弱增强：轻微的空间变换
            aug_features = self.spatial_transformer(features, strength=0.1)
            aug_features = self.noise_injector(aug_features, noise_level=0.05)
        else:
            # 强增强：强烈的变换
            aug_features = self.spatial_transformer(features, strength=0.3)
            aug_features = self.feature_transformer(aug_features)
            aug_features = self.noise_injector(aug_features, noise_level=0.15)

        return aug_features


class SpatialTransformerNetwork(nn.Module):
    """空间变换网络"""

    def __init__(self, feature_dim):
        super().__init__()

        # 定位网络
        self.localization = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, 6)  # 仿射变换参数
        )

        # 初始化为恒等变换
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, features, strength=0.1):
        """
        应用空间变换
        """
        batch_size = features.shape[0]

        # 预测变换参数
        theta = self.localization(features).view(batch_size, 2, 3)

        # 添加随机性
        if self.training:
            noise = torch.randn_like(theta) * strength
            theta = theta + noise

        # 生成采样网格
        grid = F.affine_grid(theta, features.size(), align_corners=False)

        # 应用变换
        transformed = F.grid_sample(features, grid, align_corners=False)

        return transformed


class FeatureTransformerNetwork(nn.Module):
    """特征变换网络"""

    def __init__(self, feature_dim):
        super().__init__()

        self.channel_transformer = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 1)
        )

        self.spatial_transformer = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim),
            nn.Conv2d(feature_dim, feature_dim, 1)
        )

    def forward(self, features):
        """特征空间变换"""
        # 通道维度变换
        channel_transformed = self.channel_transformer(features)

        # 空间维度变换
        spatial_transformed = self.spatial_transformer(features)

        # 残差连接
        transformed = features + 0.1 * (channel_transformed + spatial_transformed)

        return transformed


class NoiseInjectionNetwork(nn.Module):
    """噪声注入网络"""

    def __init__(self, feature_dim):
        super().__init__()

        # 自适应噪声生成器
        self.noise_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(feature_dim // 4, feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features, noise_level=0.1):
        """注入自适应噪声"""
        if not self.training:
            return features

        # 生成自适应噪声强度
        noise_strength = self.noise_generator(features) * noise_level

        # 生成噪声
        noise = torch.randn_like(features) * noise_strength

        # 注入噪声
        noisy_features = features + noise

        return noisy_features


class HardNegativeMiner(nn.Module):
    """难负样本挖掘器"""

    def __init__(self, feature_dim, num_negatives=5):
        super().__init__()
        self.num_negatives = num_negatives

        # 相似度评估网络
        self.similarity_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features1, features2):
        """挖掘难负样本"""
        batch_size = features1.shape[0]

        # 计算全局特征
        global1 = F.adaptive_avg_pool2d(features1, 1).flatten(1)
        global2 = F.adaptive_avg_pool2d(features2, 1).flatten(1)

        # 生成候选负样本
        neg_candidates1 = global1[torch.randperm(batch_size)]
        neg_candidates2 = global2[torch.randperm(batch_size)]

        # 评估相似度
        similarities1 = self.similarity_net(torch.cat([global1, neg_candidates1], dim=1))
        similarities2 = self.similarity_net(torch.cat([global2, neg_candidates2], dim=1))

        # 选择最相似的作为难负样本
        _, hard_indices1 = torch.topk(similarities1.squeeze(1), k=min(self.num_negatives, batch_size))
        _, hard_indices2 = torch.topk(similarities2.squeeze(1), k=min(self.num_negatives, batch_size))

        hard_negatives1 = neg_candidates1[hard_indices1[0]]  # 取最难的一个
        hard_negatives2 = neg_candidates2[hard_indices2[0]]

        return hard_negatives1, hard_negatives2


class TemperatureLearner(nn.Module):
    """温度自适应学习器"""

    def __init__(self, feature_dim, init_temperature=0.07):
        super().__init__()

        self.init_temperature = init_temperature

        # 温度预测网络
        self.temperature_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features1, features2):
        """学习自适应温度"""
        # 全局特征
        global1 = F.adaptive_avg_pool2d(features1, 1).flatten(1)
        global2 = F.adaptive_avg_pool2d(features2, 1).flatten(1)

        # 连接特征
        combined = torch.cat([global1, global2], dim=1)

        # 预测温度调整因子
        temp_factor = self.temperature_net(combined).squeeze(1)

        # 计算自适应温度
        adaptive_temp = self.init_temperature * (0.5 + temp_factor)  # 范围[0.035, 0.105]

        return adaptive_temp.mean()  # 返回批次平均温度


# ===== 算法实现，不使用卷积网络代替 =====

class TrueGramSchmidtOrthogonalization(nn.Module):
    """Gram-Schmidt正交化算法实现"""

    def __init__(self, tolerance=1e-8, max_iterations=100):
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def gram_schmidt_process(self, vectors):
        """
        Gram-Schmidt正交化过程
        Args:
            vectors: 输入向量组 [num_vectors, vector_dim]
        Returns:
            orthogonal_vectors: 正交化后的向量组
        """
        num_vectors, vector_dim = vectors.shape
        orthogonal_vectors = torch.zeros_like(vectors)

        for i in range(num_vectors):
            # 取当前向量
            current_vector = vectors[i].clone()

            # 减去所有之前正交向量的投影
            for j in range(i):
                # 计算投影: proj_u_v = (v·u / u·u) * u
                prev_orthogonal = orthogonal_vectors[j]
                dot_product = torch.dot(current_vector, prev_orthogonal)
                norm_squared = torch.dot(prev_orthogonal, prev_orthogonal)

                if norm_squared > self.tolerance:
                    projection = (dot_product / norm_squared) * prev_orthogonal
                    current_vector = current_vector - projection

            # 归一化当前向量
            norm = torch.norm(current_vector)
            if norm > self.tolerance:
                current_vector = current_vector / norm
            else:
                # 如果向量变为零向量，生成一个随机正交向量
                current_vector = self._generate_orthogonal_vector(orthogonal_vectors[:i])

            orthogonal_vectors[i] = current_vector

        return orthogonal_vectors

    def _generate_orthogonal_vector(self, existing_vectors):
        """生成与现有向量正交的随机向量"""
        if len(existing_vectors) == 0:
            return torch.randn(existing_vectors.shape[1]) if len(existing_vectors.shape) > 1 else torch.randn(1)

        vector_dim = existing_vectors.shape[1]
        max_attempts = 100

        for _ in range(max_attempts):
            # 生成随机向量
            random_vector = torch.randn(vector_dim, device=existing_vectors.device)

            # 正交化
            for existing_vector in existing_vectors:
                dot_product = torch.dot(random_vector, existing_vector)
                norm_squared = torch.dot(existing_vector, existing_vector)
                if norm_squared > self.tolerance:
                    projection = (dot_product / norm_squared) * existing_vector
                    random_vector = random_vector - projection

            # 检查是否成功生成正交向量
            norm = torch.norm(random_vector)
            if norm > self.tolerance:
                return random_vector / norm

        # 如果失败，返回标准基向量
        basis_vector = torch.zeros(vector_dim, device=existing_vectors.device)
        basis_vector[len(existing_vectors) % vector_dim] = 1.0
        return basis_vector

    def batch_orthogonalization(self, feature_maps):
        """
        对特征图进行批量正交化
        Args:
            feature_maps: [B, C, H, W]
        """
        B, C, H, W = feature_maps.shape
        orthogonalized_maps = torch.zeros_like(feature_maps)

        for b in range(B):
            for h in range(H):
                for w in range(W):
                    # 提取空间位置(h,w)处所有通道的特征向量
                    feature_vector = feature_maps[b, :, h, w]  # [C]

                    # 如果这是第一个位置，直接归一化
                    if h == 0 and w == 0:
                        norm = torch.norm(feature_vector)
                        if norm > self.tolerance:
                            orthogonalized_maps[b, :, h, w] = feature_vector / norm
                        else:
                            orthogonalized_maps[b, :, h, w] = feature_vector
                    else:
                        # 收集之前所有位置的正交化向量
                        prev_vectors = []
                        for prev_h in range(h + 1):
                            for prev_w in range(w + 1):
                                if prev_h == h and prev_w >= w:
                                    break
                                prev_vector = orthogonalized_maps[b, :, prev_h, prev_w]
                                if torch.norm(prev_vector) > self.tolerance:
                                    prev_vectors.append(prev_vector)

                        # 对当前向量进行正交化
                        current_vector = feature_vector.clone()
                        for prev_vector in prev_vectors:
                            dot_product = torch.dot(current_vector, prev_vector)
                            norm_squared = torch.dot(prev_vector, prev_vector)
                            if norm_squared > self.tolerance:
                                projection = (dot_product / norm_squared) * prev_vector
                                current_vector = current_vector - projection

                        # 归一化
                        norm = torch.norm(current_vector)
                        if norm > self.tolerance:
                            orthogonalized_maps[b, :, h, w] = current_vector / norm
                        else:
                            orthogonalized_maps[b, :, h, w] = current_vector

        return orthogonalized_maps

    def verify_orthogonality(self, vectors, threshold=1e-6):
        """验证向量组的正交性"""
        num_vectors = vectors.shape[0]
        max_dot_product = 0.0

        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                dot_product = torch.abs(torch.dot(vectors[i], vectors[j]))
                max_dot_product = max(max_dot_product, dot_product.item())

        is_orthogonal = max_dot_product < threshold
        return is_orthogonal, max_dot_product


class TrueGaborFilterBank(nn.Module):
    """Gabor滤波器实现"""

    def __init__(self, orientations=8, frequencies=[0.1, 0.2, 0.3, 0.4],
                 kernel_size=31, sigma_x=4.0, sigma_y=4.0):
        super().__init__()
        self.orientations = orientations
        self.frequencies = frequencies
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        # 预计算所有Gabor核
        self.gabor_kernels_real, self.gabor_kernels_imag = self._compute_gabor_kernels()

    def _compute_gabor_kernels(self):
        """计算Gabor滤波器核"""
        kernels_real = []
        kernels_imag = []

        # 创建坐标网格
        center = self.kernel_size // 2
        x = torch.arange(self.kernel_size, dtype=torch.float32) - center
        y = torch.arange(self.kernel_size, dtype=torch.float32) - center
        X, Y = torch.meshgrid(x, y, indexing='ij')

        for freq in self.frequencies:
            for orient_idx in range(self.orientations):
                # 计算方向角
                theta = orient_idx * math.pi / self.orientations

                # 旋转坐标系
                x_theta = X * math.cos(theta) + Y * math.sin(theta)
                y_theta = -X * math.sin(theta) + Y * math.cos(theta)

                # 计算高斯包络
                gaussian_envelope = torch.exp(
                    -(x_theta ** 2 / (2 * self.sigma_x ** 2) +
                      y_theta ** 2 / (2 * self.sigma_y ** 2))
                )

                # 计算复数Gabor函数
                # 实部：cos(2πfx')
                # 虚部：sin(2πfx')
                sinusoid_real = torch.cos(2 * math.pi * freq * x_theta)
                sinusoid_imag = torch.sin(2 * math.pi * freq * x_theta)

                # Gabor滤波器
                gabor_real = gaussian_envelope * sinusoid_real
                gabor_imag = gaussian_envelope * sinusoid_imag

                # 零均值化
                gabor_real = gabor_real - torch.mean(gabor_real)
                gabor_imag = gabor_imag - torch.mean(gabor_imag)

                # 归一化
                norm_real = torch.sum(torch.abs(gabor_real))
                norm_imag = torch.sum(torch.abs(gabor_imag))

                if norm_real > 0:
                    gabor_real = gabor_real / norm_real
                if norm_imag > 0:
                    gabor_imag = gabor_imag / norm_imag

                kernels_real.append(gabor_real)
                kernels_imag.append(gabor_imag)

        return torch.stack(kernels_real), torch.stack(kernels_imag)

    def apply_gabor_filtering(self, image):
        """
        应用Gabor滤波
        Args:
            image: 输入图像 [B, C, H, W]
        Returns:
            gabor_responses: Gabor响应 [B, num_filters*2, H, W]
        """
        B, C, H, W = image.shape
        num_filters = len(self.frequencies) * self.orientations

        # 转换为灰度图像（如果是多通道）
        if C > 1:
            gray_image = torch.mean(image, dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            gray_image = image

        # 存储所有Gabor响应
        gabor_responses = []

        for b in range(B):
            batch_responses = []

            for filter_idx in range(num_filters):
                # 获取当前滤波器
                kernel_real = self.gabor_kernels_real[filter_idx]
                kernel_imag = self.gabor_kernels_imag[filter_idx]

                # 应用滤波（使用卷积）
                image_np = gray_image[b, 0].cpu().numpy()
                kernel_real_np = kernel_real.cpu().numpy()
                kernel_imag_np = kernel_imag.cpu().numpy()

                # 卷积操作
                response_real = self._convolve2d_custom(image_np, kernel_real_np)
                response_imag = self._convolve2d_custom(image_np, kernel_imag_np)

                # 计算幅值和相位
                magnitude = np.sqrt(response_real ** 2 + response_imag ** 2)
                phase = np.arctan2(response_imag, response_real)

                # 转换回tensor
                magnitude_tensor = torch.from_numpy(magnitude).float().to(image.device)
                phase_tensor = torch.from_numpy(phase).float().to(image.device)

                batch_responses.append(magnitude_tensor)
                batch_responses.append(phase_tensor)

            gabor_responses.append(torch.stack(batch_responses))

        return torch.stack(gabor_responses)

    def _convolve2d_custom(self, image, kernel):
        """自定义2D卷积实现"""
        # 使用numpy的卷积实现卷积操作
        return convolve2d(image, kernel, mode='same', boundary='symm')


class TrueWaveletTransform(nn.Module):
    """小波变换实现"""

    def __init__(self, wavelet_type='db4', levels=3):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.levels = levels

        # 定义小波基函数
        self.wavelet_filters = self._get_wavelet_filters()

    def _get_wavelet_filters(self):
        """获取小波滤波器系数"""
        if self.wavelet_type == 'haar':
            # Haar小波
            low_pass = torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=torch.float32)
            high_pass = torch.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)], dtype=torch.float32)
        elif self.wavelet_type == 'db4':
            # Daubechies 4小波
            sqrt3 = math.sqrt(3)
            sqrt2 = math.sqrt(2)
            low_pass = torch.tensor([
                (1 + sqrt3) / (4 * sqrt2),
                (3 + sqrt3) / (4 * sqrt2),
                (3 - sqrt3) / (4 * sqrt2),
                (1 - sqrt3) / (4 * sqrt2)
            ], dtype=torch.float32)
            high_pass = torch.tensor([
                (1 - sqrt3) / (4 * sqrt2),
                -(3 - sqrt3) / (4 * sqrt2),
                (3 + sqrt3) / (4 * sqrt2),
                -(1 + sqrt3) / (4 * sqrt2)
            ], dtype=torch.float32)
        elif self.wavelet_type == 'bior2.2':
            # Biorthogonal 2.2小波
            low_pass = torch.tensor([0, -0.1767766953, 0.3535533906, 1.0606601718,
                                     0.3535533906, -0.1767766953, 0], dtype=torch.float32)
            high_pass = torch.tensor([0, 0.3535533906, -0.7071067812, 0.3535533906, 0], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        return {'low_pass': low_pass, 'high_pass': high_pass}

    def dwt_1d(self, signal, low_filter, high_filter):
        """
        一维离散小波变换
        Args:
            signal: 输入信号 [N]
            low_filter: 低通滤波器
            high_filter: 高通滤波器
        """
        # 卷积
        low_conv = F.conv1d(
            signal.unsqueeze(0).unsqueeze(0),
            low_filter.flip(0).unsqueeze(0).unsqueeze(0),
            padding=len(low_filter) // 2
        ).squeeze()

        high_conv = F.conv1d(
            signal.unsqueeze(0).unsqueeze(0),
            high_filter.flip(0).unsqueeze(0).unsqueeze(0),
            padding=len(high_filter) // 2
        ).squeeze()

        # 下采样
        low_freq = low_conv[::2]
        high_freq = high_conv[::2]

        return low_freq, high_freq

    def dwt_2d(self, image):
        """
        二维离散小波变换
        Args:
            image: 输入图像 [H, W]
        """
        H, W = image.shape
        low_filter = self.wavelet_filters['low_pass']
        high_filter = self.wavelet_filters['high_pass']

        # 行变换
        row_transformed = torch.zeros(H, W // 2, 2, device=image.device)
        for i in range(H):
            row = image[i, :]
            low, high = self.dwt_1d(row, low_filter, high_filter)
            row_transformed[i, :, 0] = low
            row_transformed[i, :, 1] = high

        # 列变换
        LL = torch.zeros(H // 2, W // 2, device=image.device)
        LH = torch.zeros(H // 2, W // 2, device=image.device)
        HL = torch.zeros(H // 2, W // 2, device=image.device)
        HH = torch.zeros(H // 2, W // 2, device=image.device)

        for j in range(W // 2):
            # LL和LH
            low_col = row_transformed[:, j, 0]
            LL_col, HL_col = self.dwt_1d(low_col, low_filter, high_filter)
            LL[:, j] = LL_col
            HL[:, j] = HL_col

            # HL和HH
            high_col = row_transformed[:, j, 1]
            LH_col, HH_col = self.dwt_1d(high_col, low_filter, high_filter)
            LH[:, j] = LH_col
            HH[:, j] = HH_col

        return LL, LH, HL, HH

    def idwt_1d(self, low_freq, high_freq, low_filter, high_filter):
        """一维小波逆变换"""
        # 上采样
        upsampled_low = torch.zeros(len(low_freq) * 2, device=low_freq.device)
        upsampled_high = torch.zeros(len(high_freq) * 2, device=high_freq.device)

        upsampled_low[::2] = low_freq
        upsampled_high[::2] = high_freq

        # 卷积
        low_conv = F.conv1d(
            upsampled_low.unsqueeze(0).unsqueeze(0),
            low_filter.unsqueeze(0).unsqueeze(0),
            padding=len(low_filter) // 2
        ).squeeze()

        high_conv = F.conv1d(
            upsampled_high.unsqueeze(0).unsqueeze(0),
            high_filter.unsqueeze(0).unsqueeze(0),
            padding=len(high_filter) // 2
        ).squeeze()

        return low_conv + high_conv

    def idwt_2d(self, LL, LH, HL, HH):
        """二维小波逆变换"""
        H, W = LL.shape
        low_filter = self.wavelet_filters['low_pass']
        high_filter = self.wavelet_filters['high_pass']

        # 列逆变换
        row_reconstructed = torch.zeros(H * 2, W, 2, device=LL.device)

        for j in range(W):
            # 重构低频部分
            low_col = self.idwt_1d(LL[:, j], HL[:, j], low_filter, high_filter)
            row_reconstructed[:, j, 0] = low_col

            # 重构高频部分
            high_col = self.idwt_1d(LH[:, j], HH[:, j], low_filter, high_filter)
            row_reconstructed[:, j, 1] = high_col

        # 行逆变换
        reconstructed = torch.zeros(H * 2, W * 2, device=LL.device)

        for i in range(H * 2):
            low_row = row_reconstructed[i, :, 0]
            high_row = row_reconstructed[i, :, 1]
            reconstructed_row = self.idwt_1d(low_row, high_row, low_filter, high_filter)
            reconstructed[i, :] = reconstructed_row

        return reconstructed

    def multilevel_dwt(self, image):
        """多级小波变换"""
        coefficients = []
        current = image.clone()

        for level in range(self.levels):
            LL, LH, HL, HH = self.dwt_2d(current)
            coefficients.append((LH, HL, HH))
            current = LL

        coefficients.append(LL)  # 最终的低频系数
        return coefficients

    def multilevel_idwt(self, coefficients):
        """多级小波逆变换"""
        current = coefficients[-1]  # 最低频系数

        for level in range(self.levels - 1, -1, -1):
            LH, HL, HH = coefficients[level]
            current = self.idwt_2d(current, LH, HL, HH)

        return current


class TrueBilateralFilter(nn.Module):
    """双边滤波器实现"""

    def __init__(self, spatial_sigma=1.0, intensity_sigma=0.1, kernel_size=5):
        super().__init__()
        self.spatial_sigma = spatial_sigma
        self.intensity_sigma = intensity_sigma
        self.kernel_size = kernel_size

    def bilateral_filter(self, image):
        """
        双边滤波实现
        Args:
            image: 输入图像 [B, C, H, W]
        """
        B, C, H, W = image.shape
        filtered_images = torch.zeros_like(image)

        # 计算空间权重核
        spatial_kernel = self._compute_spatial_kernel()

        for b in range(B):
            for c in range(C):
                filtered_images[b, c] = self._bilateral_filter_single_channel(
                    image[b, c], spatial_kernel
                )

        return filtered_images

    def _compute_spatial_kernel(self):
        """计算空间权重核"""
        center = self.kernel_size // 2
        spatial_kernel = torch.zeros(self.kernel_size, self.kernel_size)

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                dx = i - center
                dy = j - center
                spatial_weight = math.exp(-(dx * dx + dy * dy) / (2 * self.spatial_sigma * self.spatial_sigma))
                spatial_kernel[i, j] = spatial_weight

        return spatial_kernel

    def _bilateral_filter_single_channel(self, image, spatial_kernel):
        """单通道双边滤波"""
        H, W = image.shape
        filtered = torch.zeros_like(image)
        center = self.kernel_size // 2

        # 边界填充
        padded_image = F.pad(image, (center, center, center, center), mode='reflect')

        for i in range(H):
            for j in range(W):
                # 中心像素值
                center_intensity = image[i, j]

                # 提取邻域
                neighborhood = padded_image[i:i + self.kernel_size, j:j + self.kernel_size]

                # 计算强度权重
                intensity_diff = neighborhood - center_intensity
                intensity_weights = torch.exp(
                    -(intensity_diff * intensity_diff) /
                    (2 * self.intensity_sigma * self.intensity_sigma)
                )

                # 组合权重
                combined_weights = spatial_kernel * intensity_weights

                # 归一化权重
                weight_sum = torch.sum(combined_weights)
                if weight_sum > 0:
                    combined_weights = combined_weights / weight_sum

                # 计算滤波结果
                filtered[i, j] = torch.sum(combined_weights * neighborhood)

        return filtered


class TrueLocalBinaryPattern(nn.Module):
    """局部二值模式(LBP)实现"""

    def __init__(self, radius=1, n_points=8, method='uniform'):
        super().__init__()
        self.radius = radius
        self.n_points = n_points
        self.method = method

        # 预计算采样点坐标
        self.sample_points = self._compute_sample_points()

    def _compute_sample_points(self):
        """计算LBP采样点坐标"""
        angles = 2 * math.pi * torch.arange(self.n_points) / self.n_points
        sample_points = []

        for angle in angles:
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            sample_points.append((x, y))

        return sample_points

    def compute_lbp(self, image):
        """
        计算局部二值模式
        Args:
            image: 输入图像 [B, C, H, W]
        """
        B, C, H, W = image.shape
        lbp_images = torch.zeros(B, C, H, W, device=image.device)

        for b in range(B):
            for c in range(C):
                lbp_images[b, c] = self._compute_lbp_single_channel(image[b, c])

        return lbp_images

    def _compute_lbp_single_channel(self, image):
        """单通道LBP计算"""
        H, W = image.shape
        lbp_result = torch.zeros_like(image)

        for i in range(self.radius, H - self.radius):
            for j in range(self.radius, W - self.radius):
                center_pixel = image[i, j]
                lbp_code = 0

                for k, (dx, dy) in enumerate(self.sample_points):
                    # 计算采样点坐标
                    sample_x = i + dy  # 注意坐标系转换
                    sample_y = j + dx

                    # 双线性插值获取采样点像素值
                    sample_value = self._bilinear_interpolation(image, sample_x, sample_y)

                    # 比较并设置二进制位
                    if sample_value >= center_pixel:
                        lbp_code |= (1 << k)

                lbp_result[i, j] = lbp_code

        return lbp_result

    def _bilinear_interpolation(self, image, x, y):
        """双线性插值"""
        H, W = image.shape

        # 边界检查
        x = max(0, min(H - 1, x))
        y = max(0, min(W - 1, y))

        # 获取四个最近邻点
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, H - 1), min(y1 + 1, W - 1)

        # 计算权重
        wx = x - x1
        wy = y - y1

        # 双线性插值
        interpolated_value = (
                image[x1, y1] * (1 - wx) * (1 - wy) +
                image[x2, y1] * wx * (1 - wy) +
                image[x1, y2] * (1 - wx) * wy +
                image[x2, y2] * wx * wy
        )

        return interpolated_value

    def compute_lbp_histogram(self, lbp_image, bins=256):
        """计算LBP直方图"""
        # 展平图像
        flattened = lbp_image.flatten()

        # 计算直方图
        histogram = torch.histc(flattened, bins=bins, min=0, max=bins - 1)

        # 归一化
        histogram = histogram / torch.sum(histogram)

        return histogram


class TrueCannyEdgeDetector(nn.Module):
    """Canny边缘检测实现"""

    def __init__(self, low_threshold=0.1, high_threshold=0.2, sigma=1.0):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.sigma = sigma

        # Sobel算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    def canny_edge_detection(self, image):
        """
        Canny边缘检测完整流程
        Args:
            image: 输入图像 [B, C, H, W]
        """
        B, C, H, W = image.shape
        edge_maps = torch.zeros(B, 1, H, W, device=image.device)

        for b in range(B):
            # 转换为灰度图
            if C > 1:
                gray_image = torch.mean(image[b], dim=0)
            else:
                gray_image = image[b, 0]

            # Canny检测步骤
            edges = self._canny_single_image(gray_image)
            edge_maps[b, 0] = edges

        return edge_maps

    def _canny_single_image(self, image):
        """单图像Canny边缘检测"""
        # 1. 高斯平滑
        smoothed = self._gaussian_blur(image, self.sigma)

        # 2. 计算梯度
        grad_x, grad_y, magnitude, direction = self._compute_gradients(smoothed)

        # 3. 非极大值抑制
        suppressed = self._non_maximum_suppression(magnitude, direction)

        # 4. 双阈值检测
        strong_edges, weak_edges = self._double_threshold(suppressed)

        # 5. 边缘连接
        final_edges = self._edge_tracking_by_hysteresis(strong_edges, weak_edges)

        return final_edges

    def _gaussian_blur(self, image, sigma):
        """高斯模糊"""
        # 计算高斯核大小
        kernel_size = int(2 * math.ceil(2 * sigma) + 1)

        # 生成高斯核
        gaussian_kernel = self._generate_gaussian_kernel(kernel_size, sigma)

        # 应用高斯滤波
        H, W = image.shape
        padded_image = F.pad(image, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                             mode='reflect')

        blurred = torch.zeros_like(image)
        for i in range(H):
            for j in range(W):
                neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
                blurred[i, j] = torch.sum(neighborhood * gaussian_kernel)

        return blurred

    def _generate_gaussian_kernel(self, size, sigma):
        """生成高斯核"""
        kernel = torch.zeros(size, size)
        center = size // 2

        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))

        # 归一化
        kernel = kernel / torch.sum(kernel)
        return kernel

    def _compute_gradients(self, image):
        """计算图像梯度"""
        H, W = image.shape

        # 边界填充
        padded = F.pad(image, (1, 1, 1, 1), mode='reflect')

        grad_x = torch.zeros_like(image)
        grad_y = torch.zeros_like(image)

        # 应用Sobel算子
        for i in range(H):
            for j in range(W):
                neighborhood = padded[i:i + 3, j:j + 3]
                grad_x[i, j] = torch.sum(neighborhood * self.sobel_x.to(image.device))
                grad_y[i, j] = torch.sum(neighborhood * self.sobel_y.to(image.device))

        # 计算梯度幅值和方向
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        direction = torch.atan2(grad_y, grad_x)

        return grad_x, grad_y, magnitude, direction

    def _non_maximum_suppression(self, magnitude, direction):
        """非极大值抑制"""
        H, W = magnitude.shape
        suppressed = torch.zeros_like(magnitude)

        # 将角度量化到4个方向
        angle = direction * 180.0 / math.pi
        angle[angle < 0] += 180

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                current_angle = angle[i, j]
                current_magnitude = magnitude[i, j]

                # 确定梯度方向的两个邻居
                if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                    # 水平方向
                    neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
                elif (22.5 <= current_angle < 67.5):
                    # 对角线方向
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
                elif (67.5 <= current_angle < 112.5):
                    # 垂直方向
                    neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
                else:  # (112.5 <= current_angle < 157.5)
                    # 反对角线方向
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

                # 非极大值抑制
                if current_magnitude >= neighbors[0] and current_magnitude >= neighbors[1]:
                    suppressed[i, j] = current_magnitude

        return suppressed

    def _double_threshold(self, magnitude):
        """双阈值检测"""
        # 计算绝对阈值
        max_magnitude = torch.max(magnitude)
        low_thresh = self.low_threshold * max_magnitude
        high_thresh = self.high_threshold * max_magnitude

        # 分类像素
        strong_edges = (magnitude >= high_thresh).float()
        weak_edges = ((magnitude >= low_thresh) & (magnitude < high_thresh)).float()

        return strong_edges, weak_edges

    def _edge_tracking_by_hysteresis(self, strong_edges, weak_edges):
        """滞后边缘跟踪"""
        H, W = strong_edges.shape
        final_edges = strong_edges.clone()

        # 8连通性方向
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # 迭代连接弱边缘
        changed = True
        while changed:
            changed = False
            new_edges = final_edges.clone()

            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if weak_edges[i, j] == 1 and final_edges[i, j] == 0:
                        # 检查8邻域是否有强边缘
                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            if final_edges[ni, nj] == 1:
                                new_edges[i, j] = 1
                                changed = True
                                break

            final_edges = new_edges

        return final_edges


class TrueStateSpaceModel(nn.Module):
    """状态空间模型(Mamba)实现"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 学习的参数
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))

        # 初始化
        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化SSM参数"""
        # A矩阵初始化为负数（稳定性）
        with torch.no_grad():
            self.A.data = -torch.exp(torch.randn_like(self.A))
            nn.init.normal_(self.B, std=0.1)
            nn.init.normal_(self.C, std=0.1)
            nn.init.normal_(self.D, std=0.1)

    def discretize_ssm(self, delta):
        """
        离散化连续时间SSM
        Args:
            delta: 时间步长 [B, L, d_inner]
        """
        # 使用零阶保持器(ZOH)离散化
        # A_discrete = exp(delta * A)
        # B_discrete = (A_discrete - I) * A^{-1} * B

        B, L, D = delta.shape

        # 计算离散化的A矩阵
        delta_A = delta.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0)  # [B, L, d_inner, d_state]
        A_discrete = torch.matrix_exp(delta_A)

        # 计算离散化的B矩阵
        A_inv = torch.inverse(self.A + 1e-8 * torch.eye(self.d_state, device=self.A.device))
        temp = A_discrete - torch.eye(self.d_state, device=self.A.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        B_discrete = torch.matmul(temp, torch.matmul(A_inv.unsqueeze(0).unsqueeze(0),
                                                     self.B.unsqueeze(0).unsqueeze(0).unsqueeze(-1))).squeeze(-1)

        return A_discrete, B_discrete

    def selective_scan(self, u, delta, B, C):
        """
        选择性扫描算法 - SSM的核心计算
        Args:
            u: 输入序列 [B, L, d_inner]
            delta: 时间步长 [B, L, d_inner]
            B: 输入矩阵 [B, L, d_inner, d_state]
            C: 输出矩阵 [B, L, d_inner, d_state]
        """
        B_seq, L, D = u.shape

        # 离散化SSM
        A_discrete, B_discrete = self.discretize_ssm(delta)

        # 初始化状态
        x = torch.zeros(B_seq, D, self.d_state, device=u.device, dtype=u.dtype)
        outputs = []

        # 递归计算
        for i in range(L):
            # 状态更新: x_k = A_k * x_{k-1} + B_k * u_k
            x = torch.bmm(
                A_discrete[:, i].transpose(-2, -1),  # [B, d_state, d_inner]
                x.transpose(-2, -1)  # [B, d_state, d_inner]
            ).transpose(-2, -1) + B_discrete[:, i] * u[:, i].unsqueeze(-1)

            # 输出计算: y_k = C_k * x_k + D * u_k
            y = torch.sum(C[:, i].unsqueeze(-1) * x, dim=-1) + self.D * u[:, i]
            outputs.append(y)

        return torch.stack(outputs, dim=1)  # [B, L, d_inner]

    def forward(self, x):
        """
        SSM前向传播
        Args:
            x: 输入 [B, L, d_model]
        """
        B, L, D = x.shape

        # 投影到内部维度
        x_inner = F.linear(x, torch.randn(self.d_inner, D, device=x.device))  # [B, L, d_inner]

        # 学习时间步长
        delta = F.softplus(F.linear(x_inner, torch.randn(self.d_inner, self.d_inner, device=x.device))) + 1e-3

        # 学习输入和输出矩阵
        B_proj = F.linear(x_inner, torch.randn(self.d_state, self.d_inner, device=x.device).T).unsqueeze(
            -1)  # [B, L, d_inner, d_state]
        C_proj = F.linear(x_inner, torch.randn(self.d_state, self.d_inner, device=x.device).T).unsqueeze(
            -1)  # [B, L, d_inner, d_state]

        B_expanded = B_proj.expand(B, L, self.d_inner, self.d_state)
        C_expanded = C_proj.expand(B, L, self.d_inner, self.d_state)

        # 选择性扫描
        y = self.selective_scan(x_inner, delta, B_expanded, C_expanded)

        # 投影回原始维度
        output = F.linear(y, torch.randn(D, self.d_inner, device=x.device))

        return output


class TrueMutualInformationEstimator(nn.Module):
    """MINE互信息估计实现"""

    def __init__(self, x_dim, y_dim, hidden_dim=512):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim

        # 统计网络T(x,y)
        self.statistics_network = self._build_statistics_network()

        # 移动平均估计器
        self.register_buffer('ma_et', torch.tensor(1.0))
        self.ma_rate = 0.01

    def _build_statistics_network(self):
        """构建统计网络"""
        return nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def mine_objective(self, x, y):
        """
        MINE目标函数
        MI(X,Y) = sup_θ E_{P(x,y)}[T_θ(x,y)] - log(E_{P(x)P(y)}[exp(T_θ(x,y))])
        """
        batch_size = x.shape[0]

        # 正样本：来自联合分布P(x,y)
        joint_samples = torch.cat([x, y], dim=1)
        joint_scores = self.statistics_network(joint_samples)

        # 负样本：来自边际分布P(x)P(y)
        # 通过随机排列y来打破x和y的依赖关系
        y_shuffled = y[torch.randperm(batch_size)]
        marginal_samples = torch.cat([x, y_shuffled], dim=1)
        marginal_scores = self.statistics_network(marginal_samples)

        # 计算MINE估计
        joint_term = torch.mean(joint_scores)

        # 使用移动平均来稳定训练
        marginal_exp = torch.exp(marginal_scores)
        marginal_term = torch.mean(marginal_exp)

        # 更新移动平均
        if self.training:
            with torch.no_grad():
                self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * marginal_term

        # 使用移动平均的对数
        marginal_log_term = torch.log(self.ma_et)

        # MINE估计
        mi_estimate = joint_term - marginal_log_term

        return mi_estimate, joint_term, marginal_log_term

    def forward(self, x, y):
        """
        计算互信息估计
        Args:
            x: 第一个变量 [B, x_dim]
            y: 第二个变量 [B, y_dim]
        """
        # 确保输入维度正确
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        if y.dim() > 2:
            y = y.view(y.shape[0], -1)

        # 计算MINE目标
        mi_estimate, joint_term, marginal_term = self.mine_objective(x, y)

        return mi_estimate


class TrueGraphNeuralNetwork(nn.Module):
    """图神经网络实现"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # 图卷积层
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(GraphConvolutionLayer(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.gnn_layers.append(GraphConvolutionLayer(hidden_dim, output_dim))
            else:
                self.gnn_layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim))

    def forward(self, node_features, adjacency_matrix):
        """
        图神经网络前向传播
        Args:
            node_features: 节点特征 [N, input_dim]
            adjacency_matrix: 邻接矩阵 [N, N]
        """
        x = node_features

        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, adjacency_matrix)
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)

        return x


class GraphConvolutionLayer(nn.Module):
    """图卷积层的真正实现"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 权重矩阵
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        # 初始化
        self._initialize_parameters()

    def _initialize_parameters(self):
        """参数初始化"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, node_features, adjacency_matrix):
        """
        图卷积前向传播
        H^{(l+1)} = σ(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})
        """
        # 添加自连接
        adj_with_self_loops = adjacency_matrix + torch.eye(adjacency_matrix.size(0), device=adjacency_matrix.device)

        # 计算度矩阵
        degree_matrix = torch.diag(torch.sum(adj_with_self_loops, dim=1))

        # 计算D^{-1/2}
        degree_inv_sqrt = torch.diag(torch.pow(torch.diag(degree_matrix) + 1e-8, -0.5))

        # 归一化邻接矩阵
        normalized_adj = torch.mm(torch.mm(degree_inv_sqrt, adj_with_self_loops), degree_inv_sqrt)

        # 图卷积计算
        support = torch.mm(node_features, self.weight)
        output = torch.mm(normalized_adj, support) + self.bias

        return output


class TrueTemperatureGradientComputer(nn.Module):
    """温度梯度计算实现"""

    def __init__(self):
        super().__init__()

        # 温度梯度算子（基于物理学的热传导方程）
        self.gradient_operators = self._create_gradient_operators()

    def _create_gradient_operators(self):
        """创建温度梯度算子"""
        # Sobel算子的改进版本，专门用于温度场
        # 考虑热传导的各向异性
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # Scharr算子（更精确的梯度估计）
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)

        # Prewitt算子
        prewitt_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
        prewitt_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)

        return {
            'sobel_x': sobel_x, 'sobel_y': sobel_y,
            'scharr_x': scharr_x, 'scharr_y': scharr_y,
            'prewitt_x': prewitt_x, 'prewitt_y': prewitt_y
        }

    def compute_temperature_gradient(self, temperature_field):
        """
        计算温度场的梯度
        Args:
            temperature_field: 温度场 [B, C, H, W]
        Returns:
            gradient_magnitude: 梯度幅值
            gradient_direction: 梯度方向
            gradient_components: 梯度分量
        """
        B, C, H, W = temperature_field.shape

        # 初始化结果
        gradient_magnitudes = torch.zeros_like(temperature_field)
        gradient_directions = torch.zeros_like(temperature_field)
        gradient_x_components = torch.zeros_like(temperature_field)
        gradient_y_components = torch.zeros_like(temperature_field)

        for b in range(B):
            for c in range(C):
                temp_field = temperature_field[b, c]

                # 使用多种算子计算梯度
                gradients = self._multi_operator_gradient(temp_field)

                # 融合多个梯度估计
                grad_x = self._fuse_gradients([
                    gradients['sobel_x'],
                    gradients['scharr_x'],
                    gradients['prewitt_x']
                ])

                grad_y = self._fuse_gradients([
                    gradients['sobel_y'],
                    gradients['scharr_y'],
                    gradients['prewitt_y']
                ])

                # 计算梯度幅值和方向
                magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
                direction = torch.atan2(grad_y, grad_x)

                gradient_magnitudes[b, c] = magnitude
                gradient_directions[b, c] = direction
                gradient_x_components[b, c] = grad_x
                gradient_y_components[b, c] = grad_y

        return {
            'magnitude': gradient_magnitudes,
            'direction': gradient_directions,
            'grad_x': gradient_x_components,
            'grad_y': gradient_y_components
        }

    def _multi_operator_gradient(self, image):
        """使用多种算子计算梯度"""
        gradients = {}

        for operator_name, kernel in self.gradient_operators.items():
            gradient = self._apply_convolution(image, kernel.to(image.device))
            gradients[operator_name] = gradient

        return gradients

    def _apply_convolution(self, image, kernel):
        """应用卷积操作"""
        H, W = image.shape
        kernel_h, kernel_w = kernel.shape
        pad_h, pad_w = kernel_h // 2, kernel_w // 2

        # 边界填充
        padded_image = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

        # 卷积计算
        result = torch.zeros_like(image)
        for i in range(H):
            for j in range(W):
                neighborhood = padded_image[i:i + kernel_h, j:j + kernel_w]
                result[i, j] = torch.sum(neighborhood * kernel)

        return result

    def _fuse_gradients(self, gradients):
        """融合多个梯度估计"""
        # 使用加权平均融合
        weights = [0.4, 0.4, 0.2]  # Sobel, Scharr, Prewitt的权重

        fused_gradient = torch.zeros_like(gradients[0])
        for gradient, weight in zip(gradients, weights):
            fused_gradient += weight * gradient

        return fused_gradient

    def compute_heat_flux(self, temperature_field, thermal_conductivity=1.0):
        """
        计算热流密度（基于傅里叶定律）
        q = -k * ∇T
        """
        gradient_info = self.compute_temperature_gradient(temperature_field)

        # 热流密度
        heat_flux_x = -thermal_conductivity * gradient_info['grad_x']
        heat_flux_y = -thermal_conductivity * gradient_info['grad_y']
        heat_flux_magnitude = torch.sqrt(heat_flux_x ** 2 + heat_flux_y ** 2)

        return {
            'heat_flux_x': heat_flux_x,
            'heat_flux_y': heat_flux_y,
            'heat_flux_magnitude': heat_flux_magnitude
        }

    def compute_laplacian(self, temperature_field):
        """
        计算温度场的拉普拉斯算子（用于热扩散分析）
        ∇²T = ∂²T/∂x² + ∂²T/∂y²
        """
        # 拉普拉斯算子
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32)

        B, C, H, W = temperature_field.shape
        laplacian_result = torch.zeros_like(temperature_field)

        for b in range(B):
            for c in range(C):
                temp_field = temperature_field[b, c]
                laplacian_result[b, c] = self._apply_convolution(
                    temp_field, laplacian_kernel.to(temp_field.device)
                )

        return laplacian_result


class RADIANT(TwoBranchesFusionNet):
    """RADIANT fusion network described in the ACM MM 2026 paper.

    GIFNet supplies the shared CNN encoder, Swin interaction branch, and image
    decoder. The recovered path adds matrix-exponential non-interfering
    decomposition, task memory, and multi-granularity interaction. Experimental
    classes above are retained for research compatibility but are intentionally
    not part of this production forward pass.
    """

    def __init__(self, s, n, channel, stride, num_tasks=6, stream_dim=32):
        super().__init__(s, n, channel, stride)
        self.feature_dim = 32 + 32 + 32 + channel * 2
        self.stream_dim = stream_dim
        self.num_tasks = num_tasks

        self.vision_mamba_encoder = VisionMambaEncoder(
            dim=self.feature_dim, depth=1, patch_size=4
        )
        self.mamba_scale = nn.Parameter(torch.tensor(0.1))
        self.decomposition = NonInterferingDecomposition(
            input_dim=self.feature_dim, stream_dim=stream_dim
        )
        self.multi_grain_interaction = MultiGranularityInteraction(stream_dim)
        self.task_memory = TaskMemory(num_tasks=num_tasks, channels=stream_dim)

        self.unique_gate = nn.Sequential(
            nn.Conv2d(stream_dim * 3, stream_dim, 1),
            nn.GELU(),
            nn.Conv2d(stream_dim, 1, 1),
            nn.Sigmoid(),
        )
        self.swin_projection = nn.Conv2d(self.feature_dim, stream_dim, 1)
        self.allocation_projection = nn.Sequential(
            nn.Conv2d(stream_dim * 4, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, self.feature_dim, 1),
        )

        # Unified objective weights from Eq. (19) and the memory repulsion term.
        self.alpha_align = 0.5
        self.beta_orthogonal = 0.1
        self.gamma_interaction = 1.0
        self.lambda_memory_repulsion = 0.01

    def _refine_joint_features(self, x, y):
        basic_features = super().forward_encoder(x, y)
        mamba_features = self.vision_mamba_encoder(basic_features)
        refined_features = basic_features + torch.tanh(self.mamba_scale) * mamba_features
        return basic_features, mamba_features, refined_features

    def forward_enhanced_encoder(self, x, y, task_id=0):
        basic_features, mamba_features, refined_features = self._refine_joint_features(x, y)
        shared, unique_1, unique_2, orthogonal_losses = self.decomposition(refined_features)
        return {
            'basic_features': basic_features,
            'mamba_features': mamba_features,
            'fused_features': refined_features,
            'shared_features': shared,
            'unique_1_features': unique_1,
            'unique_2_features': unique_2,
            # Backwards-compatible names used by earlier recovery scripts.
            'content_features': shared,
            'modality_features': unique_1,
            'task_features': unique_2,
            'orthogonal_losses': orthogonal_losses,
        }

    def forward_with_radiant(
        self,
        x,
        y,
        x_mfif=None,
        y_mfif=None,
        training_tag=1,
        task_id=None,
        update_memory=None,
    ):
        """Run the complete allocation-aware fusion path.

        ``training_tag`` keeps the original GIFNet convention (1=IVIF,
        2=MFIF). ``task_id`` addresses the six task memories and is zero-based.
        The optional peer pair is used by GIFNet's cross-task Swin branch.
        """

        if task_id is None:
            task_id = training_tag - 1
        if update_memory is None:
            update_memory = self.training

        current = self.forward_enhanced_encoder(x, y, task_id=task_id)
        if x_mfif is not None and y_mfif is not None:
            _, _, peer_features = self._refine_joint_features(x_mfif, y_mfif)
        else:
            peer_features = current['fused_features']

        if training_tag == 1:
            swin_features = self.forward_MultiTask_branch(
                fea_com_ivif=current['fused_features'],
                fea_com_mfif=peer_features,
                trainingTag=1,
            )
        else:
            swin_features = self.forward_MultiTask_branch(
                fea_com_ivif=peer_features,
                fea_com_mfif=current['fused_features'],
                trainingTag=2,
            )

        interaction = self.multi_grain_interaction(
            current['unique_1_features'], current['unique_2_features']
        )
        memory = self.task_memory(
            interaction['synergy'], task_id=task_id, update_memory=update_memory
        )

        gate = self.unique_gate(torch.cat([
            current['shared_features'],
            current['unique_1_features'],
            current['unique_2_features'],
        ], dim=1))
        selected_unique = (
            gate * current['unique_1_features']
            + (1.0 - gate) * current['unique_2_features']
        )
        allocated_features = self.allocation_projection(torch.cat([
            current['shared_features'],
            selected_unique,
            memory['conditioned_synergy'],
            self.swin_projection(swin_features),
        ], dim=1))

        out_rec = self.forward_rec_decoder(current['fused_features']) / 2.0 + 0.5
        out_f = self.forward_mixed_decoder(
            current['fused_features'], allocated_features
        )

        orthogonal_total = sum(current['orthogonal_losses'].values())
        allocation_loss = (
            self.alpha_align * memory['alignment_loss']
            + self.beta_orthogonal * orthogonal_total
            + self.gamma_interaction * interaction['interaction_loss']
            + self.lambda_memory_repulsion * memory['repulsion_loss']
        )
        zero = out_f.new_zeros(())
        loss_dict = {
            'orthogonal_losses': current['orthogonal_losses'],
            'prototype_loss': memory['alignment_loss'],
            'clustering_loss': memory['repulsion_loss'],
            'contrastive_loss': interaction['interaction_loss'],
            'mi_loss': zero,
        }

        return {
            'out_rec': out_rec,
            'out_f': out_f,
            'allocation_loss': allocation_loss,
            'loss_dict': loss_dict,
            'encoder_results': current,
            'ivif_results': current,
            'mfif_results': {'fused_features': peer_features},
            'prototype_results': memory,
            'contrastive_info': interaction,
            'allocation_features': allocated_features,
        }

    def forward(self, x, y, task_id=0, training_tag=None, update_memory=None):
        if training_tag is None:
            training_tag = 1 if int(task_id) == 0 else 2
        return self.forward_with_radiant(
            x,
            y,
            training_tag=training_tag,
            task_id=task_id,
            update_memory=update_memory,
        )['out_f']


class RADIANTLoss(nn.Module):
    """Unified fusion and allocation objective from Eqs. (17)-(19)."""

    def __init__(
        self,
        lambda_orthogonal=0.1,
        lambda_prototype=0.5,
        lambda_contrastive=1.0,
        lambda_mi=0.0,
        intensity_weight=1.0,
        gradient_weight=10.0,
        reconstruction_weight=0.1,
    ):
        super().__init__()
        self.lambda_orthogonal = lambda_orthogonal
        self.lambda_prototype = lambda_prototype
        self.lambda_contrastive = lambda_contrastive
        self.lambda_mi = lambda_mi
        self.intensity_weight = intensity_weight
        self.gradient_weight = gradient_weight
        self.reconstruction_weight = reconstruction_weight
        self.gradient = SobelGradient()

    def _paper_fusion_loss(self, fused, source_1, source_2):
        intensity_target = torch.maximum(source_1, source_2)
        intensity_loss = F.l1_loss(fused, intensity_target)
        fused_gradient = self.gradient(fused)
        gradient_target = torch.maximum(
            self.gradient(source_1), self.gradient(source_2)
        )
        gradient_loss = F.l1_loss(fused_gradient, gradient_target)
        return intensity_loss, gradient_loss

    def forward(self, outputs, targets):
        losses = {}
        total_loss = outputs['out_f'].new_zeros(())

        if 'source1' in targets and 'source2' in targets:
            intensity_loss, gradient_loss = self._paper_fusion_loss(
                outputs['out_f'], targets['source1'], targets['source2']
            )
            losses['intensity'] = intensity_loss
            losses['gradient'] = gradient_loss
            fusion_loss = (
                self.intensity_weight * intensity_loss
                + self.gradient_weight * gradient_loss
            )
            losses['fusion'] = fusion_loss
            total_loss = total_loss + fusion_loss
        elif 'target_fusion' in targets:
            fusion_loss = F.l1_loss(outputs['out_f'], targets['target_fusion'])
            losses['fusion'] = fusion_loss
            total_loss = total_loss + fusion_loss

        if 'out_rec' in outputs and 'target_rec' in targets:
            reconstruction = F.l1_loss(outputs['out_rec'], targets['target_rec'])
            losses['reconstruction'] = reconstruction
            total_loss = total_loss + self.reconstruction_weight * reconstruction

        if 'allocation_loss' in outputs:
            allocation_loss = outputs['allocation_loss']
        else:
            loss_dict = outputs.get('loss_dict', {})
            orthogonal = sum(loss_dict.get('orthogonal_losses', {}).values())
            allocation_loss = (
                self.lambda_orthogonal * orthogonal
                + self.lambda_prototype * loss_dict.get('prototype_loss', total_loss.new_zeros(()))
                + self.lambda_contrastive
                * loss_dict.get('contrastive_loss', total_loss.new_zeros(()))
                + self.lambda_mi * loss_dict.get('mi_loss', total_loss.new_zeros(()))
            )
        losses['allocation'] = allocation_loss
        total_loss = total_loss + allocation_loss

        loss_dict = outputs.get('loss_dict', {})
        if 'orthogonal_losses' in loss_dict:
            losses['orthogonal'] = sum(loss_dict['orthogonal_losses'].values())
        if 'prototype_loss' in loss_dict:
            losses['alignment'] = loss_dict['prototype_loss']
        if 'contrastive_loss' in loss_dict:
            losses['interaction'] = loss_dict['contrastive_loss']
        if 'clustering_loss' in loss_dict:
            losses['memory_repulsion'] = loss_dict['clustering_loss']

        return total_loss, losses
