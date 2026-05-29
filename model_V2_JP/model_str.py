import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary

import math
import torch

class PositionalEncoding1D:
    """
    1次元の位置エンコーディング（入出力: B, L, C）
    - mask なし
    - 位置 index を 0..L-1 として 0〜2π に正規化
    - 偶数chに sin, 奇数chに cos
    """
    def __init__(self, eps: float = 1e-6, temperature: int = 10000):
        self.eps = eps
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)
        Returns:
            (B, L, C): 位置埋め込み
        """
        B, L, C = x.shape
        # チャンネルごとのスケーリング基底（偶奇ペア）
        dim_t = torch.arange(0, C, 2, dtype=x.dtype, device=x.device)   # [0,2,4,...]
        dim_t = dim_t.repeat_interleave(2)                               # [0,0,2,2,4,4,...]
        dim_t = self.temperature ** (dim_t / max(C, 1))                  # (C,)

        # 位置 index: 0..L-1 を 0..2π に線形正規化
        pos_idx = torch.arange(L, dtype=torch.float32, device=x.device)  # (L,)
        denom = (L - 1) if L > 1 else 1
        pos = 2 * math.pi * pos_idx / (denom + self.eps)                 # (L,)

        # (B, L, 1) / (1,1,C) -> (B, L, C)
        pos = pos.view(1, L, 1).expand(B, L, 1) / dim_t.view(1, 1, C)

        # 偶数→sin, 奇数→cos
        pos[..., ::2] = pos[..., ::2].sin()
        pos[..., 1::2] = pos[..., 1::2].cos()

        # dtype を元テンソルに合わせる
        if pos.dtype != x.dtype:
            pos = pos.to(dtype=x.dtype)
        return pos


import torch
import torch.nn as nn
from einops import rearrange

# 位置埋め込みは既存実装を想定
# class PositionalEncoding1D: ...

class OverLapPatchMerging(nn.Module):
    """
    x_in: (B, C_in, L_in)
      -> ReflectionPad1d
      -> Conv1d (padding=0)
      -> (B, emb_dim, L_out)
      -> 転置 (B, L_out, emb_dim)
      -> （任意）位置埋め込み加算
    """
    def __init__(self,
                 in_channels: int,
                 emb_dim: int,
                 patch_size: int,
                 stride: int,
                 use_pos: bool = True,
                 temperature: int = 10000,
                 eps: float = 1e-6):
        super().__init__()
        # 元の padding=patch_size//2 と等価な出力長にするため左右対称にパディング
        p = patch_size // 2
        self.pad = nn.ReflectionPad1d((p, p))

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,          # パディングは前段の ReflectionPad1d が担当
            bias=False
        )

        self.use_pos = use_pos
        if use_pos:
            self.pos_enc = PositionalEncoding1D(eps=eps, temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L_in)
        Returns:
            (B, L_out, emb_dim)
        """
        x = self.pad(x)                 # reflect padding
        x = self.conv(x)                # (B, emb_dim, L_out)
        x = rearrange(x, 'B C L -> B L C')

        if self.use_pos:
            pos = self.pos_enc.generate(x)  # (B, L_out, emb_dim)
            x = x + pos
        return x

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                  channels, 
                  reduction_ratio,
                  num_heads):
        super().__init__()
        
        self.rr = reduction_ratio
        
        dropout = 0
        
        if reduction_ratio > 1:
        
            self.reducer = nn.Conv1d(in_channels = channels, 
                                     out_channels = channels, 
                                     kernel_size=reduction_ratio*2-1, 
                                     stride=reduction_ratio,
                                     padding = (reduction_ratio*2-1)//2,
                                     padding_mode='reflect',
                                     bias=False)
            
            self.ln = nn.LayerNorm(channels)
        
        self.linear_q = nn.Linear(channels, channels, bias=False)
        self.linear_k = nn.Linear(channels, channels, bias=False)
        self.linear_v = nn.Linear(channels, channels, bias=False)
        
        self.ln_q = nn.LayerNorm(channels)
        self.ln_k = nn.LayerNorm(channels)
        self.ln_v = nn.LayerNorm(channels)
        
        self.head = num_heads
        self.head_ch = channels // num_heads
        self.sqrt_dh = self.head_ch**0.5 
        
        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Linear(channels, channels, bias=False)
        self.w_drop = nn.Dropout(dropout)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        
        if self.rr > 1:
            xr = rearrange(x, 'B L C -> B C L')
            
            reduced = self.reducer(xr)
            reduced = rearrange(reduced, 'B C L -> B L C')
            reduced = self.ln(reduced)
        
            q = self.linear_q(x)
            k = self.linear_k(reduced)
            v = self.linear_v(reduced)
            
        else:
            q = self.linear_q(x)
            k = self.linear_k(x)
            v = self.linear_v(x)
            
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)
            
        q = rearrange(q, 'B L (h C) -> B h L C', h=self.head)
        k = rearrange(k, 'B L (h C) -> B h L C', h=self.head)
        v = rearrange(v, 'B L (h C) -> B h L C', h=self.head)
        
        k_T = k.transpose(2, 3)
        
        dots = (q @ k_T) / self.sqrt_dh
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        out = attn @ v
        
        out = rearrange(out, 'B h L C -> B L (h C)')
        
        out = self.w_o(out) 
        out = self.w_drop(out)
        
        return out, attn
    
class MixFFN(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 kernel_size: int = 3,
                 expantion_ratio: int = 4):
        super().__init__()
        # reflectで"same"相当を狙うなら奇数カーネルを推奨
        assert kernel_size % 2 == 1, "reflect paddingで'same'相当を得るにはkernel_sizeは奇数にしてください"

        self.linear1 = nn.Conv1d(emb_dim, emb_dim, kernel_size=1, bias=False)

        # Depthwise conv（groups=emb_dim） + reflect padding
        self.conv = nn.Conv1d(
            in_channels=emb_dim,
            out_channels=emb_dim * expantion_ratio,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            padding_mode='reflect',   # ← ここでreflect指定
            groups=emb_dim,
            bias=False,
        )

        self.linear2 = nn.Conv1d(emb_dim * expantion_ratio, emb_dim, kernel_size=1, bias=False)

        self.gelu = nn.GELU()

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = rearrange(x, 'B L C -> B C L')
        x = self.linear1(x)
        x = self.conv(x)
        x = self.gelu(x)
        x = self.linear2(x)
        # 戻す: (B, C, L) -> (B, L, C)
        x = rearrange(x, 'B C L -> B L C')
        return x
        
class ViTEncoderMixFFN(nn.Module):
    def __init__(self,
                 emb_dim,
                 kernel_size,
                 reduction_ratio,
                 head_num,
                 expantion_ratio):
        
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(emb_dim, 
                                           reduction_ratio,
                                           head_num)
        
        self.ffn = MixFFN(emb_dim, 
                          kernel_size,
                          expantion_ratio)
        
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
       
    def forward(self, x):
        
        residual_mhsa = x
        mhsa_input = self.ln1(x)
        mhsa_output, attn = self.mhsa(mhsa_input)
        mhsa_output2 = mhsa_output + residual_mhsa
        
        residual_ffn = mhsa_output2
        ffn_input = self.ln2(mhsa_output2)
        ffn_output = self.ffn(ffn_input) + residual_ffn
        
        return ffn_output

class EncoderBlock(nn.Module):
    def __init__(self, 
                 emb_dim,
                 kernel_size,
                 reduction_ratio,
                 head_num,
                 expantion_ratio,
                 block_num):
        super().__init__()
       
        self.Encoder = nn.Sequential(*[ViTEncoderMixFFN(emb_dim,
                                                        kernel_size,
                                                        reduction_ratio,
                                                        head_num,
                                                        expantion_ratio)
                                       for _ in range(block_num)])
        
    def forward(self, x):
        x = self.Encoder(x)
        return x
    
class SegPhaseBlock(nn.Module):
    def __init__(self, 
                 in_length,
                 in_channels, 
                 emb_dim, 
                 patch_size,
                 stride, 
                 head_num, 
                 reduction_ratio,
                 expantion_ratio, 
                 block_num):
        super().__init__() 
        self.OLPM = OverLapPatchMerging(in_channels, 
                                        emb_dim, 
                                        patch_size, 
                                        stride,
                                        in_length)
        
        self.ENCB = EncoderBlock(emb_dim = emb_dim,
                                 kernel_size = patch_size,
                                 reduction_ratio = reduction_ratio,
                                 head_num = head_num,
                                 expantion_ratio = expantion_ratio,
                                 block_num = block_num)
        
    def forward(self,x):
        x = self.OLPM(x)
        x = self.ENCB(x)
        x = rearrange(x, 'B L C -> B C L')
        return x
    
class ConvLNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # SAME相当の左右パディング量（dilation=1想定）
        total = kernel_size - 1
        pad_l = total // 2
        pad_r = total - pad_l

        self.pad = nn.ReflectionPad1d((pad_l, pad_r))
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size,
                              bias=False, 
                              padding=0)  # パディングは前段で実施
        
        self.ln = nn.LayerNorm(out_channels)
        self.gelu = nn.GELU()
        
    @staticmethod
    def _ln_channel_last(x, ln):
        # x: (N, C, L) -> (N, L, C) にして LN(C) -> 戻す
        x = x.transpose(1, 2)            # (N, L, C)
        x = ln(x)                        # LN over C
        x = x.transpose(1, 2).contiguous()   # (N, C, L)
        return x
        
    def forward(self, x):
        x = self.pad(x)          # reflect padding
        x = self.conv(x)
        x = self._ln_channel_last(x, self.ln)
        x = self.gelu(x)
        return x
    
def gaussian_kernel1d(sigma=1.0, radius=None):
    if radius is None:
        radius = int(3*sigma)
    x = torch.arange(-radius, radius+1, dtype=torch.float32)
    k = torch.exp(-0.5*(x/sigma)**2)
    k /= k.sum()
    return k  # shape [K]

class AntiAliasedUpsample1D(nn.Module):
    def __init__(self, channels, scale=2, sigma=1.0):
        super().__init__()
        self.scale = scale
        k = gaussian_kernel1d(sigma)  # [K]
        self.register_buffer('kernel', k[None, None, :].repeat(channels, 1, 1))  # depthwise
        self.pad = (k.numel()//2, k.numel()//2)
        self.channels = channels

    def forward(self, x):
        # x: [B,C,T]
        x = F.interpolate(x, scale_factor=self.scale, mode='linear', align_corners=False)
        x = F.pad(x, self.pad, mode='reflect')
        x = F.conv1d(x, self.kernel, groups=self.channels)  # depthwise blur
        return x

class SegPhaseOutput(nn.Module):
    def __init__(self, 
                 ch1, ch2, ch3, 
                 st1, st2, st3,
                 ks,
                 class_num):
        super().__init__()
        
        self.kernel_size = ks
        self.ch = 64
        
        self.conv1 = ConvLNGELU(in_channels=ch1, out_channels=self.ch, kernel_size=self.kernel_size)
        self.conv2 = ConvLNGELU(in_channels=ch2, out_channels=self.ch, kernel_size=self.kernel_size)
        self.conv3 = ConvLNGELU(in_channels=ch3, out_channels=self.ch, kernel_size=self.kernel_size)
        
        self.conv4 = ConvLNGELU(in_channels=self.ch*3, out_channels=self.ch, kernel_size=self.kernel_size)
        
        self.conv5 = nn.Conv1d(in_channels=self.ch,
                               out_channels=class_num,
                               kernel_size=1,
                               padding='same')
        
        # self.conv5 = ConvLNGELU(in_channels=self.ch, out_channels=class_num, kernel_size=self.kernel_size)
        
        self.Up1 = nn.Upsample(scale_factor=st1*st2*st3, mode='linear', align_corners=True)
        self.Up2 = nn.Upsample(scale_factor=st2*st3, mode='linear', align_corners=True)
        self.Up3 = nn.Upsample(scale_factor=st3, mode='linear', align_corners=True)
        
        self.upsample1 = AntiAliasedUpsample1D(ch1, scale=st1*st2*st3)
        self.upsample2 = AntiAliasedUpsample1D(ch2, scale=st2*st3)
        self.upsample3 = AntiAliasedUpsample1D(ch3, scale=st3)
        
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x1, x2, x3):
        out1 = self.Up1(x1)
        out1 = self.conv1(out1)
       
        out2 = self.Up2(x2)
        out2 = self.conv2(out2)

        out3 = self.Up3(x3)
        out3 = self.conv3(out3)
       
        out = torch.concat([out1, out2, out3], dim = 1)
        
        out = self.conv4(out)
        
        out = self.conv5(out)
        out = self.softmax(out)
        return out
    
class Model(nn.Module):
    def __init__(self, 
                 in_length,
                 in_channels,
                 class_num,
                 strides,
                 kernel_size,
                 expantion_ratio: int = 4
                 ):
        super().__init__()
        
        
        self.ch1 = 32
        self.ch2 = 64
        self.ch3 = 128
        
        self.ks1 = strides[0]*2-1
        self.ks2 = strides[1]*2-1
        self.ks3 = strides[2]*2-1
        
        self.st1 = strides[0]
        self.st2 = strides[1]
        self.st3 = strides[2]
        
        self.hn1 = 2
        self.hn2 = 4
        self.hn3 = 8
        
        self.rr1 = 3
        self.rr2 = 2
        self.rr3 = 1
        
        self.bn1 = 3
        self.bn2 = 3
        self.bn3 = 3
      
        self.seg_block1 = SegPhaseBlock(in_length = in_length,
                                         in_channels = in_channels, 
                                         emb_dim = self.ch1, 
                                         patch_size = self.ks1, 
                                         stride=self.st1,
                                         head_num = self.hn1,
                                         reduction_ratio = self.rr1,
                                         expantion_ratio = expantion_ratio, 
                                         block_num = self.bn1)
        
        self.seg_block2 = SegPhaseBlock(in_length = in_length//self.st1,
                                         in_channels = self.ch1, 
                                         emb_dim = self.ch2, 
                                         patch_size = self.ks2, 
                                         stride = self.st2,
                                         head_num = self.hn2, 
                                         reduction_ratio = self.rr2,
                                         expantion_ratio = expantion_ratio, 
                                         block_num = self.bn2)
        
        self.seg_block3 = SegPhaseBlock(in_length = in_length//(self.st1*self.st2),
                                         in_channels = self.ch2, 
                                         emb_dim = self.ch3, 
                                         patch_size = self.ks3, 
                                         stride = self.st3,
                                         head_num = self.hn3, 
                                         reduction_ratio = self.rr3,
                                         expantion_ratio = expantion_ratio, 
                                         block_num = self.bn3)
        
        self.output = SegPhaseOutput(self.ch3, self.ch2, self.ch1, 
                                      self.st3, self.st2, self.st1,
                                      kernel_size,
                                      class_num)
        
    def forward(self, x):
        x1 = self.seg_block1(x)
        x2 = self.seg_block2(x1)
        x3 = self.seg_block3(x2)
        out = self.output(x3, x2, x1)
        return out
    
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    model = Model(in_length=100*30, in_channels=3, class_num=3, strides=[3,2,2], kernel_size=3).to('cpu')
    summary(model, input_size=(32, 3, 100*30))
    
    # model = Model(in_length=250*30, in_channels=3, class_num=3, strides=[5,3,2]).to('cpu')
    # summary(model, input_size=(32, 3, 250*30))
    
    # model = Model(in_length=100*30, in_channels=1, class_num=2, strides=[3,2,2]).to('cpu')
    # summary(model, input_size=(32, 1, 100*30))
