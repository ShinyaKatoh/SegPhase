import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary

class OverLapPatchMerging(nn.Module):
  def __init__(self, 
               in_channels, 
               emb_dim, 
               patch_size,
               stride,
               in_length):
    super().__init__()
    self.conv = nn.Conv1d(in_channels = in_channels,
                          out_channels = emb_dim,
                          kernel_size = patch_size,
                          stride = stride,
                          padding = patch_size//2,
                          bias=False)
    
    self.pos_emb = nn.Parameter(torch.randn(1, in_length//stride, emb_dim))
  
  def forward(self, x):
    x = self.conv(x)
    x = rearrange(x, 'B C L -> B L C')
    x = x + self.pos_emb
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
                 emb_dim,
                 kernel_size,
                 expantion_ratio):
        super().__init__()
        self.linear1 = nn.Conv1d(emb_dim, 
                                 emb_dim, 
                                 kernel_size = 1)
        
        self.linear2 = nn.Conv1d(emb_dim * expantion_ratio, 
                                 emb_dim, 
                                 kernel_size = 1)
        
        self.conv = nn.Conv1d(in_channels=emb_dim, 
                              out_channels=emb_dim * expantion_ratio, 
                              kernel_size=3, 
                              groups=emb_dim,
                              padding='same')
        
        self.gelu = nn.GELU()

    def forward(self,x):
        x = rearrange(x, 'B L C -> B C L')
        x = self.linear1(x)
        x = self.conv(x)
        x = self.gelu(x)
        x = self.linear2(x)
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
    
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size,
                              bias=False, 
                              padding='same')
        
        self.bn = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
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
        
        self.conv1 = ConvBNReLU(in_channels=ch1, out_channels=self.ch, kernel_size=self.kernel_size)
        self.conv2 = ConvBNReLU(in_channels=ch2, out_channels=self.ch, kernel_size=self.kernel_size)
        self.conv3 = ConvBNReLU(in_channels=ch3, out_channels=self.ch, kernel_size=self.kernel_size)
        
        self.conv4 = ConvBNReLU(in_channels=self.ch*3, out_channels=self.ch, kernel_size=self.kernel_size)
        
        self.conv5 = nn.Conv1d(in_channels=self.ch,
                               out_channels=class_num,
                               kernel_size=1,
                               padding='same')
        
        self.Up1 = nn.Upsample(scale_factor=st1*st2*st3, mode='linear', align_corners=True)
        self.Up2 = nn.Upsample(scale_factor=st2*st3, mode='linear', align_corners=True)
        self.Up3 = nn.Upsample(scale_factor=st3, mode='linear', align_corners=True)
        
        self.relu = nn.ReLU()
        
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
        
        
        self.ch1 = 16
        self.ch2 = 32
        self.ch3 = 64
        
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
