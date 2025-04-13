
# @title ResBlock
import torch
import torch.nn as nn

def zero_module(module):
    for p in module.parameters(): p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=3):
        super().__init__()
        out_ch = out_ch or in_ch
        act = nn.SiLU() #
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # self.block = nn.Sequential( # best?
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), act,
        #     zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1)), nn.BatchNorm2d(out_ch), act,
        #     )
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch), act, nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2),
            nn.BatchNorm2d(out_ch), act, zero_module(nn.Conv2d(out_ch, out_ch, kernel, padding=kernel//2)),
            )

    def forward(self, x): # [b,c,h,w]
        return self.block(x) + self.res_conv(x)


# @title UpDownBlock_me
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PixelShuffleConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=1, r=1):
        super().__init__()
        self.r = r
        r = max(r, int(1/r))
        out_ch = out_ch or in_ch
        if self.r>1: self.net = nn.Sequential(ResBlock(in_ch, out_ch*r**2, kernel), nn.PixelShuffle(r))
        # if self.r>1: self.net = nn.Sequential(Attention(in_ch, out_ch*r**2), nn.PixelShuffle(r))
# MaskUnitAttention(in_dim, d_model=16, n_heads=4, q_stride=None, nd=2)

        elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), ResBlock(in_ch*r**2, out_ch, kernel))
        # elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), Attention(in_ch*r**2, out_ch))
        elif in_ch != out_ch: self.net = ResBlock(in_ch*r**2, out_ch, kernel)
        else: self.net = lambda x: torch.zeros_like(x)

    def forward(self, x):
        return self.net(x)

def AdaptiveAvgPool_nd(n, *args, **kwargs): return [nn.Identity, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d][n](*args, **kwargs)
def AdaptiveMaxPool_nd(n, *args, **kwargs): return [nn.Identity, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d][n](*args, **kwargs)

def adaptive_avg_pool_nd(n, x, output_size): return [nn.Identity, F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d][n](x, output_size)
def adaptive_max_pool_nd(n, x, output_size): return [nn.Identity, F.adaptive_max_pool1d, F.adaptive_max_pool2d, F.adaptive_max_pool3d][n](x, output_size)

class AdaptivePool_at(nn.AdaptiveAvgPool1d): # AdaptiveAvgPool1d AdaptiveMaxPool1d
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dim
    def forward(self, x):
        x = x.transpose(self.dim,-1)
        shape = x.shape
        return super().forward(x.flatten(0,-2)).unflatten(0, shape[:-1]).transpose(self.dim,-1)


def adaptive_pool_at(x, dim, output_size, pool='avg'): # [b,c,h,w]
    x = x.transpose(dim,-1)
    shape = x.shape
    parent={'avg':F.adaptive_avg_pool1d, 'max':F.adaptive_max_pool1d}[pool]
    return parent(x.flatten(0,-2), output_size).unflatten(0, shape[:-1]).transpose(dim,-1)


class ZeroExtend():
    def __init__(self, dim=1, output_size=16):
        self.dim, self.out = dim, output_size
    def __call__(self, x): # [b,c,h,w]
        return torch.cat((x, torch.zeros(*x.shape[:self.dim], self.out - x.shape[self.dim], *x.shape[self.dim+1:])), dim=self.dim)

def make_pool_at(pool='avg', dim=1, output_size=5):
    parent={'avg':nn.AdaptiveAvgPool1d, 'max':nn.AdaptiveMaxPool1d}[pool]
    class AdaptivePool_at(parent): # AdaptiveAvgPool1d AdaptiveMaxPool1d
        def __init__(self, dim=1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim=dim
        def forward(self, x):
            x = x.transpose(self.dim,-1)
            shape = x.shape
            return super().forward(x.flatten(0,-2)).unflatten(0, shape[:-1]).transpose(self.dim,-1)
    return AdaptivePool_at(dim, output_size=output_size)

class Shortcut():
    def __init__(self, dim=1, c=3, sp=(3,3), nd=2):
        self.dim = dim
        # self.ch_pool = make_pool_at(pool='avg', dim=dim, output_size=c)
        self.ch_pool = make_pool_at(pool='max', dim=dim, output_size=c)
        # self.ch_pool = ZeroExtend(dim, output_size=c) # only for out_dim>=in_dim
        # self.sp_pool = AdaptiveAvgPool_nd(nd, sp)
        self.sp_pool = AdaptiveMaxPool_nd(nd, sp)

    def __call__(self, x): # [b,c,h,w]
        x = self.sp_pool(x) # spatial first preserves spatial more?
        x = self.ch_pool(x)
        return x

class UpDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, r=1):
        super().__init__()
        act = nn.SiLU()
        self.r = r
        self.block = PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)
        # self.block = nn.Sequential(
        #     nn.BatchNorm2d(in_ch), act, PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)
        # )
        # if self.r>1: self.res_conv = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel, 2, kernel//2, output_padding=1))
        # if self.r>1: self.res_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity())
        # if self.r>1: self.res_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity())
        # if self.r>1: self.res_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity())


        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 2, kernel//2))
        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity(), nn.MaxPool2d(2,2))
        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity(), nn.AvgPool2d(2,2))
        # elif self.r<1: self.res_conv = AttentionBlock(in_ch, out_ch, n_heads=4, q_stride=(2,2))

        # else: self.res_conv = nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity()

    def forward(self, x): # [b,c,h,w]
        b, num_tok, c, *win = x.shape
        x = x.flatten(0,1)
        out = self.block(x)
        # # shortcut = F.interpolate(x.unsqueeze(1), size=out.shape[1:], mode='nearest-exact').squeeze(1) # pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        # shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        # # shortcut = F.adaptive_max_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        # # shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) if out.shape[1]>=x.shape[1] else F.adaptive_max_pool3d(x, out.shape[1:])
        # shortcut(x)
        shortcut = Shortcut(dim=1, c=out.shape[1], sp=out.shape[-2:], nd=2)(x)
        out = out + shortcut
        out = out.unflatten(0, (b, num_tok))
        return out

        # return out + shortcut + self.res_conv(x)
        # return out + self.res_conv(x)
        # return self.res_conv(x)

# if out>in, inter=max=ave=near.
# if out<in, inter=ave. max=max

# stride2
# interconv/convpool
# pixelconv
# pixeluib
# pixelres
# shortcut

# in_ch, out_ch = 16,3
in_ch, out_ch = 3,16
model = UpDownBlock(in_ch, out_ch, r=1/2).to(device)
# model = UpDownBlock(in_ch, out_ch, r=2).to(device)

x = torch.rand(12, in_ch, 64,64, device=device)
x = torch.rand(12, 2, in_ch, 64,64, device=device)
out = model(x)

print(out.shape)


# @title maxpool path bchw
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

def conv_nd(n, *args, **kwargs): return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n](*args, **kwargs)
def maxpool_nd(n, *args, **kwargs): return [nn.Identity, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][n](*args, **kwargs)
def avgpool_nd(n, *args, **kwargs): return [nn.Identity, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d][n](*args, **kwargs)
import math

class MaskUnitAttention(nn.Module):
    # def __init__(self, d_model=16, n_heads=4, q_stride=None, nd=2):
    def __init__(self, in_dim, d_model=16, n_heads=4, q_stride=None, nd=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.scale = self.d_head**-.5
        # self.qkv = conv_nd(nd, d_model, 3*d_model, 1, bias=False)
        self.qkv = conv_nd(nd, in_dim, 3*d_model, 1, bias=False)
        # self.out = conv_nd(nd, d_model, d_model, 1)
        self.out = nn.Linear(d_model, d_model, 1)
        self.q_stride = q_stride # If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        if q_stride:
            self.q_stride = (q_stride,)*nd if type(q_stride)==int else q_stride
            self.q_pool = maxpool_nd(nd, self.q_stride, stride=self.q_stride)

    def forward(self, x): # [b,num_tok,c,win1,win2]
        b, num_tok, c, *win = x.shape
        # x = x.transpose(1,2).flatten(0,1) # [b*num_tok,c,win1,win2]
        x = x.flatten(0,1) # [b*num_tok,c,win1,win2]
        # print(x.shape)
        # q, k, v = self.qkv(x).reshape(b, 3, self.n_heads, self.d_head, num_tok, win*win).permute(1,0,2,4,5,3) # [b,3*d_model,num_tok*win,win] -> 3* [b, n_heads, num_tok, win*win, d_head]
        # self.qkv(x).flatten(1,-2).unflatten(-1, (self.heads,-1)).chunk(3, dim=-1) # [b,sp,n_heads,d_head]
        q,k,v = self.qkv(x).chunk(3, dim=1) # [b*num_tok,d,win,win]
        if self.q_stride:
            q = self.q_pool(q)
            win=[w//s for w,s in zip(win, self.q_stride)] # win = win/q_stride
        if math.prod(win) >= 200:
            print('MUattn', math.prod(win))
            q, k, v = map(lambda t: t.reshape(b*num_tok, self.n_heads, self.d_head, -1).transpose(-2,-1), (q,k,v)) # [b*num_tok, n_heads, win*win, d_head]
        else: q, k, v = map(lambda t: t.reshape(b, num_tok, self.n_heads, self.d_head, -1).permute(0,2,1,4,3).flatten(2,3), (q,k,v)) # [b, n_heads, num_tok*win*win, d_head]

        q, k = q.softmax(dim=-1)*self.scale, k.softmax(dim=-2) # [b, n_heads, t(/pool), d_head]
        context = k.transpose(-2,-1) @ v # [b, n_heads, d_head, d_head]
        x = q @ context # [b, n_heads, t(/pool), d_head]

        # print('attn fwd 2',x.shape)
        # x = x.transpose(1, 3).reshape(B, -1, self.d_model)
        x = x.transpose(1,2).reshape(b, -1, self.d_model) # [b,t,d]
        # x = x.transpose(-2,-1).reshape(x.shape[0], self.d_model, ) # [b,t,d]
        # [b, n_heads, num_tok, win*win, d_head] -> [b, n_heads, d_head, num_tok, win*win] -> [b,c,num_tok*win,win]
        # x = x.permute(0,1,4,2,3).reshape(b, self.d_model, num_tok*win,win) # [b, n_heads, num_tok, win*win, d_head]
        # x = x.transpose(-2,-1).reshape(b, self.d_model, num_tok, *win) # [b*num_tok, n_heads, win*win, d_head]
        x = self.out(x)
        L=len(win)
        x = x.reshape(b, num_tok, *win, self.d_model).permute(0,1,L+2,*range(2,L+2)) # [b, num_tok, out_dim, *win]
        # x = x.unflatten(0, (b, num_tok))
        return x # [b,num_tok,c,win1,win2]

d_model=16
model = MaskUnitAttention(d_model, n_heads=4, q_stride=2)
# MaskUnitAttention(d_model=16, n_heads=4, q_stride=None, nd=2)
# x=torch.randn(2,4,4,3)
# x=torch.randn(2,3,d_model,8,8)
x=torch.randn(2,3,d_model,32,32)
# [b*num_tok,c,win,win]

out = model(x)
print(out.shape)


# @title hiera vit me
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class LayerNorm_at(nn.RMSNorm): # LayerNorm RMSNorm
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dim
    def forward(self, x):
        return super().forward(x.transpose(self.dim,-1)).transpose(self.dim,-1)

def unshuffle(x, window_shape): # [b,c,h,w] -> [b, h/win1* w/win2, c, win1,win2]
    new_shape = list(x.shape[:2]) + [val for xx, win in zip(list(x.shape[2:]), window_shape) for val in [xx//win, win]] # [h,w]->[h/win1, win1, w/win2, win2]
    x = x.reshape(new_shape) # [b, c, h/win1, win1, w/win2, win2]
    # print('unsh',x.shape, window_shape, new_shape)
    L = len(new_shape)
    permute = ([0] + list(range(2, L - 1, 2)) + [1] + list(range(3, L, 2))) # [0,2,4,1,3,5] / [0,2,4,6,1,3,5,6]
    return x.permute(permute).flatten(1, L//2-1) # [b, h/win1* w/win2, c, win1,win2]

class AttentionBlock(nn.Module):
    # def __init__(self, d_model, n_heads, q_stride=None, mult=4, drop=0, nd=2):
    def __init__(self, in_dim, d_model, n_heads, q_stride=None, mult=4, drop=0, nd=2):
        super().__init__()
        self.d_model = d_model
        # self.norm = LayerNorm_at(2, d_model) # LayerNorm RMSNorm
        self.norm = LayerNorm_at(2, in_dim) # LayerNorm RMSNorm
        self.drop = nn.Dropout(drop)
        # self.attn = MaskUnitAttention(d_model, n_heads, q_stride)
        self.attn = MaskUnitAttention(in_dim, d_model, n_heads, q_stride)
        ff_dim=d_model*mult
        self.ff = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, ff_dim), nn.ReLU(), # ReLU GELU
            nn.RMSNorm(ff_dim), nn.Dropout(drop), nn.Linear(ff_dim, d_model)
            # nn.RMSNorm(d_model), act, nn.Linear(d_model, ff_dim),
            # nn.RMSNorm(ff_dim), act, nn.Linear(ff_dim, d_model)
            # nn.RMSNorm(d_model), nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout), # ReLU GELU
            # nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        # self.res = conv_nd(nd, d_model, d_model, q_stride, q_stride) if q_stride else nn.Identity()

        # self.res = nn.Sequential(
        #     conv_nd(nd, in_dim, d_model, 1, 1) if in_dim!=d_model else nn.Identity(),
        #     # maxpool_nd(nd, q_stride, q_stride) if q_stride else nn.Identity(),
        #     avgpool_nd(nd, q_stride, q_stride) if q_stride else nn.Identity(),
        # )
        self.res = UpDownBlock(in_dim, d_model, kernel=1, r=1/2 if q_stride else 1)

        # x = self.proj(x_norm).unflatten(1, (self.attn.q_stride, -1)).max(dim=1).values # pooling res # [b, (Sy, Sx, h/Sy, w/Sx), c] -> [b, (Sy, Sx), (h/Sy, w/Sx), c] -> [b, (h/Sy, w/Sx), c]


    def forward(self, x): # [b, num_tok, c, *win]
        b, num_tok, c, *win = x.shape
        # print('attnblk fwd',x.shape)
        # x = x + self.drop(self.self(self.norm(x)))
        # print('attnblk fwd',self.res(x.flatten(0,1)).shape, self.drop(self.attn(self.norm(x))).flatten(0,1).shape)
        # x = self.res(x.flatten(0,1)) + self.drop(self.attn(self.norm(x))).flatten(0,1) # [b*num_tok,c,win1,win2,win3]
        x = self.res(x) + self.drop(self.attn(self.norm(x))) # [b*num_tok,c,win1,win2,win3]
        # x = x + self.ff(x)
        # x = x + self.ff(x.transpose(1,-1)).transpose(1,-1)
        x = x + self.ff(x.transpose(2,-1)).transpose(2,-1)
        # x = self.ff(x)
        # x = x.unflatten(0, (b, num_tok))
        return x

    # def forward(self, x): # [b,t,c] # [b, (Sy, Sx, h/Sy, w/Sx), c]
    #     # Attention + Q Pooling
    #     x_norm = self.norm1(x)
    #     if self.dim != self.dim_out:
    #         x = self.proj(x_norm).unflatten(1, (self.attn.q_stride, -1)).max(dim=1).values # pooling res # [b, (Sy, Sx, h/Sy, w/Sx), c] -> [b, (Sy, Sx), (h/Sy, w/Sx), c] -> [b, (h/Sy, w/Sx), c]
    #     x = x + self.drop_path(self.attn(x_norm))
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x


class levelBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=None, depth=1, r=1):
        super().__init__()
        self.seq = nn.Sequential(
            UpDownBlock(in_ch, out_ch, r=min(1,r)) if in_ch != out_ch or r<1 else nn.Identity(),
            # AttentionBlock(d_model, d_model, n_heads, q_stride) if in_ch != out_ch or r<1 else nn.Identity(),
            # AttentionBlock(d_model, d_model, n_heads, q_stride=(2,2)),
            *[AttentionBlock(d_model, d_model, n_heads) for i in range(1)],
            # UpDownBlock(out_ch, out_ch, r=r) if r>1 else nn.Identity(),
        )
    def forward(self, x):
        return self.seq(x)

class SimpleViT(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, n_heads, depth):
        super().__init__()
        self.embed = nn.Sequential( # in, out, kernel, stride, pad
            # nn.Conv2d(in_dim, d_model, 7, 2, 7//2, bias=False), nn.MaxPool2d(3,2,1), # nn.MaxPool2d(2,2)
            nn.Conv2d(in_dim, d_model, 7, 1, 7//2, bias=False),
            # UpDownBlock(in_dim, dim, r=1/2, kernel=3), UpDownBlock(dim, dim, r=1/2, kernel=3)
            # nn.PixelUnshuffle(2), nn.Conv2d(in_dim*2**2, dim, 1, bias=False),
            # ResBlock(in_dim, d_model, kernel),
            )
        # # self.pos_emb = LearnedRoPE2D(dim) # LearnedRoPE2D, RoPE2D
        # self.pos_emb = nn.Parameter(torch.zeros(1, d_model, 32,32)) # positional_embedding == 'learnable'

        emb_shape = (32,32)
        # emb_shape = (8,32,32)
        if len(emb_shape) == 3: # for video
            pos_spatial = nn.Parameter(torch.randn(1, emb_shape[1]*emb_shape[2], d_model)*.02)
            pos_temporal = nn.Parameter(torch.randn(1, emb_shape[0], d_model)*.02)
            self.pos_emb = pos_spatial.repeat(1, emb_shape[0], 1) + torch.repeat_interleave(pos_temporal, emb_shape[1] * emb_shape[2], dim=1)
        elif len(emb_shape) == 2: # for img
            self.pos_emb = nn.Parameter(torch.randn(1, math.prod(emb_shape), d_model)*.02) # 56*56=3136


        # self.blocks = nn.Sequential(*[AttentionBlock(d_model, n_heads, q_stride=(2,2) if i in [1,3] else None) for i in range(depth)])
        # mult = [1,1,1,1]
        mult = [1,2,4,8] # [1,2,3,4] [1,2,2,2]
        ch_list = [d_model * m for m in mult] # [128, 256, 384, 512]

        self.blocks = nn.Sequential(
            # AttentionBlock(ch_list[0], ch_list[1], n_heads, q_stride=(2,2)),
            # UpDownBlock(ch_list[0], ch_list[1], kernel=1, r=1/2),
            UpDownBlock(ch_list[0], ch_list[1], kernel=4, r=1/2),
            AttentionBlock(ch_list[1], ch_list[1], n_heads),
            # AttentionBlock(ch_list[1], ch_list[2], n_heads, q_stride=(2,2)),
            # UpDownBlock(ch_list[1], ch_list[2], kernel=1, r=1/2),
            UpDownBlock(ch_list[1], ch_list[2], kernel=2, r=1/2),
            AttentionBlock(ch_list[2], ch_list[2], n_heads),
            )
        self.attn_pool = nn.Linear(ch_list[2], 1)
        self.out = nn.Linear(ch_list[2], out_dim, bias=False)

    def forward(self, img): # [b,c,h,w]
        x = self.embed(img)
        # print('vit fwd', x.shape)
        x = x + self.pos_emb
        x = unshuffle(x, (4,4)) # [b,num_tok,c,win1,win2,win3] or [b,1,c,f,h,w]
        # mask # [b, num_tok]
        # x=x[:,:7]
        # print('vit fwd1', x.shape)

        x = self.blocks(x) # [b,num_tok,c,1,1,1]
        # print('vit fwd2', x.shape)
        x = x.squeeze() # [b,num_tok,d]
        attn = self.attn_pool(x).squeeze(-1) # [batch, (h,w)] # seq_pool
        x = (attn.softmax(dim=1).unsqueeze(1) @ x).squeeze(1) # [batch, 1, (h,w)] @ [batch, (h,w), dim] -> [batch, dim]
        return self.out(x)


# pos_emb rope < learn < learned
# conv > pixel?
# droppath not required

# norm,act,conv < conv,norm,act
# 2*s1 < uib < resblock
# gatedadaln 3 < 2 = 1 < ffmult4 = 2*gatedadaln
# MaxPool2d(2,2) < MaxPool2d(3,2,3//2)

# patchattn only for

# multiendfusion negligible diff

d_model = 64
dim_head = 8
heads = d_model // dim_head
num_classes = 10
# model = SimpleViT(image_size=32, patch_size=4, num_classes=num_classes, dim=dim, depth=1, heads=heads, mlp_dim=dim*4, channels = 3, dim_head = dim_head)
model = SimpleViT(in_dim=3, out_dim=num_classes, d_model=d_model, depth=5, n_heads=heads).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 59850
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

# print(images.shape) # [batch, 3, 32, 32]
x = torch.rand(25, 3, 32, 32, device=device)
# x = torch.rand(64, 3, 28,28, device=device)
logits = model(x)
print(logits.shape)
# print(logits[0])
# print(logits[0].argmax(1))
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")


