import torch
import torch.nn.functional as F 
from torch import nn 

from einops import rearrange
from net.SPyNet import SPyNet
from net.arch_util import ResidualBlockNoBN, flow_warp, make_layer
import numbers


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x, ref):
        b,c,h,w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k,v = kv.chunk(2, dim=1) 

        q = self.q_dwconv(self.q(ref))  
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input):
        x, ref = input[0], input[1]
        x = x + self.attn(self.norm1(x), self.norm3(ref))
        x = x + self.ffn(self.norm2(x))

        return [x, ref]


class Network(nn.Module):
    def __init__(self, config, spynet_path=None):
        super(Network, self).__init__()
        self.config = config
        self.num_feat = config.num_feat
        self.num_block = config.num_blocks
        self.device = config.device

        # Flow-based Feature Alignment
        self.spynet = SPyNet(pretrained=config.spynet, device=config.device)

       # Bidirectionaal Propagation
        self.forward_resblocks = ConvResBlock(config.num_feat * 2, config.num_feat, config.num_blocks)
        self.backward_resblocks = ConvResBlock(config.num_feat * 2, config.num_feat, config.num_blocks)

        # Concatenate Aggregation
        self.concate = nn.Conv2d(config.num_feat * 2, config.num_feat, kernel_size=1, stride=1, padding=0, bias=True)

        # Pixel-Shuffle Upsampling
        self.up1 = PSUpsample(config.num_feat, config.num_feat, scale_factor=2)
        self.up2 = PSUpsample(config.num_feat, 64, scale_factor=2)

        # The channel of the tail layers is 64
        self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Global Residual Learning
        self.img_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.fusion_resblocks = ConvResBlock(config.num_feat * 2, config.num_feat, config.num_blocks)

        self.encoder_attention = nn.Sequential(*[TransformerBlock(dim=64, num_heads=2, ffn_expansion_factor=2.66, bias=False, \
            LayerNorm_type='WithBias') for i in range(4)])
        self.decoder_attention = nn.Sequential(*[TransformerBlock(dim=64, num_heads=4, ffn_expansion_factor=2.66, bias=False, \
            LayerNorm_type='WithBias') for i in range(4)])

        self.lr_resblocks = ConvResBlock(3, config.num_feat, config.enc_num_blocks)
        self.ref_resblocks = ConvResBlock(3, config.num_feat, config.enc_num_blocks)
        

    def comp_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Args:
            lrs (tensor): LR frames, the shape is (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 
            forward_flow refers to the flow from current frame to the previous frame. 
            backward_flow is the flow from current frame to the next frame.
        """
        n, t, c, h, w = lrs.size()
        forward_lrs = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)    # n t c h w -> (n t) c h w
        backward_lrs = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # n t c h w -> (n t) c h w
        
        forward_flow = self.spynet(forward_lrs, backward_lrs).view(n, t-1, 2, h, w)
        backward_flow = self.spynet(backward_lrs, forward_lrs).view(n, t-1, 2, h, w)

        return forward_flow, backward_flow

    def forward(self, lrs, refs):
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.')
   
        forward_flow, backward_flow = self.comp_flow(lrs)
        
        lrs_feas = []
        for i in range(0, t):
            lrs_feas.append(self.lr_resblocks(lrs[:, i, :, :, :]))

        if len(refs.size()) == 4:
            refs = refs
        else: 
            refs = refs[:, refs.size(1)//2, :, :, :]
        ref_fea = self.ref_resblocks(refs)
      
        lrs_feas[lrs.size(1)//2] = self.encoder_attention([lrs_feas[lrs.size(1)//2], ref_fea])[0]
        
        # Backward Propagation
        rlt = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t-1, -1, -1):
            curr_lr = lrs_feas[i]
            if i < t-1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([curr_lr, feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            rlt.append(feat_prop)
        rlt = rlt[::-1]

        # Forward Propagation
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            curr_lr = lrs_feas[i]
            if i > 0:
                flow = forward_flow[:, i-1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([curr_lr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # Fusion and Upsampling
            cat_feat = torch.cat([rlt[i], feat_prop], dim=1)
            cat_feat = self.lrelu(self.concate(cat_feat))
            sr_rlt = self.decoder_attention([cat_feat, ref_fea])[0]
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            # Global Residual Learning
            base = self.img_up(lrs[:, i, :, :, :])

            sr_rlt += base
            rlt[i] = sr_rlt
        outs = torch.stack(rlt, dim=1)
        return outs

#############################
# Conv + ResBlock
class ConvResBlock(nn.Module):
    def __init__(self, in_feat, out_feat=64, num_block=30):
        super(ConvResBlock, self).__init__()

        conv_resblock = []
        conv_resblock.append(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True))
        conv_resblock.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv_resblock.append(make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat))

        self.conv_resblock = nn.Sequential(*conv_resblock)

    def forward(self, x):
        return self.conv_resblock(x)

# Conv + ResBlock
class ConvResBlock_ref(nn.Module):
    def __init__(self, in_feat, out_feat=64, num_block=30):
        super(ConvResBlock_ref, self).__init__()

        conv_resblock = []
        conv_resblock.append(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1, bias=True))
        conv_resblock.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv_resblock.append(make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat))

        self.conv_resblock = nn.Sequential(*conv_resblock)

    def forward(self, x):
        return self.conv_resblock(x)

#############################
# Upsampling with Pixel-Shuffle
class PSUpsample(nn.Module):
    def __init__(self, in_feat, out_feat, scale_factor):
        super(PSUpsample, self).__init__()

        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(in_feat, out_feat*scale_factor*scale_factor, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)


class SRNet(nn.Module):
    def __init__(self, config):
        super(SRNet, self).__init__()
        self.config = config
        self.device = config.device
        self.Network = Network(config)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = self.config.wi)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, self.config.win)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                
    def forward(self, x, ref):
        outs = self.Network.forward(x, ref)
        return outs