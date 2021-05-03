import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
#######rel-pos-embedding
from torch import einsum
from einops import rearrange

__all__ = ['BoTNet101']

model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}
#使用部分加载
def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    print(layers)
    print("group:{}, wpg:{}".format(kwargs['groups'], kwargs['width_per_group']))
    model = ResNet_with_BotStack(block, layers, **kwargs)
    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    new_state_dict = model.state_dict()
    new_state_dict.update(state_dict)
    model.load_state_dict(new_state_dict, strict=False)
    return model

def BoTNet101(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:zz
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    kwargs['fmap_size'] = (14, 14)
    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

def expand_dim(t, dim, k):
    """
    Expand dims for t at dim to k
    """
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    """
    x: [B, Nh * H, L, 2L - 1]
    Convert relative position between the key and query to their absolute position respectively.
    Tensowflow source code in the appendix of: https://arxiv.org/pdf/1904.09925.pdf
    """
    B, Nh, L, _ = x.shape
    # pad to shift from relative to absolute indexing
    #去掉.cuda()
    col_pad = torch.zeros((B, Nh, L, 1))
    x = torch.cat((x, col_pad), dim=3)
    flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
    flat_pad = torch.zeros((B, Nh, L - 1))
    flat_x = torch.cat((flat_x, flat_pad), dim=2)
    # Reshape and slice out the padded elements
    final_x = torch.reshape(flat_x, (B, Nh, L + 1, 2 * L - 1))
    return final_x[:, :, :L, L - 1:]


def relative_logits_1d(q, rel_k):
    """
    q: [B, Nh, H, W, d]
    rel_k: [2W - 1, d]
    Computes relative logits along one dimension.
    The details of relative position is explained in: https://arxiv.org/pdf/1803.02155.pdf
    """
    B, Nh, H, W, _ = q.shape
    rel_logits = torch.einsum('b n h w d, m d -> b n h w m', q, rel_k)
    # Collapse height and heads
    rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
    rel_logits = expand_dim(rel_logits, dim=3, k=H)
    return rel_logits


class AbsPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        assert height == width
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits


class RelPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        assert height == width
        scale = dim_head ** -0.5
        self.fmap_size = height
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h = w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

#####MHSA
class MHSA(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        heads = 4,
        dim_qk = 128,
        dim_v = 128,
        rel_pos_emb = False):
        """
        dim: number of channels of feature map
        fmap_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()
        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = nn.Conv2d(dim, out_channels_qk * 2, 1, bias=False) # 1*1 conv to compute q, k
        self.to_v = nn.Conv2d(dim, out_channels_v, 1, bias=False) # 1*1 conv to compute v
        self.softmax = nn.Softmax(dim=-1)

        height, width = fmap_size
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(height, width, dim_qk)
        else:
            self.pos_emb = AbsPosEmb(height, width, dim_qk)

    def forward(self, featuremap):
        """
        featuremap: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        heads = self.heads
        B, C, H, W = featuremap.shape
        q, k = self.to_qk(featuremap).chunk(2, dim=1)
        v = self.to_v(featuremap)
        q, k, v = map(lambda x: rearrange(x, 'B (h d) H W -> B h (H W) d', h=heads), (q, k, v))

        q *= self.scale

        logits = einsum('b h x d, b h y d -> b h x y', q, k)
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = einsum('b h x y, b h y d -> b h x d', weights, v)
        attn_out = rearrange(attn_out, 'B h (H W) d -> B (h d) H W', H=H)

        return attn_out

#####BoT
"""Bottleneck Transformer (BoT) Block."""
class BoTBlock(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out,
        stride = 1,
        heads = 4,
        proj_factor = 4,
        dim_qk = 128,
        dim_v = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()):
        """
        dim: channels in feature map
        dim_out: output channels for feature map
        """
        super().__init__()
        if dim != dim_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()
        
        bottleneck_dimension = dim_out // proj_factor # from 2048 to 512
        attn_dim_out = heads * dim_v

        self.net = nn.Sequential(
            nn.Conv2d(dim, bottleneck_dimension, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_dimension),
            activation,
            MHSA(
                dim = bottleneck_dimension,
                fmap_size = fmap_size,
                heads = heads,
                dim_qk = dim_qk,
                dim_v = dim_v,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if stride == 2 else nn.Identity(), # same padding
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        nn.init.zeros_(self.net[-1].weight) # last batch norm uses zero gamma initializer
        self.activation = activation
    
    def forward(self, featuremap):
        shortcut = self.shortcut(featuremap)
        featuremap = self.net(featuremap)
        featuremap += shortcut
        return self.activation(featuremap)


"""c5 Blockgroup of BoT Blocks."""
class BoTStack(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out = 2048,
        heads = 4,
        proj_factor = 4,
        num_layers = 3,
        stride = 2, 
        dim_qk = 128,
        dim_v = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        """
        dim: channels in feature map
        fmap_size: [H, W]
        """
        super().__init__()

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)

            fmap_divisor = (2 if stride == 2 and not is_first else 1)
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(BoTBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                stride = stride if is_first else 1,
                heads = heads,
                proj_factor = proj_factor,
                dim_qk = dim_qk,
                dim_v = dim_v,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim
        assert h == self.fmap_size[0] and w == self.fmap_size[1]
        return self.net(x)

""" 
Official Source Code for ResNet
See reference in: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


"""
Add BoTNet block to layer 5 in ResNet-50 if specified, else return regular ResNet-50
"""
class ResNet_with_BotStack(nn.Module):

    def __init__(
        self,
        block: Tensor = Bottleneck,
        layers: List[int] = [3, 4, 23, 3],
        fmap_size: tuple = (14,14),
        botnet: bool = True,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 32,
        width_per_group: int = 16,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet_with_BotStack, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        block = Bottleneck

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        if botnet:
            self.layer4 = BoTStack(dim=1024, fmap_size=fmap_size, stride=1, rel_pos_emb=True)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

if __name__ == '__main__':
  # model = ResNet_with_BotStack(fmap_size=(14, 14), botnet=True)
  model = BoTNet101()
  # for k,v in model.named_parameters():
  # # print(k)
  #   print("{}: {}".format(k, v.requires_grad))
  # print(model)
  img = torch.randn(2, 3, 224, 224)
  preds = model(img) # (2, 1000)
  print(preds)
    
    
    