import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ["btresnext101_32x16d_wsl"]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',

    # https://github.com/facebookresearch/WSL-Images/blob/master/hubconf.py
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MHSA(nn.Module):
    def __init__(self, n_dims, width, height):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        print("n_dims: ", n_dims)
        print("height: ", height)
        print("width: ", width)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)
        print("q: ", q.shape)
        print("q.permute", q.permute(0, 2, 1).shape)
        print("k: ", k.shape)
        content_content = torch.bmm(q.permute(0, 2, 1), k)
        print("content_content: ", content_content.shape)
        
        print("rel_h: ", self.rel_h.shape)
        print("rel_w: ", self.rel_w.shape)
        print("self.rel_h + self.rel_w: ", (self.rel_h + self.rel_w).shape)
        # print("heads: ", self.heads)
        print("C: ", C)
        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        print("content_position1: ", content_position.shape)
        content_position = torch.matmul(content_position, q)
        print("content_position2: ", content_position.shape)
        
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if not mhsa:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            self.conv2 = nn.ModuleList()
            print("resolution: ",resolution)
            self.conv2.append(MHSA(width, width=int(resolution[0]), height=int(resolution[1])))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
        
        # identity = x
        # print("00000")
        # print(x.shape)
        # print("11111")
        # out = self.conv1(x)
        # print(out.shape)
        # out = self.bn1(out)
        # print(out.shape)
        # out = self.relu(out)
        # print(out.shape)
        # print("22222")
        # out = self.conv2(out)
        # print(out.shape)
        # out = self.bn2(out)
        # print(out.shape)
        # out = self.relu(out)
        # print(out.shape)
        # print("33333")
        # out = self.conv3(out)
        # print(out.shape)
        # out = self.bn3(out)
        # print(out.shape)
        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        # return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, resolution=(224, 224)):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        # 解析率
        self.resolution = list(resolution)
        # print("self.resolution1 ", self.resolution)
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
        
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        # print("self.resolution2 ", self.resolution)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2
        # print("self.resolution3 ", self.resolution)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], mhsa=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
        #     nn.Linear(512 * block.expansion, num_classes)
        # )

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
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, mhsa=False):
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
                            self.base_width, previous_dilation, norm_layer, mhsa, self.resolution))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, mhsa=mhsa, resolution=self.resolution))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            # print("self.resolution4 ", self.resolution)
            # self.inplanes = planes * block.expansion
            # layers.append(block(self.inplanes, planes, groups=self.groups,
            #                     base_width=self.base_width, dilation=self.dilation,
            #                     norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print("end3")
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    # new_state_dict = {}
    new_state_dict = model.state_dict()
    print("11111", len(state_dict)) 
    for k,v in state_dict.items():
      # print(k)
      if k in new_state_dict.keys() and (not k.startswith('fc') or not k.startswith('layer4')): # 不加载全连接层
        new_state_dict[k] = v
    #     # print("{}: {}".format(k,v.requires_grad))
    #   # elif k.startswith('fc') or k.startswith('layer4'):
    #   #   print("yes, ",k)
    # new_state_dict = model.state_dict() 
    # new_state_dict.update(state_dict)
    print("22222", len(new_state_dict))
    model.load_state_dict(new_state_dict)
    return model

# def ResNet50(num_classes=1000, resolution=(224, 224), heads=4):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads)

def btresnext101_32x16d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    kwargs['resolution'] = (224, 224)
    # kwargs['heads'] = 4
    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

def main():
    x = torch.randn([2, 3, 224, 224])
    model = ResNet50(resolution=tuple(x.shape[2:]), heads=8)
    print(model(x).size())
    print(get_n_params(model))


# if __name__ == '__main__':
    # main()