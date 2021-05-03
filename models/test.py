import torch
from torch import nn
from .resnet import resnet50

from bottleneck_transformer_pytorch import BottleStack

layer = BottleStack(
    dim = 1024,
    fmap_size = 14,        # set specifically for imagenet's 224 x 224
    dim_out = 2048,
    proj_factor = 4,
    downsample = True,
    heads = 4,
    dim_head = 128,
    rel_pos_emb = True,
    activation = nn.ReLU()
)
resnet = resnet50(pretrained=True)

# model surgery

backbone = list(resnet.children())
# print(backbone[:7])
# print(backbone)
model = nn.Sequential(
    *backbone[:7],  # 取layer4之前的层，固定不变
    layer,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(1),
    nn.Dropout(0.2),
    nn.Linear(2048, 40)
)
# print(model)
# model = resnet
# for k,v in enumerate(backbone[:8]):
#   print("{}: {}".format(k,v))
# print(backbone[:5])

# use the 'BotNet'

img = torch.randn(2, 3, 224, 224)
preds = model(img) # (2, 1000)
print(preds)