import torch
from torch import nn
from .resnext_wsl import resnext101_32x16d_wsl

from .bottleneck_transformer_pytorch import BottleStack

__all__ = ["botnext101"]
def botnext101(num_classes=1000, progress=True):
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
  resnet = resnext101_32x16d_wsl(progress=True)

  # model surgery

  backbone = list(resnet.children())
  # for k,v in nn.Sequential(*backbone[:7]).named_parameters():
  # # print(k)
  #   print("{}: {}".format(k, v.requires_grad))
  # sys.exit(0)

  # print(backbone[:7])
  # sys.exit(0)
  model = nn.Sequential(
      *backbone[:7],  # 取layer4之前的层，固定不变
      layer,
      # *backbone[4:],
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(1),
      # nn.Dropout(0.2),
      # nn.Linear(2048, num_classes)
  )
  # state_dict = model.state_dict()
  # print("111111111111:{}".format(model))
  return model
# print(model)
# model = resnet
# for k,v in enumerate(backbone[:8]):
#   print("{}: {}".format(k,v))
# print(backbone[:5])

# use the 'BotNet'
def test():
  model = botnext101()
  img = torch.randn(2, 3, 224, 224)
  preds = model(img) # (2, 1000)
  print(preds)
# test()