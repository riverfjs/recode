import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from torch import nn
import torchvision.models as models
import models as customized_models

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names
# model_names = default_model_names + customized_models_names
# customized_models = \
#     {
#         "resnext101_32x16d_wsl": resnext_wsl,
#     }
def make_model(args=None, predict=False, modelname=None, num_classes=None):
    # print(args.arch)
    if not predict:
      print("=> creating model '{}'".format(args.arch))
      model = models.__dict__[args.arch](progress=True)
      model.fc = nn.Sequential(
          nn.Dropout(0.2),
          nn.Linear(2048, args.num_classes)
      )
      print(type(model))
      return model
    else:
      print("predict forward {}".format(num_classes))
      print("=> creating model '{}'".format(modelname))
      model = models.__dict__[modelname](progress=True)
      model.fc = nn.Sequential(
          nn.Dropout(0.2),
          nn.Linear(2048, num_classes)
      )
      # print(type(model))
      return model

if __name__=='__main__':
    # print(models.__dict__['resnext101_32x16d_wsl'](progress=True))
    all_model = sorted(name for name in models.__dict__ if not name.startswith("__"))
    print(all_model)
#     # print(len(default_model_names))
#     # print(len(customized_models_names))
#     # print(customized_models_names)
#     print(model_names)