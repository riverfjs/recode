from args import args
from build_net import make_model

model = make_model(args)
# print(model)
for k,v in model.named_parameters():
      # print("{}: {}".format(k,v.requires_grad))
      if not k.startswith('7') and not k.startswith('6') and not k.startswith('fc'):
        # print(k)
        v.requires_grad = False
# sys.exit(0)
for k,v in model.named_parameters():
  # print(k)
  print("{}: {}".format(k, v.requires_grad))
