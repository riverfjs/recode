import torch
import time
import torchvision
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn import metrics  # 计算混淆矩阵
# from progress.bar import Bar
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from args import args
from build_net import make_model
from transform import get_transforms
import dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, get_optimizer, save_checkpoint

class_id2name = {
    0: "其他垃圾/一次性快餐盒",
    1: "其他垃圾/污损塑料",
    2: "其他垃圾/烟蒂",
    3: "其他垃圾/牙签",
    4: "其他垃圾/破碎花盆及碟碗",
    5: "其他垃圾/竹筷",
    6: "厨余垃圾/剩饭剩菜",
    7: "厨余垃圾/大骨头",
    8: "厨余垃圾/水果果皮",
    9: "厨余垃圾/水果果肉",
    10: "厨余垃圾/茶叶渣",
    11: "厨余垃圾/菜叶菜根",
    12: "厨余垃圾/蛋壳",
    13: "厨余垃圾/鱼骨",
    14: "可回收物/充电宝",
    15: "可回收物/包",
    16: "可回收物/化妆品瓶",
    17: "可回收物/塑料玩具",
    18: "可回收物/塑料碗盆",
    19: "可回收物/塑料衣架",
    20: "可回收物/快递纸袋",
    21: "可回收物/插头电线",
    22: "可回收物/旧衣服",
    23: "可回收物/易拉罐",
    24: "可回收物/枕头",
    25: "可回收物/毛绒玩具",
    26: "可回收物/洗发水瓶",
    27: "可回收物/玻璃杯",
    28: "可回收物/皮鞋",
    29: "可回收物/砧板",
    30: "可回收物/纸板箱",
    31: "可回收物/调料瓶",
    32: "可回收物/酒瓶",
    33: "可回收物/金属食品罐",
    34: "可回收物/锅",
    35: "可回收物/食用油桶",
    36: "可回收物/饮料瓶",
    37: "有害垃圾/干电池",
    38: "有害垃圾/软膏",
    39: "有害垃圾/过期药物"
}
class_list = [class_id2name[i] for i in list(range(args.num_classes))]
def main():
    # Data
    TRAIN = args.trainroot
    VAL = args.valroot
    # TRAIN = '/content/train'
    # VAL = '/content/val'
    transform = get_transforms(input_size=args.image_size, test_size=args.image_size, backbone=None)

    print('==> Preparing dataset %s' % args.trainroot)
    # trainset = datasets.ImageFolder(root=TRAIN, transform=transform['train'])
    # valset = datasets.ImageFolder(root=VAL, transform=transform['val'])
    trainset = dataset.Dataset(root=args.trainroot, transform=transform['train'])
    valset = dataset.TestDataset(root=args.valroot, transform=transform['val'])

    train_loader = DataLoader(
        trainset, 
        batch_size=args.train_batch, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True)
    
    val_loader = DataLoader(
        valset, 
        batch_size=args.test_batch, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True)

    # model initial
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(args)

    # TODO:merge in function
    for k,v in model.named_parameters():
      # print("{}: {}".format(k,v.requires_grad))
      if not k.startswith('layer4') and not k.startswith('fc'):
        # print(k)
        v.requires_grad = False
    # sys.exit(0)
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    elif args.ngpu:
        model = torch.nn.DataParallel(model).cuda()

    model.to(device)

    cudnn.benchmark = True

    print('Total params:%.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))  # 打印模型参数数量
    print('Trainable params:%.2fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad)/ 1000000.0))
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(model, args)
    
    # 基于标准的学习率更新
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)

    # Resume
    epochs = args.epochs
    start_epoch = args.start_epoch
    title = 'log-' + args.arch
    if args.resume:
        # --resume checkpoint/checkpoint.pth.tar
        # load checkpoint
        print('Resuming from checkpoint...')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!!'
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        optim = checkpoint['optimizer']
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(optim)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        # logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        logger.set_names(['LR', 'epoch', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.',])
    
    # Evaluation:Confusion Matrix:Precision  Recall F1-score
    if args.evaluate and args.resume:
        print('\nEvaluate only')
        test_loss, test_acc, test_acc_5, predict_all, labels_all = test_model(val_loader, model, criterion, device, test=True)
        
        print('Test Loss:%.8f,Test top1:%.2f top5:%.2f' %(test_loss,test_acc,test_acc_5))
        # 混淆矩阵
        report = metrics.classification_report(labels_all,predict_all,target_names=class_list,digits=4)
        confusion = metrics.confusion_matrix(labels_all,predict_all)
        print('\n report ',report)
        print('\n confusion',confusion)
        with open(args.resume[:-3]+"txt", "w+") as f_obj:
          f_obj.write(report)
        # plot_Matrix(args.resume[:-3], confusion, class_list)
        return
    
    # model train and val
    best_acc = 0
    for epoch in range(start_epoch, epochs + 1):
        print('[{}/{}] Training'.format(epoch, args.epochs))
        # train
        train_loss, train_acc, train_acc_5 = train_model(train_loader, model, criterion, optimizer, device)
        # val
        test_loss, test_acc, test_acc_5 = test_model(val_loader, model, criterion, device, test=None)

        scheduler.step(test_loss)

        lr_ = optimizer.param_groups[0]['lr']
        # 核心参数保存logger
        logger.append([lr_, int(epoch), train_loss, test_loss, train_acc, test_acc,])
        print('train_loss:%f, val_loss:%f, train_acc:%f,  val_acc:%f, train_acc_5:%f,  val_acc_5:%f' % (train_loss, test_loss, train_acc, test_acc, train_acc_5, test_acc_5))
        # 保存模型 保存最优
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if not args.ngpu:
            name = 'checkpoint_' + str(epoch) + '.pth.tar'
        else:
            name = 'ngpu_checkpoint_' + str(epoch) + '.pth.tar'
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()

        }, is_best, checkpoint=args.checkpoint, filename=name)
        # logger.close()
        # logger.plot()
        # savefig(os.path.join(args.checkpoint, 'log.eps'))
        print('Best acc:')
        print(best_acc)



def train_model(train_loader, model, criterion, optimizer, device):

        # 定义保存更新变量
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        model.train()

        # 训练每批数据，然后进行模型的训练
        ## 定义bar 变量
        bar = Bar('Processing', max=len(train_loader))
        for batch_index, (inputs, targets) in enumerate(train_loader):
            data_time.update(time.time() - end)
            # move tensors to GPU if cuda is_available
            inputs, targets = inputs.to(device), targets.to(device)
            # #不知道有什么用先加上
            # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            
            # 模型的预测
            outputs = model(inputs)
            # 计算loss
            loss = criterion(outputs, targets)
            

            # 计算acc和变量更新
            prec1, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # 在进行反向传播之前，我们使用zero_grad方法清空梯度
            optimizer.zero_grad()  #加上set_to_none=True而不是留空据说能加快速度
            # backward pass:
            loss.backward()
            # perform as single optimization step (parameter update)
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            ## 把主要的参数打包放进bar中
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}| top5: {top5: .4f}'.format(
                batch=batch_index + 1,
                size=len(train_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg, top5.avg)

def test_model(val_loader, model, criterion, device, test=None):

  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  predict_all = np.array([], dtype=int)
  labels_all = np.array([], dtype=int)

  model.eval()
  end = time.time()

  # 训练每批数据，然后进行模型的训练
  ## 定义bar 变量
  bar = Bar('Processing', max=len(val_loader))
  for batch_index, (inputs, targets) in enumerate(val_loader):
      data_time.update(time.time() - end)
      # move tensors to GPU if cuda is_available
      inputs, targets = inputs.to(device), targets.to(device)
      # # 不知道有啥用先加上
      # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
      # inputs, targets = inputs.cuda(), targets.cuda()
      inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
      # 模型的预测
      with torch.no_grad():
        outputs = model(inputs)
        # 计算loss
        loss = criterion(outputs, targets)

        # 计算acc和变量更新
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 3))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # 评估混淆矩阵的数据
        targets = targets.data.cpu().numpy()  # 真实数据的y数值
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # 预测数据y数值
        labels_all = np.append(labels_all, targets)  # 数据赋值
        predict_all = np.append(predict_all, predic)

        ## 把主要的参数打包放进bar中
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}| top5: {top5: .4f}'.format(
            batch=batch_index + 1,
            size=len(val_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
  bar.finish()

  if test:
      return (losses.avg, top1.avg, top5.avg, predict_all, labels_all)
  else:
      return (losses.avg, top1.avg, top5.avg)
  
if __name__ == '__main__':
    main()