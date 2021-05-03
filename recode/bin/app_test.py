import torch
from flask import Flask, request
from flask_ngrok import run_with_ngrok
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from utils.json_utils import jsonify
from build_net import make_model
# from gcnet.train import GarbageClassifier
from transform import transforms_image
import time
from collections import OrderedDict
import codecs
import base64
import skimage.io
import matplotlib.pyplot as plt
import argparse

# 获取所有配置参数
app = Flask(__name__)
# 设置编码-否则返回数据中文时候-乱码
app.config['JSON_AS_ASCII'] = False  # 防止中文乱码
run_with_ngrok(app)  # Start ngrok when app is run

class_id4name = {0: '其他垃圾', 1: '厨余垃圾', 2: '可回收物', 3: '有害垃圾'}
class_id40name = {
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
# for line in codecs.open('data/garbage_label.txt', 'r', encoding='utf-8'):
#     line = line.strip()
#     _id = line.split(":")[0]
#     _name = line.split(":")[1]
#     class_id2name[int(_id)] = _name
# Parse arguments
# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--num4-classes', default=4, type=int, metavar='N',help='number of classfication of image')
# parser.add_argument('--num40-classes', default=40, type=int, metavar='N',help='number of classfication of image')
# parser.add_argument('--arch4', '-a', metavar='ARCH', default='resnext101_32x16d_wsl')
# parser.add_argument('--arch40', '-a', metavar='ARCH', default='resnext101_32x16d_wsl')




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
# device = torch.device('cpu')  # 设备

print('Pytorch garbage-classification Serving on {} ...'.format(device))
# num_classes = len(class_id2name)
model_name = 'resnext101_32x16d_wsl'
model_path = '/content/drive/Shareddrives/riverfjs/model/checkpoint/40_layer4_cbam_resize/best_checkpoint_8.pth.tar'  # args.resume # --resume checkpoint/garbage_resnext101_model_2_1111_4211.pth
model_path_4 = "/content/drive/Shareddrives/riverfjs/model/checkpoint/4_layer4_cbam_resize/best_checkpoint_9.pth.tar"
# print("model_name = ",model_name)
# print("model_path = ",model_path)

net4 = make_model(predict=True, modelname=model_name, num_classes=4)
net4.to(device)
# GCNet = GarbageClassifier(model_name, num_classes, ngpu=0, feature_extract=True)
# GCNet.model.to(device)  # 设置模型运行环境
# 如果要使用cpu环境,请指定 map_location='cpu' 
# state_dict = torch.load(model_path, map_location='cpu')['state_dict']  # state_dict=torch.load(model_path)
#modelState = torch.load(model_path, map_location='cpu')['state_dict']
"""from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in modelState.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v"""
# load params
net4.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path_4)['state_dict'].items()}) # remove `module.`

# remove `module.`
# GCNet.model.load_state_dict(new_state_dict)

#GCNet.model.load_state_dict(new_state_dict)
net4.eval()


net40 = make_model(predict=True, modelname=model_name, num_classes=40)
net40.to(device)
net40.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path)['state_dict'].items()})
net40.eval()



def base64_to_rgb(base64_str):
    """
    默认base64中的图像为rgb，直接转换成即可
    :param base64:
    :return:
    """
    if isinstance(base64_str, bytes):
        base64_str = base64_str.decode("utf-8")

    imgdata = base64.b64decode(base64_str)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img

@app.route('/')
def hello():
    return "Nice To Meet You!"

@app.route('/test', methods=['POST'])
def test():
    # if request.files['image']:

    # file = request.files['image']
    img_base64 = request.form.get("base64")  # 接受base64的图片
    img = base64_to_rgb(img_base64)
    print(type(img))
    # plt.imshow(img)
    # plt.show()

@app.route('/predict', methods=['POST'])
def predict():
    import json
    # 获取输入数据
    # file = request.files['image']
    try:
        data = json.loads(request.data)
    # img_base64 = request.args.get("base64")  # 接受base64的图片
        img = base64_to_rgb(data["base64"])
        predict_type = data["type"]
    except:
        return jsonify({"msg": "json only"})
    else:
        # img_bytes = file.read()
        # 处理图片和特征提取
        feature = transforms_image(img)
        feature = feature.to(device)  # 在device 上进行预测
        # 模型预测
        if predict_type is 0:
            with torch.no_grad():
                t1 = time.time()
                outputs = net4.forward(feature)  
                consume = (time.time() - t1) * 1000  # ms
                consume = int(consume)
                # API 结果封装
            label_c_mapping = {}
            ## The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            ## 通过softmax 获取每个label的概率
            outputs = torch.nn.functional.softmax(outputs[0], dim=0)
            # outputs = torch.max(outputs[0], dim=1)
            pred_list = outputs.cpu().numpy().tolist()

            for i, prob in enumerate(pred_list):
                label_c_mapping[int(i)] = prob
            ## 按照prob 降序，获取topK = 4
            dict_list = []
            for label_prob in sorted(label_c_mapping.items(), key=lambda x: x[1], reverse=True)[:5]:
                label = int(label_prob[0])
                result = {'label': label, 'c': label_prob[1], 'name': class_id4name[label]}
                dict_list.append(result)
            ## dict 中的数值按照顺序返回结果
            result = OrderedDict(error=0, errmsg='success', consume=consume, data=dict_list)
            return jsonify(result)
        elif predict_type is 1:
            with torch.no_grad():
                t1 = time.time()
                outputs = net40.forward(feature)  
                consume = (time.time() - t1) * 1000  # ms
                consume = int(consume)
            # API 结果封装
            label_c_mapping = {}
            ## The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            ## 通过softmax 获取每个label的概率
            outputs = torch.nn.functional.softmax(outputs[0], dim=0)
            # outputs = torch.max(outputs[0], dim=1)
            pred_list = outputs.cpu().numpy().tolist()

            for i, prob in enumerate(pred_list):
                label_c_mapping[int(i)] = prob
            ## 按照prob 降序，获取topK = 4
            dict_list = []
            for label_prob in sorted(label_c_mapping.items(), key=lambda x: x[1], reverse=True)[:5]:
                label = int(label_prob[0])
                result = {'label': label, 'c': label_prob[1], 'name': class_id40name[label]}
                dict_list.append(result)
            ## dict 中的数值按照顺序返回结果
            result = OrderedDict(error=0, errmsg='success', consume=consume, data=dict_list)
            return jsonify(result)


if __name__ == '__main__':
    # curl -X POST -F image=@cat_pic.jpeg http://localhost:5000/predict
    app.run()
