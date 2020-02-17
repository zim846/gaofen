## 数据集

**准备train数据集和val数据集：**

比如gemfield的数据集存放路径是/bigdata/gemfield/github/data/，则在这个目录下有如下结构：

```text
train/class1/*.jpg
train/class2/*.jpg
...
train/classN/*.jpg
val/class1/*.jpg
val/class2/*.jpg
...
val/classN/*.jpg
```

不止支持jpg格式

## 训练

训练的话使用如下脚本和参数：

/opt/conda/bin/python tars_train.py -idl /bigdata/gemfield/github/data/ -sl checkpoint/ -mo "resnet50" -ep 5 -b 8 -fi

分别指定了数据集路径、模型存放路径、网络的名字、epoch数量、batch size等。

```text
root@gemfield:/bigdata/gemfield/github/pytorch_classifiers# /opt/conda/bin/python tars_train.py -idl /bigdata/gemfield/github/data/ -sl checkpoint/ -mo "resnet50" -ep 5 -b 8  -fi
/opt/conda/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
[Scale: 256 , mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]]
['0', '1']
[Load the model...]
Loading model using class: 2, use_gpu: True, freeze_layers: True, freeze_initial_layers: False, name_of_model: resnet50
[Building resnet50]
[All layers will be trained]
[Building inception_v3 or Resnet]
[Using all the available GPUs]
class number:  2048
[Using CrossEntropyLoss...]
[Using small learning rate with momentum...]
[Creating Learning rate scheduler...]
[Training the model begun ....]
True 0.1
MIXUP
gemfield1:  {'train': 8816, 'val': 8816}
Epoch 0/4
----------
  0%|                                                                                                                                                               | 0/1102 [00:00<?, ?it/s]Unexpected end of /proc/mounts line `overlay / overlay rw,relatime,lowerdir=/var/lib/docker/overlay2/l/CHOHFRBED4V7NXY6LVU4CM4X5P:/var/lib/docker/overlay2/l/NORTRBV3VRDXKYG4BCBIZV7RIY:/var/lib/docker/overlay2/l/7LUGZOXWN6ZO6ADW3WDUM6O4CE:/var/lib/docker/overlay2/l/CNDFM3INQQTPRWSLPHSMHIYM65:/var/lib/docker/overlay2/l/QQSGHCW4L36SLFKBREOEPUMRBG:/var/lib/docker/overlay2/l/NNCDWYSW7GI26GVAVFSB6HDWDO:/var/lib/docker/overlay2/l/LDMKZYVVCNJALMD5MF5WYD3TOE:/var/lib/docker/overlay2/l/3XVBUMEVC4GQRB3AOJO36CJOG6:/var/lib/docker/overlay2/l/LDW27JWMGWRGU'
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1102/1102 [01:32<00:00, 12.44it/s]
train Loss: 0.0323 Acc: 0.8955
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1102/1102 [00:40<00:00, 27.89it/s]
val Loss: 0.0005 Acc: 1.0000

Epoch 1/4
----------
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1102/1102 [01:30<00:00, 12.16it/s]
train Loss: 0.0145 Acc: 0.9574
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1102/1102 [00:40<00:00, 27.19it/s]
val Loss: 0.0001 Acc: 0.9997
```



这个命令会把表现最好的epoch的参数序列化成pth文件，保存在checkpoint目录下。

## **实际测试**

实际测试一个pth的话，一般来说要经历以下步骤：

**1，初始化网络图**

使用torchvision模块中的models功能来创建resnet50网络图：

```text
import torchvision.models as models
resnet50 = models.resnet50(pretrained=False)

#修改分类的数量为2
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 2)
```

**2，加载模型**

load训练好的参数：

```text
#参数反序列化为python dict
state_dict = torch.load('checkpoint/resnet50_True_freeze_False_freeze_initial_layer.pth')

#加载训练好的参数
resnet50.load_state_dict(state_dict)

#变成测试模式，dropout和BN在训练和测试时不一样
#eval()会把模型中的每个module的self.training设置为False 
resnet50 = resnet50.cuda().eval()
```

需要注意的是，如果模型之前save的时候使用了dataparallel，那么在load的时候需要转换下字典的key，把key前面的module.去掉：

```text
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
resnet50.load_state_dict(new_state_dict)
```

**3，提供输入**

以测试单张图片为例，假设输入的图片是/bigdata/gemfield/github/data/val/1/img_0.jpg：

```python3
from torchvision import transforms
#cv2读进来是numpy类型
img = cv2.imread('/bigdata/gemfield/github/data/val/1/img_0.jpg')
img = cv2.resize(img, (224,224))
#numpy -> tensor, hwc -> chw
image = transforms.ToTensor()(img)
#Normalize
image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
#chw -> bchw, b在这里是1
image = image[None]
#放到GPU上
inputs = Variable(image.cuda())
```



**4， 前向**

走网络的前向进行predict

```text
output = resnet50(inputs)
```



**5，处理模型输出**

经过softmax，得到分类和置信度信息：

```text
outputs = torch.stack([nn.Softmax(dim=0)(i) for i in output])

outputs = outputs.mean(0)
p, preds = torch.max(outputs, 0)
# tensor -> scaler
print(p.data.cpu().item())
print(preds.data.cpu().item())
```



**6， 完整代码需要的import**

```python3
import os
import cv2
from collections import OrderedDict
import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
```