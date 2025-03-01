import torch
import torch.nn as nn
import secrets
from MixKernel import Mix_Kernel
from numpy.random import choice

AR = [['11'],
      ['110','101','011'],
      ['1100','1010','0110','1001','0101','0011'],
      ['11000','10100','01100','10010','01010','00110','10001','01001','00101','00011']]

Ar = ['11','110','101','011','1100','1010','0110','1001','0101','0011','11000','10100',
      '01100','10010','01010','00110','10001','01001','00101','00011']



A = {
    # 1
    '11-110-1100-11000':['11-110-1100-11000'],
    # 2
    '11-101-1100-11000':['11-101-1100-11000','11-110-1010-11000','11-110-1100-10100',
                         '11-110-1001-11000','11-110-1100-10010','11-110-1100-10001'],
    # 3
    '11-011-1100-11000':['11-011-1100-11000','11-110-0110-11000','11-110-1100-01100',
                        '11-110-0101-11000','11-110-1100-01010','11-110-1100-01001'],
    # 4
    '11-110-0011-11000':['11-110-0011-11000','11-110-1100-00110','11-110-1100-00101',
                        '11-110-1100-00011'],
    # 5
    '11-101-1010-11000':['11-110-1010-10100','11-110-1001-10010','11-101-1010-11000',
                        '11-101-1100-10100'],
    # 6
    '11-101-1100-10001':['11-110-1001-10100','11-110-1010-10010','11-101-1100-10001'],
    # 7
    '11-101-1001-11000':['11-110-1010-10001','11-110-1001-10001','11-101-1001-11000',
                        '11-101-1100-10010'],
    # 8
    '11-101-0110-11000':['11-011-1010-11000','11-011-1100-10100','11-101-0110-11000',
                        '11-101-1100-01100','11-110-0110-10100','11-110-1010-01100',
                        '11-110-0101-10010','11-110-1001-01010'],
    # 9
    '11-101-1100-01001':['11-110-0101-10100','11-110-1010-01010','11-110-0110-10010',
                        '11-110-1001-01100','11-101-1100-01001','11-011-1100-10001'],
    # 10
    '11-101-0101-11000':['11-110-1010-01001','11-110-1001-01001','11-101-0101-11000',
                        '11-101-1100-01010'],

    #################### 11
    '11-110-0011-10100':['11-110-0011-10100','11-110-1010-00110','11-110-0011-10010',
                        '11-110-1001-00110','11-101-1100-00101'],

    # 12
    '11-101-0011-11000':['11-110-1010-00101','11-110-1001-00011','11-101-0011-11000',
                        '11-101-1100-00110'],
    # 13
    '11-101-1100-00011':['11-110-1010-00011','11-110-1001-00101','11-101-1100-00011'],
    # 15
    '11-011-0101-11000':['11-110-0110-01001','11-110-0101-01001','11-011-0101-11000',
                        '11-011-1100-01010'],
    # 16
    '11-011-1001-11000':['11-011-1001-11000','11-011-1100-10010','11-110-0101-10001',
                         '11-110-0110-10001'],
    # 17
    '11-011-0110-11000':['11-011-0110-11000','11-011-1100-01100','11-110-0101-01010',
                         '11-110-0110-01100'],
    # 18
    '11-011-1100-01001':['11-011-1100-01001','11-110-0101-01100','11-110-0110-01010'],

    ########################### 19
    '11-110-0011-01100':['11-011-1100-00101','11-110-0011-01010','11-110-0011-01100',
                         '11-110-0101-00110','11-110-0110-00110'],

    # 20
    '11-011-0011-11000':['11-011-0011-11000','11-011-1100-00110','11-110-0101-00011',
                         '11-110-0110-00101'],
    # 21
    '11-011-1100-00011':['11-011-1100-00011','11-110-0101-00101','11-110-0110-00011'],


    # 22
    '11-110-0011-10001':['11-110-0011-10001'],
    # 23
    '11-110-0011-01001':['11-110-0011-01001'],
    # 24
    '11-110-0011-00110': ['11-110-0011-00110'],

    ########################### 25
    '11-110-0011-00101': ['11-110-0011-00101', '11-110-0011-00011'],

    # 26
    '11-101-1010-10100': ['11-101-1010-10100'],
    # 27
    '11-101-1001-10100': ['11-101-1001-10100', '11-101-1010-10001', '11-101-1010-10010'],
    # 28
    '11-101-1010-01100': ['11-101-0110-10100', '11-011-1010-10100', '11-101-1010-01100'],
    # 29
    '11-101-0101-10100': ['11-101-0101-10100', '11-101-1010-01001', '11-101-1010-01010'],
    # 30
    '11-101-0011-10100': ['11-101-0011-10100', '11-101-1010-00101', '11-101-1010-00110'],
    # 31
    '11-101-1010-00011': ['11-101-1010-00011'],
    # 32
    '11-101-1001-10010': ['11-101-1001-10010'],
    # 33
    '11-101-1001-10001': ['11-101-1001-10001'],
    # 34
    '11-101-0101-10010': ['11-101-1001-01010','11-101-0101-10010'],
    # 35
    '11-101-1001-01001': ['11-101-1001-01001'],
    # 36
    '11-101-1001-00110': ['11-101-1001-00110', '11-101-0011-10010'],
    # 37
    '11-101-1001-00101': ['11-101-1001-00101'],
    # 38
    '11-101-1001-00011': ['11-101-1001-00011'],
    # 39
    '11-101-1001-01100': ['11-101-1001-01100', '11-011-1010-10001', '11-101-0110-10010'],
    # 40
    '11-011-1001-10100': ['11-011-1001-10100', '11-011-1010-10010', '11-101-0110-10001'],
    # 41
    '11-101-0110-01100': ['11-101-0110-01100', '11-011-0110-10100', '11-011-1010-01100'],
    # 42
    '11-101-0101-01100': ['11-101-0101-01100', '11-011-1010-01001', '11-101-0110-01010'],
    # 43
    '11-011-0101-10100': ['11-011-0101-10100', '11-011-1010-01010', '11-101-0110-01001'],

    # 44
    '11-101-0011-01100': ['11-101-0011-01100', '11-011-1010-00101', '11-101-0110-00110'],


    # 45
    '11-011-0011-10100': ['11-011-0011-10100', '11-011-1010-00110', '11-101-0110-00101'],


    # 48
    '11-101-0110-00011': ['11-101-0110-00011', '11-011-1010-00011'],
    # 49
    '11-101-0101-10001': ['11-101-0101-10001'],
    # 50
    '11-101-0101-01010': ['11-101-0101-01010'],
    # 51
    '11-101-0101-01001': ['11-101-0101-01001'],
    # 52
    '11-101-0101-00110': ['11-101-0101-00110', '11-101-0011-01010'],
    # 53
    '11-101-0101-00101': ['11-101-0101-00101'],
    # 54
    '11-101-0101-00011': ['11-101-0101-00011'],
    # 55
    '11-101-0011-10001': ['11-101-0011-10001'],
    # 56
    '11-101-0011-01001': ['11-101-0011-01001'],
    # 57
    '11-101-0011-00110': ['11-101-0011-00110'],
    # 58
    '11-101-0011-00101': ['11-101-0011-00101'],
    # 59
    '11-101-0011-00011': ['11-101-0011-00011'],
    # 60
    '11-011-1001-10010': ['11-011-1001-10010'],
    # 61
    '11-011-1001-10001': ['11-011-1001-10001'],
    # 62
    '11-011-1001-01010': ['11-011-1001-01010', '11-011-0101-10010'],
    # 63
    '11-011-1001-01001': ['11-011-1001-01001'],
    # 64
    '11-011-1001-00110': ['11-011-1001-00110', '11-011-0011-10010'],
    # 65
    '11-011-1001-00101': ['11-011-1001-00101'],
    # 66
    '11-011-1001-00011': ['11-011-1001-00011'],
    # 67
    '11-011-1001-01100': ['11-011-1001-01100', '11-011-0110-10001','11-011-0110-10010'],
    # 68
    '11-011-0110-01100': ['11-011-0110-01100'],
    # 69
    '11-011-0101-01100': ['11-011-0101-01100', '11-011-0110-01001', '11-011-0110-01010'],
    # 71
    '11-011-0011-01100': ['11-011-0011-01100', '11-011-0110-00101', '11-011-0110-00110'],
    # 72
    '11-011-0110-00011': ['11-011-0110-00011'],
    # 73
    '11-011-0101-10001': ['11-011-0101-10001'],
    # 74
    '11-011-0101-01010': ['11-011-0101-01010'],
    # 75
    '11-011-0101-01001': ['11-011-0101-01001'],
    # 76
    '11-011-0101-00110': ['11-011-0101-00110', '11-011-0011-01010'],
    # 77
    '11-011-0101-00101': ['11-011-0101-00101'],
    # 78
    '11-011-0101-00011': ['11-011-0101-00011'],
    # 79
    '11-011-0011-10001': ['11-011-0011-10001'],
    # 80
    '11-011-0011-01001': ['11-011-0011-01001'],
    # 81
    '11-011-0011-00110': ['11-011-0011-00110'],
    # 82
    '11-011-0011-00101': ['11-011-0011-00101'],
    # 83
    '11-011-0011-00011': ['11-011-0011-00011']
}
Process = {}
for i in A:
    for j in A[i]:
        Process[j] = i
# print(Process)

AR_location = {}

a = 0
for i,li in enumerate(AR):
    for j,ar in enumerate(li):
        AR_location[ar] = [a]
        a = a + 1
        for no,k in enumerate(ar):
            if k == '1':
                AR_location[ar].append(no)

Ar_All = []
Ar_All_Model = {}
for j,ar1 in enumerate(AR[1]):
    for k,ar2 in enumerate(AR[2]):
        for z,ar3 in enumerate(AR[3]):
            ar = ['11',ar1,ar2,ar3]
            Ar_All.append('-'.join(ar))
for n,i in enumerate(Ar_All):
    Ar_All_Model[i] = n

Ar_allModel = []
Ar_split = []
Ar_Model = {}

for n,i in enumerate(A):
    Ar_allModel.append(i)
    Ar_Model[i] = n


PRIMITIVES = [
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

PRIMITIVES_O = [
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3',
    'Mix_conv',
]

O1 = {
   'skip_connect' : lambda C, affine: Identity(),
   'avg_pool_3x3' : lambda C, affine: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
   'max_pool_3x3' : lambda C, affine: nn.MaxPool2d(3, stride=1, padding=1),
   'Mix_conv' : lambda C, affine: Mix_Kernel(C, C, 1,affine=affine),
}

O2 = {
   'skip_connect' : lambda C, affine: FactorizedReduce(C, C, affine=affine),
   'avg_pool_3x3' : lambda C, affine: nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
   'max_pool_3x3' : lambda C, affine: nn.MaxPool2d(3, stride=2, padding=1),
   'Mix_conv' : lambda C, affine: Mix_Kernel(C, C, 2,affine=affine),
}

o1 = {
   '1' : lambda C, affine: Identity(),
   '2' : lambda C, affine: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
   '3' : lambda C, affine: nn.MaxPool2d(3, stride=1, padding=1),
   '4' : lambda C, affine: Mix_Kernel(C, C, 1,affine=affine),
}

o2 = {
   '1' : lambda C, affine: FactorizedReduce(C, C, affine=affine),
   '2' : lambda C, affine: nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
   '3' : lambda C, affine: nn.MaxPool2d(3, stride=2, padding=1),
   '4' : lambda C, affine: Mix_Kernel(C, C, 2,affine=affine),
}



Ops1 = {
   'skip_connect' : lambda C, affine: Identity(),
   'avg_pool_3x3' : lambda C, affine: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
   'max_pool_3x3' : lambda C, affine: nn.MaxPool2d(3, stride=1, padding=1),
   'sep_conv_3x3' : lambda C, affine: SepConv(C, C, 3, 1, 1, affine=affine),
   'sep_conv_5x5' : lambda C, affine: SepConv(C, C, 5, 1, 2, affine=affine),
   'dil_conv_3x3' : lambda C, affine: DilConv(C, C, 3, 1, 2, 2, affine=affine),
   'dil_conv_5x5' : lambda C, affine: DilConv(C, C, 5, 1, 4, 2, affine=affine),
}


Ops2 = {
    'skip_connect' : lambda C, affine: FactorizedReduce(C, C, affine=affine),
    'avg_pool_3x3' : lambda C, affine: nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
    'max_pool_3x3' : lambda C, affine: nn.MaxPool2d(3, stride=2, padding=1),
    'sep_conv_3x3' : lambda C, affine: SepConv(C, C, 3, 2, 1, affine=affine),
    'sep_conv_5x5' : lambda C, affine: SepConv(C, C, 5, 2, 2, affine=affine),
    'dil_conv_3x3' : lambda C, affine: DilConv(C, C, 3, 2, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, affine: DilConv(C, C, 5, 2, 4, 2, affine=affine),
}

ops1 = {
    '1' : lambda C, affine: Identity(),
    '2' : lambda C, affine: nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
    '3' : lambda C, affine: nn.MaxPool2d(3, stride=1, padding=1),
    '4' : lambda C, affine: SepConv(C, C, 3, 1, 1, affine=affine),
    '5' : lambda C, affine: SepConv(C, C, 5, 1, 2, affine=affine),
    '6' : lambda C, affine: DilConv(C, C, 3, 1, 2, 2, affine=affine),
    '7' : lambda C, affine: DilConv(C, C, 5, 1, 4, 2, affine=affine),
}


ops2 = {
    '1': lambda C, affine: FactorizedReduce(C, C, affine=affine),
    '2' : lambda C, affine: nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
    '3' : lambda C, affine: nn.MaxPool2d(3, stride=2, padding=1),
    '4' : lambda C, affine: SepConv(C, C, 3, 2, 1, affine=affine),
    '5' : lambda C, affine: SepConv(C, C, 5, 2, 2, affine=affine),
    '6' : lambda C, affine: DilConv(C, C, 3, 2, 2, 2, affine=affine),
    '7' : lambda C, affine: DilConv(C, C, 5, 2, 4, 2, affine=affine),
}


class operation:
    def __init__(self):
        self.a = 30

    def offspring(self,C1,C2):
        C1 = C1.split('-')
        C2 = C2.split('-')
        # 取某一个部分进行交叉变异
        encoding = C1
        k = secrets.choice([1, 2, 3])
        encoding[k] = C2[k]

        kk = secrets.randbelow(100)
        if kk < self.a:
            k = secrets.choice([1, 2, 3])
            o = secrets.SystemRandom().sample(AR[k], 1)[0]
            encoding[k] = o

        encoding = '-'.join(encoding)

        return Process[encoding]
        # return encoding


    def cross_mutation(self,x1,x2):
        # x1与x2为不同的模型编码
        X1 = x1.split(' ')

        X2 = x2.split(' ')

        model = []

        for n,i in enumerate(X1):
            model.append(self.offspring(i, X2[n]))

        return ' '.join(model),model

# operation = operation()
# print(operation.offspring('11-110-0110-00101','11-011-0110-10010'))


class AvgMaxPool(nn.Module):
    def __init__(self, C,kernel_size=3, stride=1):
        super(AvgMaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,padding=1, count_include_pad=False)
        self.bn = nn.BatchNorm2d(C, affine=False)

    def forward(self, x):
        # 对输入图像进行最大池化和平均池化，并将结果相加
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        x = x1 + x2
        x = self.bn(x)
        return x

class Rectangle_Conv(nn.Module):
  # 正常的卷积操作
    def __init__(self, C_in, C_out, stride, affine=True):
        super(Rectangle_Conv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(C_in, C_out, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):
  # 正常的卷积操作
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
          nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

# 'dil_conv_3x3' : lambda C, affine: DilConv(C, C, 3, 1, 2, 2, affine=affine),
#  'dil_conv_5x5' : lambda C, affine: DilConv(C, C, 5, 1, 4, 2, affine=affine),
class DilConv(nn.Module):
    # 扩张深度可分离卷积，但由于步长为2，又维持了特征大小
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
          nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_out, affine=affine),
          )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
  # 两轮深度可分离卷积，维持特征大小
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
          nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_in, affine=affine),
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
          nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_out, affine=affine),
          )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
  # 跳跃连接
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
          return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
  # 步长为2，表示为特征图缩小一半
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out
