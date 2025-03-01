import torch
import torch.nn as nn
from operation import *
from torch.autograd import Variable
from utils2 import drop_path

class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
          nn.ReLU(inplace=True),
          nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
          nn.Conv2d(C, 128, 1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 768, 2, bias=False),
          nn.BatchNorm2d(768),
          nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x

class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class Cell(nn.Module):
    def __init__(self, encoding, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)
        self.C = C
        self.affine = True
        self.multiplier = 4
        # 对输入做预处理，
        if reduction_prev:
            # 如果上一个是reduction，则需要将上上个的输入进行特征图减半，以与上一个输入同大小
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            # 若是普通的cell，则以1*1卷积进行维度转换
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self.connections = nn.ModuleList([])

        self.Ce = []

        if reduction:
            self.encoding = encoding.split(' ')[1]
            self.stride = 2
            # 如果当前cell是reduction，则
            l = self.encoding.split('-')
            for node, i in enumerate(l):
                connect = nn.ModuleList([])
                re = []
                for no, op in enumerate(i):
                    if op == '0':
                        continue
                    else:
                        if no >1:
                            connect.append(ops1[op](self.C, True))
                            re.append(no)
                        else:
                            connect.append(ops2[op](self.C, True))
                            re.append(no)
                self.Ce.append(re)
                self.connections.append(connect)
        else:
            self.encoding = encoding.split(' ')[0]
            self.stride = 1
            l = self.encoding.split('-')
            for node, i in enumerate(l):
                connect = nn.ModuleList([])
                Nor = []
                for no, op in enumerate(i):
                    if op == '0':
                        continue
                    else:
                        connect.append(ops1[op](self.C, True))
                        Nor.append(no)

                self.Ce.append(Nor)
                self.connections.append(connect)

    def forward(self, s0, s1,drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        # 输入
        states = [s0, s1]
        for num,i in enumerate(self.Ce):
            h1 = states[i[0]]
            h2 = states[i[1]]
            # 当前的操作
            op1 = self.connections[num][0]
            op2 = self.connections[num][1]
            # 应用操作获得下一个节点输出
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]

        return torch.cat([states[i] for i in [2, 3, 4, 5]], dim=1)

class NetworkCIFAR(nn.Module):


    def __init__(self, C, num_classes, layers,auxiliary, encoding):
        super(NetworkCIFAR, self).__init__()
        self.drop_path_prob = 0
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(encoding, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1,self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits,logits_aux

class NetworkImageNet(nn.Module):


    def __init__(self, C, num_classes, layers,auxiliary, encoding):
        super(NetworkImageNet, self).__init__()
        self.drop_path_prob = 0
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )


        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(encoding, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1,self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits,logits_aux