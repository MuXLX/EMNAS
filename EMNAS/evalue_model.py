import torch
import torch.nn as nn
from operation import *
from torch.autograd import Variable
from utils2 import drop_path

class Cell(nn.Module):
    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)
        self.C = C
        self.affine = True
        self.multiplier = 4
        self.reduction = reduction
        # 对输入做预处理，
        if reduction_prev:
            # 如果上一个是reduction，则需要将上上个的输入进行特征图减半，以与上一个输入同大小
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            # 若是普通的cell，则以1*1卷积进行维度转换
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self.connections = nn.ModuleList([])


        if reduction:
            self.stride = 2
            for i in Ar:
                connect = nn.ModuleList([])
                for no, op in enumerate(i):
                    if op != '0':
                        c = nn.ModuleList([])
                        if no > 1:
                            for o in ops1:
                                c.append(ops1[o](self.C, True))
                        else:
                            for o in ops2:
                                c.append(ops2[o](self.C, True))
                        connect.append(c)
                self.connections.append(connect)
        else:
            self.stride = 1
            for i in Ar:
                connect = nn.ModuleList([])
                for no, op in enumerate(i):
                    if op != '0':
                        c = nn.ModuleList([])
                        for o in ops1:
                            c.append(ops1[o](self.C, True))
                        connect.append(c)
                self.connections.append(connect)


    def forward(self, s0, s1,drop_prob,encoding):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # 输入
        states = [s0, s1]

        l = encoding.split('-')
        for node,i in enumerate(l):
            en = ''
            a = []
            for j in i:
                if j != '0':
                    en = en + '1'
                    a.append(int(j))
                else:
                    en = en + '0'

            location = AR_location[en]
            connect = self.connections[location[0]]

            h1 = states[location[1]]
            h2 = states[location[2]]
            # 当前的操作
            op1 = connect[0][a[0]-1]
            op2 = connect[1][a[1]-1]

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

        return torch.cat([states[i] for i in [2,3,4,5]], dim=1)


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


class evl_model(nn.Module):
    def __init__(self, C, num_classes, layers,auxiliary):
        super(evl_model, self).__init__()
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
            cell = Cell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            # cell = Cell2(encoding, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, inputs, encoding):
        logits_aux = None
        encoding = encoding.split(' ')
        t = 0
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                t = t + 1

            s0, s1 = s1, cell(s0, s1,self.drop_path_prob,encoding[t])

            if i in [self._layers // 3, 2 * self._layers // 3]:
                t = t + 1

            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits,logits_aux


class Cell_yuan(nn.Module):
    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell_yuan, self).__init__()
        # print(C_prev_prev, C_prev, C)
        self.C = C
        self.affine = True
        self.multiplier = 4
        self.reduction = reduction
        # 对输入做预处理，
        if reduction_prev:
            # 如果上一个是reduction，则需要将上上个的输入进行特征图减半，以与上一个输入同大小
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            # 若是普通的cell，则以1*1卷积进行维度转换
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self.connections = nn.ModuleList([])


        if reduction:
            self.stride = 2
            # 如果当前cell是reduction，则
            for i in Ar:
                connect = nn.ModuleList([])
                for no, op in enumerate(i):
                    if op != '0':
                        c = nn.ModuleList([])
                        if no > 1:
                            for o in ops1:
                                c.append(ops1[o](self.C, True))
                        else:
                            for o in ops2:
                                c.append(ops2[o](self.C, True))
                        connect.append(c)
                self.connections.append(connect)
        else:
            self.stride = 1
            # 如果当前cell是reduction，则
            for i in Ar:
                connect = nn.ModuleList([])
                for no, op in enumerate(i):
                    if op != '0':
                        c = nn.ModuleList([])
                        for o in ops1:
                            c.append(ops1[o](self.C, True))
                        connect.append(c)
                self.connections.append(connect)


    def forward(self, s0, s1,drop_prob,encoding):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # 输入
        states = [s0, s1]
        if self.reduction:
            l = encoding.split(' ')[1].split('-')
        else:
            l = encoding.split(' ')[0].split('-')
        for node,i in enumerate(l):
            en = ''
            a = []
            for j in i:
                if j != '0':
                    en = en + '1'
                    a.append(int(j))
                else:
                    en = en + '0'

            location = AR_location[en]
            connect = self.connections[location[0]]

            h1 = states[location[1]]
            h2 = states[location[2]]
            # 当前的操作
            op1 = connect[0][a[0]-1]
            op2 = connect[1][a[1]-1]

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

        return torch.cat([states[i] for i in [2,3,4,5]], dim=1)


class evl_model_yuan(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary):
        super(evl_model_yuan, self).__init__()
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
            cell = Cell_yuan(C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            # cell = Cell2(encoding, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, inputs, encoding):
        logits_aux = None
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob, encoding)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux