import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from operation import *
from torch.autograd import Variable
import torch.nn.functional as F
import random

O3 = [0,2]
O5 = [1,3]
OpN = [1,2]
class MMixedOp(nn.Module):
# 两个节点之间的连接的混合
    def __init__(self, C,OPS):
        super(MMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.extra_op = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        # self.extra_op = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
        #                               nn.BatchNorm2d(C, affine=False))
        self.max_avg_pool1 = AvgMaxPool(C)
        self.max_avg_pool2 = AvgMaxPool(C)
        self.auxiliary_op = OPS['skip_connect'](C, False)

        self.li = [j for j in range(4)]
        self.i = 0


        # PRIMITIVES
        for primitive in PRIMITIVES_O:
          # ops是一个操作方法的集合
            op = OPS[primitive](C, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights,V,T,S,n,E=0):
        res = sum(w * op(x) for w, op in zip(weights[:3], self._ops[:3]))
        # 返回权重与各操作的乘积和

        # 2
        # res = sum(w * op(x) for w, op in zip(weights[:3], self._ops[:3]))


        # res += self.max_avg_pool1(sum(weights[3+n] * self._ops[3](x, n) for n in O3))
        # res += self.max_avg_pool2(sum(weights[3+n] * self._ops[3](x, n) for n in O5))

        # res += sum(w * self._ops[3](x, n) for n, w in enumerate(weights[3:]))
        if T:
            # res += weights[3 + n] * self._ops[3](x, n)
            # res += weights[3+n] * self._ops[3](x, n)

            # res += weights[3+self.li[self.i]] * self._ops[3](x, self.li[self.i])
            #
            # # res += sum(weights[3:]) * self._ops[3](x, self.li[self.i])
            #
            # self.i = self.i + 1
            # if self.i == 4:
            #     self.i = 0
            #     random.shuffle(self.li)
            if S:
                t1 = 0.8 * (70 - E) / 70 + 0.1
                if t1 <= 0:
                    t1 = 0
                P1 = [1 - t1, t1]
                p1 = np.random.choice([0, 1], p=P1)
                if p1:
                # if E:
                    # res += weights[4] * self._ops[3](x, 1)
                    # t2 = 0.9 * (70 - E) / 70 + 0.1
                    P2 = [0.05, 0.95]
                    p2 = np.random.choice([0, 1], p=P2)
                    if p2:
                    # if V:
                        # res += weights[4] * self._ops[3](x, 1)
                        res += weights[3] * self._ops[3](x, 0)
                    else:
                        # res += weights[4] * self._ops[3](x, 1)
                        res += weights[3 + n] * self._ops[3](x, n)
                else:
                    # m = int(torch.argmax(weights[3:], 0).data.cpu().numpy())
                    m = int(torch.argmax(weights[3:-1], 0).data.cpu().numpy())
                    res += weights[3 + m] * self._ops[3](x, m)

                    # self.i = self.i + 1
                    # if self.i == 4:
                    #     self.i = 0
                    #     random.shuffle(self.li)
            else:
                # # res += weights[3 + n] * self._ops[3](x, n)
                t = 0.8*(70 - E)/70 + 0.1
                # t = 0.8 * (70-20 - (E-20)) / 70 + 0.1
                if t <= 0:
                    t = 0
                P = [1-t,t]
                p = np.random.choice([0, 1], p=P)
                if p:
                # if V:
                    res += weights[3 + n] * self._ops[3](x, n)
                else:
                    # m = int(torch.argmax(weights[3:], 0).data.cpu().numpy())
                    m = int(torch.argmax(weights[3:-1], 0).data.cpu().numpy())
                    res += weights[3 + m] * self._ops[3](x, m)
        else:
            # m = int(torch.argmax(weights[3:], 0).data.cpu().numpy())
            m = int(torch.argmax(weights[3:-1], 0).data.cpu().numpy())
            res += weights[3+m] * self._ops[3](x, m)

            # P_num = random.randint(0, 1)
            # if P_num:
            #     m = int(torch.argmax(weights, 0).data.cpu().numpy())
            #     res += weights[m] * self._ops[3](x, m)
            # else:
            #     res += weights[3+self.li[self.i]] * self._ops[3](x, self.li[self.i])
            #     self.i = self.i + 1
            #     if self.i == 4:
            #         self.i = 0
            #         random.shuffle(self.li)
        return res

class Cell(nn.Module):
    def __init__(self, C_prev_prev, C_prev, C, reduction, reduction_prev):
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
        self.reduction = reduction
        if reduction:
            for i in range(4):
                for j in AR[i]:
                    connect = nn.ModuleList([])
                    for num,ar in enumerate(j):
                        if ar == '1':
                            if num >= 2:
                                # O1
                                # Ops1
                                # MixedOp_yuan
                                # MixedOp_gai
                                op = MMixedOp(C,O1)
                            else:
                                op = MMixedOp(C,O2)
                            connect.append(op)
                    self.connections.append(connect)
        else:
            for i in range(4):
                for j in AR[i]:
                    connect = nn.ModuleList([])
                    op1 = MMixedOp(C, O1)
                    op2 = MMixedOp(C, O1)
                    connect.append(op1)
                    connect.append(op2)
                    self.connections.append(connect)

    def forward(self, s0, s1,encoding,weights,V,T,S,n,E=0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        # Ar_All_Model
        # Ar_Model
        if self.reduction:
            t = 1
            encoding = encoding.split(' ')[1]
            l = Ar_Model[encoding]
            # W_local = Ar_allModel_back_local[encoding]
            encoding = encoding.split('-')
        else:
            encoding = encoding.split(' ')[0]
            l = Ar_Model[encoding]
            # W_local = Ar_allModel_back_local[encoding]
            encoding = encoding.split('-')
            t = 0

        for no,i in enumerate(encoding):
            site = AR_location[i]

            op1 = self.connections[site[0]][0]
            op2 = self.connections[site[0]][1]

            # w1 = weights[t * 79 + l][no * 2]
            # w2 = weights[t * 79 + l][no * 2 + 1]
            w1 = F.softmax(weights[t*79 + l][no*2], dim=-1)
            w2 = F.softmax(weights[t*79 + l][no*2 + 1], dim=-1)

            # s = op1(states[site[1]],w1,V) + op2(states[site[2]],w2,V)
            s = op1(states[site[1]], w1, V,T,S,n,E) + op2(states[site[2]], w2, V,T,S,n,E)

            # s = sum(op1(states[site[1]],w1),op2(states[site[2]],w2))
            states.append(s)

        return torch.cat([states[i] for i in [2, 3, 4, 5]], dim=1)

class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, criterion):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._criterion = criterion

        self.li1 = [j for j in range(4)]
        self.i1 = 0
        self.li2 = [j for j in range(4)]
        self.i2 = 0
        self.c = [self._layers // 3, 2 * self._layers // 3]

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
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        num_ops = len(PRIMITIVES)
        self._arch_parameters = []
        for i in range(79*2):
            self._arch_parameters.append(Variable(1e-3 * torch.randn(8, num_ops).cuda(), requires_grad=True))

    def forward(self, input, encoding,V,T,S,E=0):
        s0 = s1 = self.stem(input)

        t1 = 0.8 * (70 - E) / 70 + 0.1
        if t1 <= 0:
            t1 = 0
        P1 = [1 - t1, t1]
        p1 = np.random.choice([0, 1], p=P1)

        # t2 = 0.8 * (70 - E) / 70 + 0.1
        t2 = 0.8 * (70-20 - (E-20)) / 70 + 0.1
        if t2 >= 1:
            t2 = 1
        if t2 <= 0:
            t2 = 0
        P2 = [1 - t2, t2]
        p2 = np.random.choice([0, 1], p=P2)

        for i, cell in enumerate(self.cells):
            # s0, s1 = s1, cell(s0, s1, encoding, self._arch_parameters, V)
            if i in self.c:
                s0, s1 = s1, cell(s0, s1, encoding, self._arch_parameters, V, T=T, S=S, n=self.li2[self.i2],E=E)
                # s0, s1 = s1, cell(s0, s1, encoding, self._arch_parameters, p2, T=T, S=S, n=self.li2[self.i2], E=p1)
            else:
                s0, s1 = s1, cell(s0, s1, encoding, self._arch_parameters, V, T=T, S=S, n=self.li1[self.i1],E=E)
                # s0, s1 = s1, cell(s0, s1, encoding, self._arch_parameters, p2, T=T, S=S, n=self.li1[self.i1], E=p1)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        self.i1 = self.i1 + 1
        if self.i1 == 4:
            self.i1 = 0
            random.shuffle(self.li1)

        self.i2 = self.i2 + 1
        if self.i2 == 4:
            self.i2 = 0
            random.shuffle(self.li2)

        return logits

    def _loss(self, input, target, encoding,V,E):
        # logits = self(input, encoding, V)
        logits = self(input, encoding,V,True,False,E)
        return self._criterion(logits, target)

    def arch_parameters(self):
        return self._arch_parameters

    def updata_arch_parameters(self,arch_parameters):
        self._arch_parameters = arch_parameters




