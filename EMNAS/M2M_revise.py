import secrets
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import random
import sys
from numpy.random import choice
import pandas as pd
from non_dominated_sort import *
import utils2 as utils
from operation import *


L = [i for i in range(79)]

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class M2M:
    def __init__(self,Epoch,Train_epoch,Warm_epoch,Ar_model,Ar_all,model,val_loader,S,s):
        self.HiddenPrints = HiddenPrints()
        # 子区域个体数
        self.s = s
        self.S = S
        self.f1min = 1000
        self.f2min = 1000

        self.old = None
        self.new = None

        # p是种群
        # P是分类后种群,记录p中个体的索引
        self.Ar_model = Ar_model
        self.Ar_all = Ar_all
        self.Best_model = None
        # 种群所有个体的具体信息
        self.p = []
        # 父代种群
        self.parent = []
        # 非训练种群
        self.no_Current_Ar = []
        # 种群后代
        self.off = None
        # 当前种群与后代的合并
        self.Q = None
        # k是权重个数
        self.k = 0
        self.val_loader = val_loader
        # 巢网络
        self.Model = model
        self.last_arch_parameters = model._arch_parameters
        self.Epoch = Epoch
        self.Train_epoch = Train_epoch
        self.Warm_epoch = Warm_epoch
        self.step = 0
        self.criterion = nn.CrossEntropyLoss().to('cuda')
        self.optimizer = torch.optim.SGD(self.Model.parameters(), 0.025, 0.9, 3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,Train_epoch,eta_min=0.001)
        # 当前代的非支配解
        self.non_dominated = []
        # 设置权重向量组
        self.set_W()
        # 设置验算模型mac与参数的数据
        self.set_val_MacPrms()
        # 各领域的操作
        self.operation = operation()
        self.no_Current_Ar = []
        tr = []
        for i in self.p:
            tr.append(i[0])
        for i in self.Ar_all:
            if i not in tr:
                self.no_Current_Ar.append(i)

    def set_W(self):
        self.W2 = np.loadtxt('/dim2.csv')
        self.W4 = np.loadtxt('/dim4.csv')
        self.W9 = np.loadtxt('/dim9.csv')
        # self.k = len(self.W)

    # 设置检查模型macs与参数量的input
    def set_val_MacPrms(self):
        t = torch.randn(1, 3, 32, 32).to('cuda')
        self.val_MacPrms = t

    def update_model(self,new):
        self.Ar_model = new

    def allocation(self,P,f1,f2,limit):
        A = [[],[],[],[],[],[],[],[],[]]
        print(limit, self.f1min, f1, self.f2min, f2)
        # 对所有个体进行分区域
        for p in P:
            # 提取三个目标值
            l = [p[1]-f1,p[2]-f2]
            ma = -1
            t = -1
            # 计算内积得知夹角，从而将个体分配到i类中
            for i, w in enumerate(self.W9):
                dot = np.dot(l, w)
                if dot > ma:
                    ma = dot
                    t = i
            A[t].append(p)
        print(len(A[0]), len(A[1]), len(A[2]),len(A[3]),len(A[4]),len(A[5]),len(A[6]),len(A[7]),len(A[8]))

        p_next = []
        num = 0
        N = []

        for i in A[:5]:
            n = non_dominated_sort(i)
            N.append(n)
        for k in range(2):
            for i in N:
                if (len(i)-2) < k:
                    continue
                l = len(i[k])
                if (l + num) <= self.S:
                    p_next = p_next + i[k]
                    num = num + l
                else:
                    if num == self.S:
                        break
                    t = self.S - num
                    b = int(1.5 * t)
                    x = sorted(i[k], key=lambda stu: stu[1])
                    if l >= b:
                        x = x[:b]
                    d = crowding_distance_assignment(x, t)
                    p_next = p_next + d
                    num = num + t

                if num == self.S:
                    break
            if num == self.S:
                break

        AA = []
        for i in A:
            a = []
            if len(i) >0:
                for j in i:
                    if j not in p_next:
                        a.append(j)
            AA.append(a)

        N = []
        ml = 0
        for i in AA:
            n = non_dominated_sort(i)
            N.append(n)
            if len(n) > ml:
                ml = len(n)
        for k in range(ml):
            for i in N:
                if (len(i) - 2) < k:
                    continue
                l = len(i[k])
                if (l + num) <= self.S:
                    p_next = p_next + i[k]
                    num = num + l
                else:
                    if num == self.S:
                        break
                    t = self.S - num
                    b = int(1.5 * t)
                    x = sorted(i[k], key=lambda stu: stu[1])
                    if l >= b:
                        x = x[:b]
                    d = crowding_distance_assignment(x, t)
                    p_next = p_next + d
                    num = num + t

                if num == self.S:
                    break
            if num == self.S:
                break

        self.p = p_next
        # print(p_next)
        parent = sorted(p_next, key=lambda stu: stu[2])[:20]
        # print(parent)
        self.parent = parent

        self.no_Current_Ar = []
        tr = []
        for i in self.p:
            tr.append(i[0])
        for i in self.Ar_all:
            if i not in tr:
                self.no_Current_Ar.append(i)

    def NonD(self,P,f1,f2,limit):
        p_next = []
        num = 0
        N = []

        n = non_dominated_sort(P)
        for i in n:
            l = len(i)
            if (l + num) <= self.S:
                p_next = p_next + i
                num = num + l
            else:
                if num == self.S:
                    break
                t = self.S - num
                if t <= 0:
                    t = 0
                b = int(1.5 * t)
                x = sorted(i, key=lambda stu: stu[1])
                if l >= b:
                    x = x[:b]
                d = crowding_distance_assignment(x, t)
                p_next = p_next + d
                num = num + t
            if num == self.S:
                break

        self.p = p_next
        # print(p_next)
        parent = sorted(p_next, key=lambda stu: stu[2])[:20]
        # print(parent)
        self.parent = parent

        self.no_Current_Ar = []
        tr = []
        for i in self.p:
            tr.append(i[0])
        for i in self.Ar_all:
            if i not in tr:
                self.no_Current_Ar.append(i)

    def offspring(self, epoch, arch_parameters,Up,Low):
        en = []
        q = []
        N = []
        not_parent = []
        p = []
        Ul = Up.cuda() - Low.cuda()
        if self.step < 9:
            self.step += 1

        parent = []
        Bn = self.Best_model.split(' ')[0]

        # Ar_All_Model
        # Ar_Model
        Bn = Ar_Model[Bn]
        # print(self.parent)
        for i in self.parent:
            parent.append(i[0])
            en.append([i[0], 0])


        for i in self.p:
            p.append(i[0])
            n = i[0].split(' ')[0]
            if n not in N:
                N.append(n)
        for i in p:
            if i not in parent:
                not_parent.append(i)

        print(len(not_parent), len(parent))

        for i in not_parent:
            n1, r1 = i.split(' ')
            p2 = secrets.choice(parent)
            n2, r2 = p2.split(' ')

            # r通过交叉变异获取
            r = self.operation.offspring(r1, r2)
            Ar = n1 + ' ' + r

            en.append([Ar, 0])
            q.append(Ar)

            # 与父母进行交叉
            l1 = Ar_Model[n1]
            l2 = Ar_Model[n2]
            # a = random.sample([0, 1, 2, 3, 4, 5, 6, 7], 2)
            P_num = random.randint(0, 8)
            k = random.sample([0, 1, 2, 3, 4, 5, 6, 7], P_num)
            with torch.no_grad():
                for n_a in k:
                    s = random.sample([-1, 1], 1)
                    num = torch.rand(7).cuda()
                    # num = torch.rand(8).cuda()
                    # arch_parameters[l1][n_a] = arch_parameters[l2][n_a] + s[0] * (1 - num ** (
                    #         1 - (epoch / 50) ** 0.7)) * (arch_parameters[l2][n_a] - arch_parameters[l1][n_a])

                    # arch_parameters[l1][n_a] = arch_parameters[l1][n_a] + s[0] * (1 - num ** (
                    #         1 - (self.step / 10) ** 0.7)) * (
                    #         arch_parameters[l2][n_a] - arch_parameters[l1][n_a])

                    arch_parameters[l1][n_a] = arch_parameters[l1][n_a] + s[0] * (1 - num ** (
                            1 - ((epoch - self.Warm_epoch) / (self.Epoch - self.Warm_epoch)) ** 0.7)) * (
                                                       arch_parameters[l2][n_a] - arch_parameters[l1][n_a])

        random.shuffle(L)
        for i in L:
            if len(en) < 2*self.S:
                n = Ar_allModel[i]
                if n not in N:
                    r = secrets.choice(Ar_allModel)
                    Ar = n + ' ' + r
                    p2 = secrets.choice(parent)
                    n2, r2 = p2.split(' ')

                    en.append([Ar, 0])
                    q.append(Ar)
                    N.append(n)

                    # 与父母进行交叉
                    l1 = Ar_Model[n]
                    l2 = Ar_Model[n2]
                    P_num = random.randint(0, 8)
                    k = random.sample([0, 1, 2, 3, 4, 5, 6, 7], P_num)
                    with torch.no_grad():
                        for n_a in k:
                            s = random.sample([-1, 1], 1)
                            num = torch.rand(7).cuda()
                            # num = torch.rand(8).cuda()
                            # arch_parameters[l1][n_a] = arch_parameters[l2][n_a] + s[0] * (1 - num ** (
                            #         1 - (epoch / 50) ** 0.7)) * (arch_parameters[l2][n_a] - arch_parameters[l1][n_a])

                            # arch_parameters[l1][n_a] = arch_parameters[l1][n_a] + s[0] * (1 - num ** (
                            #         1 - (self.step / 10) ** 0.7)) * (
                            #                                    arch_parameters[l2][n_a] - arch_parameters[l1][n_a])

                            arch_parameters[l1][n_a] = arch_parameters[l1][n_a] + s[0] * (1 - num ** (
                                    1 - ((epoch - self.Warm_epoch) / (self.Epoch - self.Warm_epoch)) ** 0.7)) * (
                                                               arch_parameters[l2][n_a] - arch_parameters[l1][n_a])
        for i in q:
            a = secrets.randbelow(100)
            if a < 30:
                n, r = i.split(' ')
                l1 = Ar_Model[n]
                # a = random.sample([0, 1, 2, 3, 4, 5, 6, 7], 2)
                P_num = random.randint(0, 8)
                k = random.sample([0, 1, 2, 3, 4, 5, 6, 7], P_num)
                with torch.no_grad():
                    for n_a in k:
                        s = random.sample([-1, 1], 1)
                        num = torch.rand(7).cuda()
                        # num = torch.rand(8).cuda()
                        # if s[0] > 0:
                        #     Ul = Up.cuda()
                        # else:
                        #     Ul = Low.cuda()
                        arch_parameters[l1][n_a] = arch_parameters[l1][n_a] + s[0]*0.15 * (1 - num ** (
                                1 - (self.step / 10) ** 0.7)) * (Ul - arch_parameters[l1][n_a])

        return en, q,arch_parameters

    def arch(self,epoch, arch_parameters,Up,Low):
        Bn = self.Best_model.split(' ')[0]
        # Ar_All_Model
        # Ar_Model
        Bn = Ar_Model[Bn]
        # for i in self.parent:
        #     parent.append(i[0])
        Ul = Up.cuda() - Low.cuda()

        for i in self.p:
            n = i[0].split(' ')[0]
            p2 = secrets.choice(self.parent)[0]
            n2, _ = p2.split(' ')
            l2 = Ar_Model[n2]

            # 与父母进行交叉
            l1 = Ar_Model[n]
            P_num = random.randint(0, 8)
            k = random.sample([0, 1, 2, 3, 4, 5, 6, 7], P_num)
            with torch.no_grad():
                for n_a in k:
                    s = random.sample([-1, 1], 1)
                    num = torch.rand(7).cuda()
                    # num = torch.rand(8).cuda()
                    arch_parameters[l1][n_a] = arch_parameters[l1][n_a] + s[0] * (1 - num ** (
                            1 - (9 / 10) ** 0.7)) * (arch_parameters[l2][n_a] - arch_parameters[l1][n_a])

            a = secrets.randbelow(100)
            if a < 30:
                # 变异
                l1 = Ar_Model[n]
                P_num = random.randint(0, 8)
                k = random.sample([0, 1, 2, 3, 4, 5, 6, 7], P_num)
                with torch.no_grad():
                    for n_a in k:
                        s = random.sample([-1, 1], 1)
                        num = torch.rand(7).cuda()
                        # num = torch.rand(8).cuda()
                        # if s[0] > 0:
                        #     Ul = Up.cuda()
                        # else:
                        #     Ul = Low.cuda()
                        arch_parameters[l1][n_a] = arch_parameters[l1][n_a] + s[0]*0.15 * (1 - num ** (
                                1 - (9 / 10) ** 0.7)) * (Ul - arch_parameters[l1][n_a])
        return arch_parameters
