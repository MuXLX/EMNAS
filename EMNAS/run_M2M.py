# os.environ['CUDA_VISIBLE_DEVICES']='1'
import os
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

from Network import NetworkCIFAR as NetworkCIFAR
# from Network import NetworkCIFAR2 as NetworkCIFAR
# from Network import NetworkCIFAR_Tiny as NetworkCIFAR_Tiny


from evalue_model import evl_model_yuan as evl_model
from operation import *
import M2M_revise
import torch.nn.functional as F
from non_dominated_sort import *
import argparse
from thop import profile
import torchvision
import utils2 as utils
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import secrets
from architect import Architect
import logging
import torch.backends.cudnn as cudnn
import sys
from numpy.random import choice

parser = argparse.ArgumentParser("M2M")
parser.add_argument('--exp_name', type=str, default='EMNAS', help='experiment name')
# Training Settings
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--print_freq', type=int, default=100, help='print frequency of training')
parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
parser.add_argument('--tissue_dir', type=str, default='./tissue_model/', help='Tissue direction')
parser.add_argument('--seed', type=int, default=2, help='training seed')
# Dataset Settings
parser.add_argument('--data_root', type=str, default='/dataset/', help='dataset dir')
parser.add_argument('--classes', type=int, default=10, help='dataset classes')
parser.add_argument('--dataset', type=str, default='cifar100', help='path to the dataset')
parser.add_argument('--cutout', action='store_true', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

parser.add_argument('--epochs', type=int, default=70, help='epoch')
parser.add_argument('--Warm_epoch', type=int, default=0, help='Warm_epoch')
parser.add_argument('--Train_epoch', type=int, default=60, help='Train_epoch')
parser.add_argument('--x', type=int, default=5, help='P_epoch')

args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

np.random.seed(args.seed)
torch.cuda.set_device(0)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled = True
torch.cuda.manual_seed(args.seed)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def train(args, train_loader,val_loader, model,architect, criterion, optimizer,epoch,Train_Ar,All,Stop):
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)
    val_loader = iter(val_loader)

    l = len(Train_Ar)
    li = [j for j in range(l)]
    random.shuffle(li)
    i = 0
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device,non_blocking=True)
        input_search, target_search = next(val_loader)
        input_search, target_search = input_search.to(args.device), target_search.to(args.device,non_blocking=True)

        if All:
            n1 = secrets.choice(Ar_allModel)
            n2 = secrets.choice(Ar_allModel)
            Ar = n1 + ' ' + n2
            Ar = [Ar, 0]
        else:
            if i < l:
                Ar = Train_Ar[li[i]]
                i = i + 1
            else:
                random.shuffle(li)
                Ar = Train_Ar[li[0]]
                i = 1

        if (epoch+1) >= args.Warm_epoch:
            architect.step(input_search, target_search, Ar[0], Ar[1],epoch+1)

        if not Stop:
            optimizer.zero_grad()
            outputs = model(inputs, Ar[0],0,True,True,epoch+1)

            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            train_loss.update(loss.item(), n)
            train_acc.update(prec1.item(), n)

            if step % args.print_freq == 0 or step == len(train_loader) - 1:
                logging.info(
                    '[Model Training] lr: %f epoch: %03d/%03d, step: %03d/%03d, '
                    'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
                    % (lr, epoch+1, args.epochs, step+1, steps_per_epoch,
                       loss.item(), train_loss.avg, prec1, train_acc.avg)
                )

    return train_loss.avg, train_acc.avg

def validate(model,val_loader, criterion,encoding,V):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda',non_blocking=True)
            outputs = model(inputs,encoding,V,False,False)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)
    return val_loss.avg,val_acc.avg

XX = 0
xx = 0
def update_value(M2M,Train_data,evl,evl_data,arch_parameters):
    f1min = 1000
    f2min = 1000
    Best_model = ''
    Q = []
    for n,Ar in enumerate(Train_data):
        logging.info('正在更新第%d个架构族：%s' % (n,Ar[0]))
        loss, acc = validate(M2M.Model, M2M.val_loader, M2M.criterion, Ar[0],0)
        m = Ar[0].split(' ')
        M = []
        for n, ar in enumerate(m):
            a = ar.split('-')
            l = Ar_Model[ar]
            mo = []
            for no, j in enumerate(a):
                w1 = F.softmax(arch_parameters[n * 79 + l][no * 2][:-1], dim=-1)
                w2 = F.softmax(arch_parameters[n * 79 + l][no * 2 + 1][:-1], dim=-1)
                op1 = torch.argmax(w1, 0).cpu().data.numpy() + 1
                op2 = torch.argmax(w2, 0).cpu().data.numpy() + 1

                t = 0
                d = ''
                for k in j:
                    if k != '0':
                        if t == 0:
                            s = str(int(op1))
                            t = t + 1
                        else:
                            s = str(int(op2))
                        d = d + s
                    else:
                        d = d + k
                mo.append(d)
            M.append('-'.join(mo))
        encoding = ' '.join(M)

        print(encoding)
        with HiddenPrints():
            macs, params = profile(evl, inputs=(evl_data, encoding,))
        Model = [Ar[0],params / 1e5,100-acc]

        M2M.Ar_model[Ar[0]] = Model
        Q.append(Model)
        if Model[1] < f1min:
            f1min = Model[1]
        if Model[2] < f2min:
            f2min = Model[2]
            Best_model = Model[0]
    return Q,f1min,f2min,Best_model

def init_population(s):
    # 保存具体架构
    Ar_all = []
    N = []
    # 保存具体model
    Ar_model = {}
    for j in range(s*2):
        # 旨在将个体均衡产生
        n1 = secrets.choice(Ar_allModel)
        n2 = secrets.choice(Ar_allModel)
        Ar = n1 + ' ' + n2
        while n1 in N:
            n1 = secrets.choice(Ar_allModel)
            n2 = secrets.choice(Ar_allModel)
            Ar = n1 + ' ' + n2

        Ar_all.append(Ar)
        N.append(n1)
        Ar_model[Ar] = []
    return Ar_all,Ar_model

def up_p(P, arch_parameters,epoch,T,Stop):
    p = []
    Train_data = []
    TT = [0,0,0,0]
    for i in P:
        m = i[0].split(' ')
        M = []
        num = 0
        for n,ar in enumerate(m):
            a = ar.split('-')
            l = Ar_Model[ar]
            mo = []
            if n == 0:
                nn = 1
            else:
                nn = 0
            N = 0
            for no,j in enumerate(a):
                w1 = F.softmax(arch_parameters[n * 79 + l][no * 2][:-1], dim=-1)
                w2 = F.softmax(arch_parameters[n * 79 + l][no * 2 + 1][:-1], dim=-1)
                op1 = torch.argmax(w1, 0).cpu().data.numpy() + 1
                op2 = torch.argmax(w2, 0).cpu().data.numpy() + 1
                if nn:
                    if op1 == 1:
                        N = N + 1
                    if op2 == 1:
                        N = N + 1

                t = 0
                d = ''
                for k in j:
                    if k != '0':
                        if t == 0:
                            s = str(int(op1))
                            t = t + 1
                        else:
                            s = str(int(op2))
                        d = d + s
                    else:
                        d = d + k
                mo.append(d)

            num = num + N
            M.append('-'.join(mo))

        if num <= 2:
            TT[num] = TT[num] + 1
        else:
            TT[3] = TT[3] + 1

        encoding = ' '.join(M)
        p.append([encoding, i[1], i[2]])

        if num <= 2:
            Train_data.append([i[0],0])
        else:
            Train_data.append([i[0], 1])
    if T:
        nd = pd.DataFrame(p, columns=['Ar', 'P', 'Er'])
        if Stop:
            name = 'Result/Stop_nd' + str(epoch + 1) + '.xlsx'
            f = '/Result/Stop_nd' + str(epoch + 1) + '.png'
        else:
            name = '/Result/nd' + str(epoch + 1) + '.xlsx'
            f = '/Result/nd' + str(epoch + 1) + '.png'
        nd.to_excel(name)
        s = nd.loc[:, ['P', 'Er']]
        ax = s.plot(kind='scatter', x=0, y=1)
        fig = ax.get_figure()
        fig.savefig(f)
        plt.close()
    return TT,Train_data

E = 0


def mymain():
    # 设置训练集与验证集
    # cifra
    train_transform, valid_transform = utils.data_transforms(args)
    trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_root, args.dataset),
                                            train=True, download=True, transform=train_transform)
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(0.5 * num_train))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:20000]),
        pin_memory=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[20000:40000]),
        pin_memory=True, num_workers=8)

    val = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[40000:]),
        pin_memory=True, num_workers=8)

    criterion = nn.CrossEntropyLoss().to('cuda')
    # Load Pretrained Supernet

    # model = NetworkCIFAR(16, 10, 8, criterion).to('cuda')
    model = NetworkCIFAR(28, 10, 14, criterion).to('cuda')
    # model = NetworkCIFAR(36, 10, 20, criterion).to('cuda')

    architect = Architect(model, args)

    evl = evl_model(36, 10, 20, args.auxiliary).to('cuda')
    evl_data = torch.randn(1, 3, 32, 32).to('cuda')

    # 建立种群
    S = 39
    s = int(S/3)

    logging.info('开始初始化种群')

    Ar_all, Ar_model = init_population(S)

    Train_data = []
    q = []
    for i in Ar_all:
        Train_data.append([i, 0])
        q.append(i)

    # 建立M2M
    M2M = M2M_revise.M2M(args.epochs, args.Train_epoch, args.Warm_epoch, Ar_model, Ar_all, model, val, S, s)

    All = True

    pp = []
    p1 = []
    k = [0,0,0,0]
    Stop = False
    End = False
    st = []

    last_mean = 100
    now_mean = 100
    last_mean_all = 100
    now_mean_all = 100

    Epoch = 0
    max_data=-100
    min_data= 100
    Up = None
    Low = None

    ERR = []
    for epoch in range(args.epochs):
        train_loss, train_acc = train(args, train_loader, val_loader, M2M.Model, architect, M2M.criterion, M2M.optimizer,
              epoch,Train_data,All,Stop)
        M2M.scheduler.step()

        if not Stop:
            if ((epoch + 1) % args.x == 0) and ((epoch + 1) >= args.Warm_epoch):
                All = False

                op_noMax = []
                for num, i in enumerate(M2M.Model._arch_parameters):
                    Amax, Amin = i.max(), i.min()
                    Amax, Amin = Amax.cpu().data, Amin.cpu().data
                    if Amax > max_data:
                        max_data = Amax
                    if Amin < min_data:
                        min_data = Amin

                    for j in i:
                        o = j.cpu().data.numpy().tolist()
                        op_noMax.append(o)

                    Up = torch.full((1, 7), max_data).squeeze()
                    Low = torch.full((1, 7), min_data).squeeze()
                    # Ul = Up-Low

                op_noMax = pd.DataFrame(op_noMax)
                name_op = '/Result/op_noMax' + str(epoch + 1) + '.xlsx'
                op_noMax.to_excel(name_op, index=False)

                # 根据现有的model，更新个体数据
                logging.info('开始更新种群适应值')
                Q, f1min, f2min,M2M.Best_model = update_value(M2M, Train_data,evl,evl_data,M2M.Model._arch_parameters)

                if M2M.f1min < f1min:
                    f1min = M2M.f1min
                if M2M.f2min < f2min:
                    f2min = M2M.f2min

                x1, y1 = 50, 100
                x2, y2 = 150, 25
                limit = (y1-y2)*(epoch+1-x2)/(x1-x2) + y2

                logging.info('开始分配种群')
                M2M.allocation(Q, f1min, f2min, limit)
                # 仅非支配排序
                # M2M.NonD(Q, f1min, f2min, limit)

                last_mean = now_mean
                last_mean_all = now_mean_all
                # 去除离散点
                x1 = sorted(M2M.p, key=lambda stu: stu[2])[:20]
                x2 = M2M.p
                # x2 = Q
                now_mean_all = pd.DataFrame(x2)[2].mean()
                now_mean = pd.DataFrame(x1)[2].mean()
                ERR.append([epoch+1,now_mean,now_mean_all, train_acc])

                print(now_mean, last_mean,now_mean_all,last_mean_all)
                if (abs(now_mean - last_mean) <= 0.3):
                    logging.info('开始评估模型权重优化程度')
                    _, Train_data = up_p(M2M.p, M2M.Model._arch_parameters, epoch, 1, Stop)
                    Stop = True
                    Epoch = epoch
                    break

                logging.info('开始评估模型权重优化程度')
                _, _ = up_p(M2M.p, M2M.Model._arch_parameters, epoch, 1, Stop)

                # 根据种群更新数据
                logging.info('开始种群交叉变异')
                Train_data,q,arch_parameters = M2M.offspring(epoch+1, M2M.Model.arch_parameters(),Up,Low)
                M2M.Model.updata_arch_parameters(arch_parameters)
        print('\n')
    else:
        logging.info('开始评估模型权重优化程度')
        _, Train_data = up_p(M2M.p, M2M.Model._arch_parameters, args.epochs, 1, Stop)
        Stop = True
        Epoch = args.epochs + 1

    while True:
        Epoch += 1
        if (Epoch + 1) <= (args.epochs*2):
            train_loss, train_acc = train(args, train_loader, val_loader, M2M.Model, architect, M2M.criterion, M2M.optimizer,
                  Epoch, Train_data, All, Stop)
            if ((Epoch + 1) % args.x == 0):
                op_noMax = []
                for num, i in enumerate(M2M.Model._arch_parameters):
                    Amax, Amin = i.max(), i.min()
                    Amax, Amin = Amax.cpu().data, Amin.cpu().data
                    if Amax > max_data:
                        max_data = Amax
                    if Amin < min_data:
                        min_data = Amin

                    for j in i:
                        o = j.cpu().data.numpy().tolist()
                        op_noMax.append(o)

                    Up = torch.full((1, 7), max_data).squeeze()
                    Low = torch.full((1, 7), min_data).squeeze()

                op_noMax = pd.DataFrame(op_noMax)
                name_op = '/home/l708/Code/NAS/EA3/Result/op_noMax' + str(Epoch + 1) + '.xlsx'
                op_noMax.to_excel(name_op, index=False)
                M2M.p, f1min, f2min, M2M.Best_model = update_value(M2M, Train_data, evl, evl_data, M2M.Model._arch_parameters)

                if M2M.f1min < f1min:
                    f1min = M2M.f1min
                if M2M.f2min < f2min:
                    f2min = M2M.f2min

                logging.info('开始分配种群')
                M2M.allocation(M2M.p, f1min, f2min, 100)
                logging.info('最终模型权重')
                k, Train_data = up_p(M2M.p, M2M.Model._arch_parameters, Epoch, 1, Stop)
                print(k)

                last_mean = now_mean
                last_mean_all = now_mean_all
                x1 = sorted(M2M.p, key=lambda stu: stu[2])[:20]
                x2 = M2M.p
                now_mean_all = pd.DataFrame(x2)[2].mean()
                now_mean = pd.DataFrame(x1)[2].mean()

                ERR.append([Epoch+1,now_mean,now_mean_all,train_acc])

                print(now_mean, last_mean, now_mean_all, last_mean_all)
                if (abs(now_mean - last_mean) <= 0.3):
                    if now_mean > last_mean:
                        print('上一个可用')
                    break

                arch_parameters = M2M.arch(Epoch+1, M2M.Model.arch_parameters(),Up,Low)
                M2M.Model.updata_arch_parameters(arch_parameters)
            else:
                logging.info('开始第%d轮评估模型权重优化程度', Epoch + 1)
                k, Train_data = up_p(M2M.p, M2M.Model._arch_parameters, Epoch, 0, Stop)
                print(k)
            print('\n')
        else:
            break

    op_noMax = pd.DataFrame(ERR,columns=['Epoch','now_mean','now_mean_all','train_acc'])
    name_op = '/Result/ERR.xlsx'
    op_noMax.to_excel(name_op, index=False)

if __name__ == '__main__':
    mymain()