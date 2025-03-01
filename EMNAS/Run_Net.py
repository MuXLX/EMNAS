import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import logging
import sys
import time
# ids = [0,1]
import torch
import torch.nn as nn
import torchvision
from thop import profile
from torchvision import datasets
from Net import NetworkCIFAR as Darts
from torch.autograd import Variable
import pandas as pd
import utils2 as utils
import numpy as np
import torch.backends.cudnn as cudnn
from utils2 import data_transforms

parser = argparse.ArgumentParser("Darts")
parser.add_argument('--exp_name', type=str, default='NET', help='experiment name')
# Supernet Settings
# parser.add_argument('--layers', type=int, default=20, help='batch size')
# parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
# Training Settings
# 96
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--print_freq', type=int, default=100, help='print frequency of training')
parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
parser.add_argument('--tissue_dir', type=str, default='./tissue_model/cell3/', help='Tissue direction')
parser.add_argument('--seed', type=int, default=0, help='training seed')
# Dataset Settings
parser.add_argument('--data_root', type=str, default='/home/l708/Code/NAS/dataset/', help='dataset dir')
parser.add_argument('--classes', type=int, default=10, help='dataset classes')
parser.add_argument('--dataset', type=str, default='cifar100', help='path to the dataset')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true',default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()
np.random.seed(args.seed)
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(args.device)
torch.cuda.set_device(0)

cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled = True
torch.cuda.manual_seed(args.seed)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
# utils.set_seed(args.seed)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def train(args, epoch, train_loader, model, criterion, optimizer):
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device,non_blocking=True)
        optimizer.zero_grad()
        outputs,outputs_aux = model(inputs)
        loss = criterion(outputs, targets)
        if args.auxiliary:
            loss_aux = criterion(outputs_aux, targets)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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


def validate(args, val_loader, model, criterion):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc_1 = utils.AverageMeter()
    val_acc_5 = utils.AverageMeter()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device,non_blocking=True)
            outputs,_ = model(inputs)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc_1.update(prec1.item(), n)
            val_acc_5.update(prec5.item(), n)
    return val_loss.avg, val_acc_1.avg,val_acc_5.avg

if __name__ == '__main__':
    HiddenPrints = HiddenPrints()

    train_transform, valid_transform = data_transforms(args)
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset),
                                            train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=True, num_workers=8)

    valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset),
                                          train=False, download=True, transform=valid_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, pin_memory=True, num_workers=8)

    # C10
    encoding = '14-404-0440-34000 13-101-0044-33000'
    # C100
    # encoding = '14-404-6070-04600 34-011-3007-04100'

    model = Darts(36, 100, 20,args.auxiliary, encoding)
    model = model.to('cuda')

    x = torch.randn(1,3,32,32).to('cuda')

    with HiddenPrints:
        macs, params = profile(model, inputs=(x,))
    print(encoding)
    print(macs / 1e6, params / 1e6)


    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(),args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_acc = 0.0

    Train = []
    Best_A = [encoding,0,0]
    Best_A = pd.DataFrame(Best_A)
    for epoch in range(args.epochs):
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        train_loss, train_acc = train(args, epoch, train_loader, model, criterion, optimizer)
        scheduler.step()
        logging.info(
            '[Model Training] epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
            (epoch + 1, train_loss, train_acc)
        )


        loss, acc1,acc5 = validate(args, val_loader, model, criterion)

        if best_val_acc < acc1:
            best_val_acc = acc1

            best_tissue = os.path.join('/weight/', '%s_%s' % (args.exp_name, 'best.pth'))
            torch.save(model.state_dict(), best_tissue)

            Best_A[0][1] = epoch + 1
            Best_A[0][2] = best_val_acc
            Best_A.to_excel('/weight/Best_A.xlsx')

            logging.info('Save best model to %s' % best_tissue)
            # T[4] = acc1
        logging.info(
            '[Model Validation] epoch: %03d, val_loss: %.3f, val_acc1: %.3f, val_acc5: %.3f, best_acc: %.3f'
            % (epoch + 1, loss, acc1,acc5, best_val_acc)
        )
        Train.append([epoch+1,train_loss,loss,train_acc, acc1])
        print('\n')

    Record = pd.DataFrame(Train, columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    Record.to_excel('/weight/Record.xlsx')
