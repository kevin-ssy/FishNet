import argparse
import time
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.profile import count_params
import os
from torch.autograd.variable import Variable
import models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='models architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--config', default='cfgs/local_test.yaml')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate models on validation set')
parser.add_argument('--train_image_list', default='', type=str, help='path to train image list')

parser.add_argument('--input_size', default=224, type=int, help='img crop size')
parser.add_argument('--image_size', default=256, type=int, help='ori img size')

parser.add_argument('--model_name', default='', type=str, help='name of the models')

best_prec1 = 0

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

USE_GPU = torch.cuda.is_available()


def main():
    global args, best_prec1, USE_GPU
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    # create models
    if args.input_size != 224 or args.image_size != 256:
        image_size = args.image_size
        input_size = args.input_size
    else:
        image_size = 256
        input_size = 224
    print("Input image size: {}, test size: {}".format(image_size, input_size))

    if "model" in config.keys():
        model = models.__dict__[args.arch](**config['model'])
    else:
        model = models.__dict__[args.arch]()

    if USE_GPU:
        model.cuda()
        model = torch.nn.DataParallel(model)

    count_params(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_size = args.input_size

    ratio = 224.0 / float(img_size)
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(int(256 * ratio)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ]))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # else:
    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=(train_sampler is None), sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_name = '{}/{}_{}_best.pth.tar'.format(args.save_path, args.model_name, epoch) if is_best else\
            '{}/{}_{}.pth.tar'.format(args.save_path, args.model_name, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, filename=save_name)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #  pytorch 0.4.0 compatible
        if '0.4.' in torch.__version__:
            if USE_GPU:
                input_var = torch.cuda.FloatTensor(input.cuda())
                target_var = torch.cuda.LongTensor(target.cuda())
            else:
                input_var = torch.FloatTensor(input)
                target_var = torch.LongTensor(target)
        else:  # pytorch 0.3.1 or less compatible
            if USE_GPU:
                input = input.cuda()
                target = target.cuda(async=True)
            input_var = Variable(input)
            target_var = Variable(target)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))

        # measure accuracy and record loss
        reduced_prec1 = prec1.clone()
        reduced_prec5 = prec5.clone()

        top1.update(reduced_prec1[0])
        top5.update(reduced_prec5[0])

        reduced_loss = loss.data.clone()
        losses.update(reduced_loss)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #  check whether the network is well connected
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            with open('logs/{}_{}.log'.format(time_stp, args.arch), 'a+') as flog:
                line = 'Epoch: [{0}][{1}/{2}]\t ' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                        batch_time=batch_time, loss=losses, top1=top1, top5=top5)
                print(line)
                flog.write('{}\n'.format(line))


def validate(val_loader, model, criterion):
    global time_stp
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #  pytorch 0.4.0 compatible
        if '0.4.' in torch.__version__:
            with torch.no_grad():
                if USE_GPU:
                    input_var = torch.cuda.FloatTensor(input.cuda())
                    target_var = torch.cuda.LongTensor(target.cuda())
                else:
                    input_var = torch.FloatTensor(input)
                    target_var = torch.LongTensor(target)
        else:  # pytorch 0.3.1 or less compatible
            if USE_GPU:
                input = input.cuda()
                target = target.cuda(async=True)
            input_var = Variable(input, volatile=True)
            target_var = Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            line = 'Test: [{0}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                   loss=losses, top1=top1, top5=top5)

            with open('logs/{}_{}.log'.format(time_stp, args.arch), 'a+') as flog:
                flog.write('{}\n'.format(line))
                print(line)

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    time_stp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    main()
