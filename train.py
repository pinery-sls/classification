import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils
from thop import profile

import os
import time
import math
from data import cifar10
from importlib import import_module

device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')

checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

print("Preparing data...")
loader = cifar10.Data(args)


def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter('Time', ':6.3f')
    accurary = utils.AverageMeter('Time', ':6.3f')
    top5_accuracy = utils.AverageMeter('Time', ':6.3f')
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accurary.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Accuracy {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accurary.avg), cost_time
                    )
                )
            else:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), cost_time
                    )
                )
            start_time = current_time

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter('Time', ':6.3f')
    accurary = utils.AverageMeter('Time', ':6.3f')
    top5_accuracy = utils.AverageMeter('Time', ':6.3f')

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accurary.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            logger.info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
            )
        else:
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accurary.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return accurary.avg
    else:
        return top5_accuracy.avg

def main():

    if args.arch == 'vgg':
        model = import_module(f'model.{args.arch}').vgg(args.cfg).to(device)
    elif args.arch == 'resnet':
        model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(args.cfg).to(device)
    elif args.arch == 'mobilenetv1':
        model = import_module(f'model.{args.arch}').mobilenetv1(args.cfg).to(device)
    elif args.arch == 'mobilenetv2':
        model = import_module(f'model.{args.arch}').mobilenet_v2().to(device)
    else:
        raise('arch not exist!')

    print("model=", model)

    start_epoch = 0
    best_acc = 0.0
    #test(origin_model,loader.testLoader, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
    elif args.lr_type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    
    # 训练num_epochs次
    for epoch in range(start_epoch, args.num_epochs):
        # 训练
        train(model, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.dataset == 'Imagenet' else (1, ))
        scheduler.step()
        # 测试
        test_acc = test(model, loader.testLoader, topk=(1, 5) if args.dataset == 'Imagenet' else (1, ))

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
        }
        # 保存最好的compact model
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accuracy: {:.3f}'.format(float(best_acc)))
    #* Compute flops, params
    inputs = torch.randn(1, 3, 32, 32)
    model = model.cpu()
    flops, params = profile(model, inputs=(inputs, ))
    logger.info(f'{args.arch}\'s baseline model: FLOPs = {flops/10**6:.2f}M, Params = {params/10**6:.2f}M')



if __name__ == '__main__':
    main()