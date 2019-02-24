from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
#torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from models import GCN, GAT

from data.face_bbx import FACE
#from data.W300LP import W300LP
#from data.AFLW2000 import AFLW2000
from pylib import FaceAcc, Evaluation

import numpy as np
#np.random.seed(0)

from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
#import pose.datasets as datasets

print(models)
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print('Model names --.{}'.format(model_names))

idx = [1,2,3,4,5,6,11,12,15,16]

best_acc = 0.1

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    #random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

    print('Seed set --> {}'.format(seed_value))

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def saveToPts(filename, landmarks):
    #pts = landmarks + 1
    pts = landmarks
    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(filename, pts, delimiter=' ', header=header, footer='}', fmt='%.3f', comments='')


def main(args):

    random_seed(0, True)

    global best_acc

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.num_classes)

    model = torch.nn.DataParallel(model).to(device)
    

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss(reduction='mean').to(device)

    optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = '300W-' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    #cudnn.benchmark = True  # There is BN issue here see https://github.com/bearpaw/pytorch-pose/issues/33
    print(' Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    
    #Data loading code
    train_loader = torch.utils.data.DataLoader(
         FACE('dataset/animal_21Feb_processed_sep.json', 'face_datasets', is_train=True),
         batch_size=args.train_batch, shuffle=True,
         num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
         FACE('dataset/animal_21Feb_processed_sep.json', 'face_datasets', is_train=False),
         batch_size=args.test_batch, shuffle=False,
         num_workers=args.workers, pin_memory=True)

    #############################################################################################################################################################################   
    #############################################################################################################################################################################   
 


    #data_string = 'face_datasets/AFLW2000' # change this for 300W-LP training and AFLW2000 testing

    # train_loader = torch.utils.data.DataLoader(
    #     W300LP(data_string, is_train=True),
    #     batch_size=args.train_batch, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     W300LP(data_string', is_train=False),
    #     batch_size=args.test_batch, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    #train_loader = torch.utils.data.DataLoader(
    #    AFLW2000(data_string, is_train=True),
    #    batch_size=args.train_batch, shuffle=True,
    #    num_workers=args.workers, pin_memory=True)

    #val_loader = torch.utils.data.DataLoader(
    #    AFLW2000(data_string, is_train=False),
    #    batch_size=args.test_batch, shuffle=False,
    #    num_workers=args.workers, pin_memory=True)

    
    #############################################################################################################################################################################   
    #############################################################################################################################################################################   
 
    
    
    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader, model, criterion, args.num_classes, args.debug, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return



    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # # decay sigma
        # if args.sigma_decay > 0:
        #     train_loader.dataset.sigma *=  args.sigma_decay
        #     val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, args.debug, args.flip, args.train_batch)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, args.num_classes, args.debug, args.flip)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc < best_acc
        if is_best:
            print('Found Reduced Val_RMSE --> {} at epoch -- > {}'.format(valid_acc, epoch+1))
        best_acc = min(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(train_loader, model, criterion, optimizer, debug=False, flip=True, bsize=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    


    gt_win, pred_win = None, None
    bar = Bar('Processing', max=len(train_loader))
    for i, (input, target, pts) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device, non_blocking=True)

        # compute output
        output, _ = model(input)

        

        loss = 0
        for per_out in output:
            tmp_loss = (per_out - target) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()
        
                
        pred_pts_0, pred_pts_1, pred_pts_2 = FaceAcc.heatmap2pts(output[-1].cpu(), flag=0)

        #pred_pts_0, pred_pts_1, pred_pts_2 = FaceAcc.heatmap2pts(output_GCN.cpu(), flag=0)

        pred_pts_2 -= 1
        # rmse0 = np.sum(FaceAcc.per_image_rmse(pred_pts_0.numpy() * 4., pts.numpy())) / img.size(0)
        # rmse1 = np.sum(FaceAcc.per_image_rmse(pred_pts_1.numpy() * 4., pts.numpy())) / img.size(0)
        acc = np.sum(FaceAcc.per_image_rmse(pred_pts_2.numpy() * 4., pts.numpy(), norm_type='face_size')) / input.size(0)
        
        if debug: # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(input, target)
            pred_batch_img = batch_with_heatmap(input, output[0])
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | RMSE: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg
                    )
        bar.next()

    bar.finish()
    return losses.avg, acces.avg


def validate(val_loader, model, criterion, num_classes, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    #debug = True

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for i, (input, target, pts, index, center, scale, f_name) in enumerate(val_loader):
        
        #print('Val...Image -->{} '.format(i))
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output, _ = model(input)
        score_map = output[-1].cpu()
        if flip:
            flip_input = torch.autograd.Variable(
                    torch.from_numpy(fliplr(input.clone().numpy())).float().to(device),
                    volatile=True
                )
            flip_output_var, _ = model(flip_input)
            flip_output = flip_back(flip_output_var[-1].cpu())
            score_map += flip_output




        loss = 0
        for per_out in output:
            tmp_loss = (per_out - target) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()

        # loss = 0
        # for o in output:
        #     loss += criterion(o, target)

        

        # generate predictions
        preds = Evaluation.final_preds(score_map, center, scale, [64, 64], rot=0)
       
        acc = np.sum(FaceAcc.per_image_rmse(preds.numpy(), pts.numpy(), norm_type='face_size')) / input.size(0)
        
      	_, fn1 = os.path.split(f_name[0])

        #fit_save_loc = 'face_datasets/animals_21Feb_fittings/' + fn1[:-4] + '.pts'
        
        #print('fname -> {}'.format(fit_save_loc))
        
        #saveToPts(fit_save_loc, torch.squeeze(preds).numpy())
        
        for n in range(score_map.size(0)):
            predictions[index[n], :, :] = preds[n, :, :]


        if debug:
            gt_batch_img = batch_with_heatmap(input, target)
            pred_batch_img = batch_with_heatmap(input, score_map)
            print('im here')
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | RMSE: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.val,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg
                    )
        bar.next()

    bar.finish()
    return losses.avg, acces.avg, predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='HG',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=9, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=110, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=10, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')


    main(parser.parse_args())
