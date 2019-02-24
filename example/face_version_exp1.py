from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
torch.manual_seed(0)
#torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from torch.autograd import Variable
from scipy.misc import imsave



from data.face_bbx import FACE
#from data.W300LP import W300LP
#from data.AFLW2000 import AFLW2000
from pylib import FaceAcc, Evaluation

from edge_info import *

import numpy as np
np.random.seed(0)

from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap, batch_with_heatmap2, show_sample, show_joints
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
#import pose.datasets as datasets

from models import GCN

print(models)
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print('Model names --.{}'.format(model_names))

idx = [1,2,3,4,5,6,11,12,15,16]

best_acc = 0.1

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def normalize(mx):
    
#     D = np.array(np.sum(mx, axis=1))
       
#     r_inv = np.power(D, -1).flatten()
    
#     r_inv[np.isinf(r_inv)] = 0.
    
#     r_mat_inv = np.matrix(np.diag(r_inv))

#     print(r_mat_inv.shape)
    
#     mx = r_mat_inv.dot(mx)

#     return mx.astype(np.float32)

def normalize_adj(adj_matrix):

    adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

    
    row_sum = np.sum(adj_matrix, axis=-1)
    D_inv_sqrt = np.diag(np.power(row_sum, -0.5).flatten() + 1e-7)
    adj_matrix = D_inv_sqrt.dot(adj_matrix).dot(D_inv_sqrt)

    return adj_matrix.astype(np.float32)
    
    #print(mx)
    
    # D = np.array(np.sum(mx, axis=1))
    # r_inv_sqrt = np.power(D, -0.5).flatten()
    
    # r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.

    # r_mat_inv_sqrt = np.matrix(np.diag(r_inv_sqrt))

    # #print(r_mat_inv_sqrt)

    # #print(r_mat_inv_sqrt.transpose().dot(r_mat_inv_sqrt))

    # return mx.dot((r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)).astype(np.float32)


def saveToPts(filename, landmarks):
    #pts = landmarks + 1
    pts = landmarks
    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(filename, pts, delimiter=' ', header=header, footer='}', fmt='%.3f', comments='')


def main(args):
    global best_acc

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.num_classes)

    model = torch.nn.DataParallel(model).to(device)

    # for param in model.parameters():

    #     param.requires_grad = False

    model_GCN = GCN(nfeat=4096, nhid=4096, nclass=4096, dropout=1.0, init='uniform')

    model_GNN = torch.nn.DataParallel(model_GCN).to(device)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss(reduction='mean').to(device)

    # optimizer = torch.optim.RMSprop(model.parameters(),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.RMSprop(model_GNN.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optimizer = torch.optim.RMSprop(list(model.parameters()) + list(model_GNN.parameters()),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = '300W-' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    cudnn.benchmark = True  # There is BN issue here see https://github.com/bearpaw/pytorch-pose/issues/33
    print(' Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

   
    #Data loading code
    train_loader = torch.utils.data.DataLoader(
         FACE('dataset/face.json', 'face_datasets', is_train=True),
         batch_size=args.train_batch, shuffle=True,
         num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
         FACE('dataset/face.json', 'face_datasets', is_train=False),
         batch_size=args.test_batch, shuffle=False,
         num_workers=args.workers, pin_memory=True)

    #############################################################################################################################################################################   
    #############################################################################################################################################################################   
 


    data_string = 'face_datasets/AFLW2000' # change this for 300W-LP training and AFLW2000 testing

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
    
    edges = create_edges(args.pairwise, args.symmetrical)

    A = create_adjacency_matrix(edges, 68, 2)

    A = A[:, :68]
    
    A = normalize_adj(A)
    
    print(type(A))

    #A = np.expand_dims(A, axis=0)

    #A = np.repeat(A, args.train_batch, axis=0)
    
    
    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader, model, model_GCN, A, criterion, args.num_classes, args.debug, args.flip)
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
        train_loss, train_acc = train(train_loader, model, model_GCN, criterion, optimizer, args.debug, args.flip, args.train_batch, A)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = validate(val_loader, model, model_GCN, A, criterion, args.num_classes, args.debug, args.flip)

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


def train(train_loader, model, model_GNN, criterion, optimizer, debug, flip, bsize, A):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.eval()

    model_GNN.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Processing', max=len(train_loader))
    for i, (input, target, pts) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #model_GNN.zero_grad()

        input, target = input.to(device), target.to(device, non_blocking=True)

        # compute output
        output, _ = model(input)

        #print('Model output size {}'.format(output[0].size()))

        # loss = 0
        # for per_out in output:
        #     tmp_loss = (per_out - target) ** 2
        #     loss = loss + tmp_loss.sum() / tmp_loss.numel()

        # loss = criterion(output[0], target)
        # for j in range(1, len(output)):
        #     loss += criterion(output[j], target)

        ############################################################################################################################################################################################
        ############################################################################################################################################################################################
        A1 = A.copy()


        # A1 = np.repeat(A1, input.size(0), axis=0)

        # # for ii in range(0, input.size(0)):

        # #     #features[ii] = normalize(features[ii])
        # #     A1[ii] = normalize_adj(A1[ii])

        
        adj = torch.from_numpy(A1).to(device) # A normalized and converted to tensor
                
        
        output1 = output[-1]

        features = output1.view(output1.size(0),output1.size(1),-1) # torch of size -> Batchsize x N x (HxW)


        features = torch.squeeze(features)
        
        #print(features.size())

        output_GNN = model_GNN(features, adj)

        #     #features = features.numpy() # featuers to numpy

        #print(features.dtype)
        #print(adj.dtype)

        output_GNN = torch.unsqueeze(output_GNN, 0)

        output_GNN = output_GNN.view(output_GNN.size(0),output_GNN.size(1), 64, 64)

        
        
        ############################################################################################################################################################################################
        ############################################################################################################################################################################################


        loss = 0
        for per_out in output:
            tmp_loss = (per_out - target) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()
        
        # last GCN output loss --
        
        #print(output_GNN)

        tmp_loss = (output_GNN - target) ** 2
        loss = loss + tmp_loss.sum() / tmp_loss.numel()


        
        #pred_pts_0, pred_pts_1, pred_pts_2 = FaceAcc.heatmap2pts(output[-1].cpu(), flag=0)

        pred_pts_0, pred_pts_1, pred_pts_2 = FaceAcc.heatmap2pts(output_GNN.cpu(), flag=0)

        pred_pts_2 -= 1
        # rmse0 = np.sum(FaceAcc.per_image_rmse(pred_pts_0.numpy() * 4., pts.numpy())) / img.size(0)
        # rmse1 = np.sum(FaceAcc.per_image_rmse(pred_pts_1.numpy() * 4., pts.numpy())) / img.size(0)
        acc = np.sum(FaceAcc.per_image_rmse(pred_pts_2.numpy() * 4., pts.numpy(), norm_type='inter')) / input.size(0)
        
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


def validate(val_loader, model, model_GNN, A, criterion, num_classes, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    #debug = True

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    model_GNN.eval()

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
        output, res_output = model(input)


        ############################################################################################################################################################################################
        ############################################################################################################################################################################################
        A1 = A.copy()
        #A1 = np.repeat(A1, input.size(0), axis=0)

        # for ii in range(0, input.size(0)):

        #     #features[ii] = normalize(features[ii])
        #     A1[ii] = normalize_adj(A1[ii])



        adj = torch.from_numpy(A1).to(device) # A normalized and converted to tensor
                
        output1 = output[-1]

        features = output1.view(output1.size(0),output1.size(1),-1) # torch of size -> Batchsize x N x (HxW)

        # features_max = features.max(-1, keepdim=True)[0]

        # masked_features = features.ge(features_max)

        # masked_features = masked_features.type(torch.FloatTensor)

        #print(features.dtype)
        #print(adj.dtype)

        # masked_features = Variable(masked_features)
        # adj = Variable(adj)
        # target = Variable(target)

        #masked_features = masked_features.to(device)

        features = torch.squeeze(features)
        #print(features.size())

        output_GNN = model_GNN(features, adj)

        #     #features = features.numpy() # featuers to numpy

        #print(features.dtype)
        #print(adj.dtype)

        output_GNN = torch.unsqueeze(output_GNN, 0)

        output_GNN = output_GNN.view(output_GNN.size(0),output_GNN.size(1), 64, 64)

        
        ############################################################################################################################################################################################
        ############################################################################################################################################################################################



        #score_map = output[-1].cpu()

        score_map = output_GNN.cpu()


        # if flip:
        #     flip_input = torch.autograd.Variable(
        #             torch.from_numpy(fliplr(input.clone().numpy())).float().to(device),
        #             volatile=True
        #         )
        #     flip_output_var = model(flip_input)
        #     flip_output = flip_back(flip_output_var[-1].cpu())
        #     score_map += flip_output

        
        loss = 0
        for per_out in output:
            tmp_loss = (per_out - target) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()

        #last GCN output loss --
        tmp_loss = (output_GNN - target) ** 2
        loss = loss + tmp_loss.sum() / tmp_loss.numel()

        # loss = 0
        # for o in output:
        #     loss += criterion(o, target)

        

        # generate predictions
        preds = Evaluation.final_preds(score_map, center, scale, [64, 64], rot=0)
       
        acc = np.sum(FaceAcc.per_image_rmse(preds.numpy(), pts.numpy(), norm_type='inter')) / input.size(0)

        
        
        _, fn1 = os.path.split(f_name[0])

        #fit_save_loc = 'face_datasets/300W_private_fittings/' + fn1[:-4] + '.pts'


        #print('fname -> {}'.format(fit_save_loc))
        
        
        #saveToPts(fit_save_loc, torch.squeeze(preds).numpy())
        
        
        for n in range(score_map.size(0)):
            predictions[index[n], :, :] = preds[n, :, :]



        #res_fmap = res_output[-1].cpu()
        
        # debug = True

        # if debug:
        #     gt_batch_img = batch_with_heatmap(input, target)

        #     feat_batch_img = batch_with_heatmap2(input, res_fmap)
        #     pred_batch_img = batch_with_heatmap(input, score_map)

        #     # pred_batch_img = show_sample(input, score_map)

        #     # pred_batch_img = pred_batch_img.numpy()

        #     # pred_batch_img = np.transpose(pred_batch_img, (1, 2, 0))
            
        #     #print('size ->{}'.format(pred_batch_img.shape))
        #     # if not gt_win or not pred_win:
        #     #     plt.subplot(121)
        #     #     gt_win = plt.imshow(gt_batch_img)
        #     #     plt.subplot(122)
        #     #     pred_win = plt.imshow(pred_batch_img)
        #     # else:
        #     #     gt_win.set_data(gt_batch_img)
        #     #     pred_win.set_data(pred_batch_img)
        #     # plt.pause(.05)
        #     # plt.draw()

        # hmap_fit_save_loc1 = 'face_datasets/300W_private_fittings_hmaps/' + fn1[:-4] + '.jpg'

        # hmap_fit_save_loc3 = 'face_datasets/300W_private_fittings_hmaps/' + fn1[:-4] + '_pt' + '.jpg'

        # hmap_fit_save_loc2 = 'face_datasets/300W_private_fittings_hmaps/' + fn1[:-4] + '_ft' + '.jpg'

        # #print('fname -> {}'.format(hmap_fit_save_loc))

        # if acc > 0.08:
            
        #     imsave(hmap_fit_save_loc1, pred_batch_img)
            
        #     imsave(hmap_fit_save_loc2, feat_batch_img)

            #mean=torch.Tensor([0.5, 0.5, 0.5])

            #inp = input[0]#.cpu() + mean.view(3, 1, 1).expand_as(input[0].cpu())



            #show_joints(inp.clamp(0, 1), preds[0]*0.25, hmap_fit_save_loc3)




        
        
        
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
    parser.add_argument('--num-classes', default=68, type=int, metavar='N',
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
    parser.add_argument('--test-batch', default=1, type=int, metavar='N',
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

    parser.add_argument('--state_dim', type=int, default=4096, help='GGNN hidden state size')
    parser.add_argument('--n_steps', type=int, default=3, help='propogation steps number of GGNN')
    parser.add_argument('--pairwise', action='store_true', help='enable pairwise connections')
    parser.add_argument('--symmetrical', action='store_true', help='enable symmetrical connections')

    main(parser.parse_args())
