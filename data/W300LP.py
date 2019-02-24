from __future__ import print_function

import os
import numpy as np
import random
import math
from skimage import io

import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua

# from utils.utils import *
#from utils.imutils import *
from utils.transforms import *
import matplotlib.pyplot as plt

from utils import imutils
from pylib import HumanPts, FaceAug, HumanAug, FacePts
#matplotlib.use('Agg')

class W300LP(data.Dataset):

    def __init__(self, img_folder, is_train=True, scale_factor=0.25, rot_factor=30):
        self.nParts = 68
        self.pointType = '3D'

        print('Point type --> {}'.format(self.pointType))
        
        self.img_folder = img_folder
        
        self.is_train = is_train
        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.mean, self.std = self._comput_mean()

    def _getDataFaces(self, is_train):
        base_dir = os.path.join(self.img_folder, 'landmarks')
        dirs = os.listdir(base_dir)
        lines = []
        vallines = []
        for d in dirs:
            files = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.mat')]
            for f in files:
                if f.find('test') == -1:
                    lines.append(f)
                else:
                    vallines.append(f)
        if is_train:
            print('=> loaded train set, {} images were found'.format(len(lines)))
            return lines
        else:
            print('=> loaded validation set, {} images were found'.format(len(vallines)))
            return vallines

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        inp, out, pts, c, s, tpts = self.generateSampleFace(index)
        self.pts, self.c, self.s = pts, c, s
        if self.is_train:
            return inp, out, tpts
        else:
            #meta = {'index': index, 'center': c, 'scale': s, 'pts': pts,}
            return inp, out, pts, index, c, s

    def generateSampleFace(self, idx):
        sf = self.scale_factor
        rf = self.rot_factor

        main_pts = load_lua(
            os.path.join(self.img_folder, 'landmarks_t7', self.anno[idx].split('_')[0],
                         self.anno[idx][:-4] + '.t7'))
        pts = main_pts[0] if self.pointType == '2D' else main_pts[1]

        #pts2 = main_pts[1]
        c = torch.Tensor((450 / 2, 450 / 2 + 50))
        s = 1.8

        img = load_image(os.path.join(self.img_folder, self.anno[idx].split('_')[0], self.anno[idx][:-8] +
                         '.jpg'))

        r = 0
        if self.is_train:
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='w300lp')
                c[0] = img.size(2) - c[0]

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = HumanAug.crop(imutils.im_to_numpy(img), c.numpy(), s, r, 256, 200)

        inp = imutils.im_to_torch(inp).float()

        pts_input_res = HumanAug.TransformPts(pts.numpy(), c.numpy(), s, r, 256, 200)
        pts_aug = pts_input_res * (1.*64/256)

        # Generate ground truth
        heatmap, pts_aug = HumanPts.pts2heatmap(pts_aug, [64, 64], sigma=1)
        heatmap = torch.from_numpy(heatmap).float()


        

        # inp = crop(img, c, s, [256, 256], rot=r)
        # # inp = color_normalize(inp, self.mean, self.std)

        # tpts = pts.clone()
        # out = torch.zeros(self.nParts, 64, 64)
        # for i in range(self.nParts):
        #     if tpts[i, 0] > 0:
        #         tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [64, 64], rot=r))
        #         out[i] = draw_labelmap(out[i], tpts[i] - 1, sigma=1)

        return inp, heatmap, pts, c, s, pts_input_res
        #return img, pts2, pts, c, s

    def _comput_mean(self):
        meanstd_file = './face_datasets/300W_LP_LFPW/mean.pth.tar'
        if os.path.isfile(meanstd_file):
            ms = torch.load(meanstd_file)
        else:
            print("\tcomputing mean and std for the first time, it may takes a while, drink a cup of coffe...")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            if self.is_train:
                for i in range(self.total):
                    a = self.anno[i]
                    img_path = os.path.join(self.img_folder, self.anno[i].split('_')[0],
                                            self.anno[i][:-8] + '.jpg')
                    img = load_image(img_path)
                    mean += img.view(img.size(0), -1).mean(1)
                    std += img.view(img.size(0), -1).std(1)

            mean /= self.total
            std /= self.total
            ms = {
                'mean': mean,
                'std': std,
            }
            torch.save(ms, meanstd_file)
        if self.is_train:
            print('\tMean: %.4f, %.4f, %.4f' % (ms['mean'][0], ms['mean'][1], ms['mean'][2]))
            print('\tStd:  %.4f, %.4f, %.4f' % (ms['std'][0], ms['std'][1], ms['std'][2]))
        return ms['mean'], ms['std']

# if __name__=="__main__":
#     import opts, demo
#     args = opts.argparser()
#     dataset = W300LP(args, 'test')
#     crop_win = None
#     for i in range(dataset.__len__()):
#         input, target, meta = dataset.__getitem__(i)
#         input = input.numpy().transpose(1,2,0) * 255.
#         target = target.numpy()

#         input = input.astype(np.uint8)
#         # if crop_win is None:
#         #     crop_win = plt.imshow(input)
#         # else:
#         #     crop_win.set_data(input)
#         #print(input)
    
#         pts1 = meta['pts'].numpy()


#         plt.figure(1)
        
#         plt.subplot(211)    
                
#         plt.imshow(input)

#         plt.scatter(pts1[:,0], pts1[:,1], s = 12)

#         plt.subplot(212)

#         plt.imshow(input)

#         plt.scatter(target[:,0], target[:,1], s = 12, c = 'r')

#         #plt.pause(0.5)
#         plt.show()
