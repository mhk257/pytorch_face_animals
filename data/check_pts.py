

from __future__ import print_function

import os
import numpy as np
import random
import math
from skimage import io
from scipy import io as sio


path_to_imgs = '/home/ubuntu/pytorch-pose/face_datasets/AFLW2000/'

files = [f for f in os.listdir(path_to_imgs) if f.endswith('.mat')]


for ff in range(0, len(files)):


    path_to_img = path_to_imgs + files[ff]
    
    main_pts = sio.loadmat(path_to_img)

    pts = main_pts['pt3d_68'][0:2, :].transpose()

    max_val = np.amax(pts)

    min_val = np.amin(pts)

    if ff == 875:

        
        print(pts)
        print(min_val)
        print(path_to_img)
        print(ff)

    
