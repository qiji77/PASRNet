from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import trimesh
import pointcloud_processor
import time
import os
import sys
sys.path.append('./auxiliary/')
sys.path.append('./')
import my_utils

def unwrap_self(arg, **kwarg):
    return arg[0]._getitem(*(arg[1:]), **kwarg)


class SURREAL(data.Dataset):
    def __init__(self, train, npoints=2500, regular_sampling=False, normal=False, data_augmentation_Z_rotation=False,
                 data_augmentation_Z_rotation_range=360, data_augmentation_3D_rotation=False):

        self.normal = normal
        self.train = train
        self.regular_sampling = regular_sampling  # sample points uniformly or proportionaly to their adjacent area
        self.npoints = npoints
        self.mesh_ref = trimesh.load("./data/template/template.ply", process=False)
        self.datas = []
        start = time.time()
        self.datas=self.getvoldata()



    def __getitem__(self, index):
        # LOAD a training sample
        points = self.datas[index].squeeze()
        # Clone it to keep the cached data safe
        points = points.clone()
        rot_matrix = 0
        random_sample = 0
        points_sesup=0
        if self.train:
            random_sample = np.random.choice(6890, size=6890,replace=False)
            points = points[random_sample]
            return points, random_sample, rot_matrix, index,points_sesup
        else:
            return points, random_sample, rot_matrix, index#,points


    def __len__(self):
        if self.train:
            return self.datas.size(0)
        else:
            return 256
    def getvoldata(self):
        datalist=[]
        if self.train:
            datalist=np.load("./Dataset/CAPE/CAPE_train.npy")     
            datalist=torch.Tensor(datalist)
            print("CAPE train",datalist.size())

        else:
            datalist=np.load("./Dataset/CAPE/CAPE_test.npy")     
            datalist=torch.Tensor(datalist)
            print("CAPE test",datalist.size())
        datalist=torch.Tensor(datalist)
        print("datalist",datalist.size())
        return datalist
    
if __name__ == '__main__':
    import random

    manualSeed = 1  # random.randint(1, 10000)  # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    print('Testing Shapenet dataset')
    d = SURREAL(train=True, regular_sampling=True, data_augmentation_3D_rotation=False)
    a, b, c, d = d[0]
    print(a, b, c, d)
    min_vals = torch.min(a, 0)[0]
    max_vals = torch.max(a, 0)[0]
    print(min_vals)
    print(max_vals)
