from __future__ import print_function
import sys
sys.path.append('./auxiliary/')
import my_utils
seed = 1024 
my_utils.plant_seeds(randomized_seed=False,manualSeed=seed)
import argparse
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.optim as optim

from dataset import *
from Network import *
from ply import *
import json
import datetime
from LaplacianLoss import *
from knn_cuda import KNN

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
sys.path.append('./Tool_list/')
import Visual
writer = SummaryWriter()


# import testedge
# =============PARAMETERS======================================== #
lambda_laplace = 0.001
lambda_ratio = 0.005

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--model_point', type=str, default='', help='optional reload model path')
parser.add_argument('--model_edge', type=str, default='', help='optional reload model path')
parser.add_argument('--env', type=str, default="PASRNet", help='visdom environment')
parser.add_argument('--laplace', type=int, default=1, help='regularize towords 0 curvature, or template curvature')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
sys.path.append("./extension/")
sys.path.append("./ChamferDistancePytorch/")
import chamfer3D.dist_chamfer_3D

distChamfer = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

seedname = os.path.join(dir_name, 'seed.txt')
with open(seedname, 'w') as f:
    f.write(str(seed))

# meters to record stats on learning
# ========================================================== #


# ===================CREATE DATASET================================= #
dataset = SURREAL(train=True, regular_sampling=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)
dataset_smpl_test = SURREAL(train=False)
dataloader_smpl_test = torch.utils.data.DataLoader(dataset_smpl_test, batch_size=opt.batchSize,
                                                   shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #

network_shape=PASRNet()

faces = network_shape.mesh.faces
faces = [faces for i in range(opt.batchSize)]
faces = np.array(faces)
faces = torch.from_numpy(faces).cuda()
# takes cuda torch variable repeated batch time

vertices = network_shape.mesh.vertices
vertices = [vertices for i in range(opt.batchSize)]
vertices = np.array(vertices)
vertices = torch.from_numpy(vertices).cuda()
toref = opt.laplace  
# Initialize Laplacian Loss
laplaceloss = LaplacianLoss(faces, vertices, toref)

laplaceloss(vertices)

network_shape.cuda()
network_shape.apply(my_utils.weights_init)


if opt.model != '':
    network_shape.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

# ===================CREATE optimizer================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network_shape.parameters(), lr=lrate)

with open(logname, 'a') as f:  # open and append
    f.write(str(network_shape) + '\n')

# ========================================================== #
def init_regul(source):
    sommet_A_source = source.vertices[source.faces[:, 0]]
    sommet_B_source = source.vertices[source.faces[:, 1]]
    sommet_C_source = source.vertices[source.faces[:, 2]]
    target = []
    target.append(np.sqrt(np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt(np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt(np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target


target = init_regul(network_shape.mesh)
target = np.array(target)
target = torch.from_numpy(target).float().cuda()
target = target.unsqueeze(1).expand(3, opt.batchSize, -1)

def compute_score(points, faces, points_pose):
    score = 0
    sommet_A = points[:, faces[:, 0]]
    sommet_B = points[:, faces[:, 1]]
    sommet_C = points[:, faces[:, 2]]
    
    sommet_A_pose = points_pose[:, faces[:, 0]]
    sommet_B_pose = points_pose[:, faces[:, 1]]
    sommet_C_pose = points_pose[:, faces[:, 2]]


    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / torch.sqrt(torch.sum((sommet_A_pose - sommet_B_pose) ** 2, dim=2)) - 1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) /  torch.sqrt(torch.sum((sommet_B_pose - sommet_C_pose) ** 2, dim=2)) - 1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / torch.sqrt(torch.sum((sommet_A_pose - sommet_C_pose) ** 2, dim=2)) - 1)
    return torch.mean(score)

# ========================================================== #

# Load all the points from the template
template_points = network_shape.vertex.clone()
template_points = template_points.unsqueeze(0).expand(opt.batchSize, template_points.size(0), template_points.size(
    1)) 
template_points = Variable(template_points, requires_grad=False)
template_points = template_points.cuda()


# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
    # optimizer_gen=optim.Adam(GenNetwork.parameters(),lr=0.001)
    if epoch == 360: #40
        lrate = lrate / 10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network_shape.parameters(), lr=lrate) 
    # TRAIN MODE
    network_shape.train()
    if epoch ==0:
        # initialize reconstruction to be same as template to avoid symmetry issues
        init_step = 0
        for i, data in enumerate(dataloader, 0):
            if (init_step > 100):
                break
            init_step = init_step + 1
            optimizer.zero_grad()
            points, fn, idx, _ ,_= data
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            PoseModel,points_add  = network_shape(points,0)  # forward pass
            Pointend=PoseModel+points_add
            loss_net = torch.mean((PoseModel - template_points) ** 2)+torch.mean((Pointend - PoseModel) ** 2)
            loss_net.backward()
            optimizer.step()  # gradient update
            print('init [%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))

    for i, data in enumerate(dataloader, 0):
        points, fn, idx, _ ,_= data
        points = points.transpose(2, 1).contiguous()
        points = points.cuda()
        
        if epoch<200:
            PoseModel  = network_shape(points,1)
            points= points.transpose(2, 1)
            optimizer.zero_grad()
            dist1_pose,dist2_pose,_,_=distChamfer(points.contiguous(), PoseModel)
            chamfer_pose=torch.mean(dist1_pose)+torch.mean(dist2_pose)
            regul = laplaceloss(PoseModel)
            edge_pose=lambda_ratio* compute_score(PoseModel, network_shape.mesh.faces, template_points)
            dist_end=chamfer_pose+edge_pose+lambda_ratio*regul
        else:
            PoseModel,points_add = network_shape(points,0)
            
            points= points.transpose(2, 1)
            optimizer.zero_grad()
            Pointend=PoseModel+points_add
            regul = laplaceloss(Pointend)
            dist1_pose,dist2_pose,_,_=distChamfer(points.contiguous(), PoseModel)
            dist1, dist2, _, _ = distChamfer(points.contiguous(), Pointend)
            chamfer_pose=torch.mean(dist1_pose)+torch.mean(dist2_pose)
            chamfer_shape=torch.mean(dist1)+torch.mean(dist2)            
            edge_pose=0.3*lambda_ratio* compute_score(PoseModel, network_shape.mesh.faces, template_points)
            shape_lap=lambda_laplace * regul
            edge_shape=0.01*lambda_ratio* compute_score(Pointend, network_shape.mesh.faces, template_points)
            dist_end=chamfer_pose+chamfer_shape+shape_lap+edge_shape+edge_pose

  
        loss_net=dist_end
        loss_net.backward()
        optimizer.step()  # gradient update
        print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))

    with torch.no_grad():
        # val on SMPL data
        network_shape.eval()
        for i, data in enumerate(dataloader_smpl_test, 0):
            points, fn, idx, _ = data
            
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            if epoch<200:
                Pointend = network_shape(points,1)
            else:
                PoseModel,points_add = network_shape(points,0)
                Pointend=PoseModel+points_add
            loss_net = torch.mean(
                (Pointend - points.transpose(2, 1).contiguous()) ** 2)
            print('[%d: %d/%d] test smlp loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))

        # save latest network
        torch.save(network_shape.state_dict(), '%s/network_lastshape.pth' % (dir_name))
