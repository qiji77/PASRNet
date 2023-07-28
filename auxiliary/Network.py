from __future__ import print_function
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from my_utils import sampleSphere
import trimesh
import pointcloud_processor
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import sys
sys.path.append('./Tool_list/')
from Inverse_smpl import *
sys.path.append("./extension/")
sys.path.append("./ChamferDistancePytorch/")
import chamfer3D.dist_chamfer_3D

distChamfer = chamfer3D.dist_chamfer_3D.chamfer_3DDist()



edge_index=[[13],[2,4],[1,13],[5],[1],[3,1],[5,13],[8,10],[7,13],[11],[7],[9,12],[11,13],[0,2,6,8,12]]
class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500,output_dim=1024, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.output_dim=output_dim
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.output_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.output_dim)
        self.trans = trans


        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, self.output_dim)
        return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 10, 1)
        self.conv_1 = torch.nn.Conv1d(4,4, 1)
        self.conv_2 = torch.nn.Conv1d(3,3, 1)
        self.conv_3 = torch.nn.Conv1d(3,3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)
        self.sof=nn.Sigmoid()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x=self.conv4(x)
        x[:,[0,1,2,3],:] = self.th(self.conv_1(x[:,[0,1,2,3],:]))
        x[:,  [ 4, 5, 6],:] = self.th(self.conv_2(x[:, [4, 5, 6], :]))
        x[:,  [ 7, 8, 9],:] = 0.6*self.sof(self.conv_3(x[:, [7, 8, 9], :]))+0.8
        return x
class Feature_extract_Rec(nn.Module):
    def __init__(self, emb_dims=1024,  k=20, low_dim_idx=0):
        super(Feature_extract_Rec, self).__init__()
        self.emb_dims = emb_dims
        self.k = k
        self.low_dim_idx = low_dim_idx
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm1d(self.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6 , 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                   nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True),
                                   nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2, inplace = True))


    def knn(self, x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   
        return idx

    def get_graph_feature(self, x, k=20, idx=None, dim9=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if dim9 == False:
                idx = self.knn(x, k=k)   
            else:
                idx = self.knn(x[:, 6:], k=k)
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()   
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.z_dims,1).cuda()
        cuboid_vec = self.cuboid_vector.unsqueeze(0).repeat(num_samples,1,1)            # (batch_size, num_cuboid, num_cuboid)
        cuboid_vec = self.enc_cuboid_vec(cuboid_vec)                                    # (batch_size, 64, num_cuboid)

        x_cuboid = torch.cat((z.repeat(1,1,self.num_cuboid),cuboid_vec),dim=1)          # (batch_size, emb_dims + 64, num_cuboid)
        x_cuboid = self.conv_cuboid(x_cuboid)                                           # (batch_size, 128, num_points)  
        return x_cuboid
    
    def interpolation(self, z1, z2, num_samples):
        delta = (z2 - z1) / (num_samples-1)
        z = torch.zeros(num_samples,self.z_dims).cuda()
        for i in range(num_samples):
            if i == (num_samples - 1):
                z[i,:] = z2
            else:
                z[i,:] = z1 + delta * i
        z = z.unsqueeze(-1)
        cuboid_vec = self.cuboid_vector.unsqueeze(0).repeat(num_samples,1,1)            # (batch_size, num_cuboid, num_cuboid)
        cuboid_vec = self.enc_cuboid_vec(cuboid_vec)                                    # (batch_size, 64, num_cuboid)

        x_cuboid = torch.cat((z.repeat(1,1,self.num_cuboid),cuboid_vec),dim=1)          # (batch_size, emb_dims + 64, num_cuboid)
        x_cuboid = self.conv_cuboid(x_cuboid)                                           # (batch_size, 128, num_points)  
        return x_cuboid

    def forward(self, pc):
        batch_size = pc.size(0)
        x = pc.transpose(2, 1)
        idx = self.knn(x, k=self.k)
        x = self.get_graph_feature(x, k=self.k, idx = idx if self.low_dim_idx == 1 else None)    
        x = self.conv1(x)                                                                        
        x1 = x.max(dim=-1, keepdim=False)[0]                                                     
        x = self.get_graph_feature(x1, k=self.k ,idx = idx if self.low_dim_idx == 1 else None)  
        x = self.conv2(x)                                                                        
        x2 = x.max(dim=-1, keepdim=False)[0]                                                     

        x_per = torch.cat((x1, x2), dim=1)                 


        return x_per

class PointGenCon_shape(nn.Module):
    def __init__(self, bottleneck_size=2048):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon_shape, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//4, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//4, self.bottleneck_size//8, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//8, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//8)
        self.sof=nn.Sigmoid()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x=0.5*self.th(self.conv4(x))
        return x

class GCN_Our(nn.Module):
    def __init__(self):
        super(GCN_Our, self).__init__()
        self.conv1 = torch.nn.Conv1d(128, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 256, 1)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2=torch.nn.BatchNorm1d(256)
    def forward(self, x,adj):
        
        x=torch.bmm(x,adj)
        x = F.relu(self.bn1(self.conv1(x)))
        x=torch.bmm(x,adj)
        x=F.relu(self.bn2(self.conv2(x)))
        return x

class GCN_Our2(nn.Module):
    def __init__(self):
        super(GCN_Our2, self).__init__()
        self.conv1 = torch.nn.Conv1d(512, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2=torch.nn.BatchNorm1d(512)
    def forward(self, x,adj):
        x=torch.bmm(x,adj)
        x = F.relu(self.bn1(self.conv1(x)))
        x=torch.bmm(x,adj)
        x=F.relu(self.bn2(self.conv2(x)))
        return x

class GCN_Our3(nn.Module):
    def __init__(self):
        super(GCN_Our3, self).__init__()
        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2=torch.nn.BatchNorm1d(1024)
    def forward(self, x,adj):
        x=torch.bmm(x,adj)
        x = F.relu(self.bn1(self.conv1(x)))
        x=torch.bmm(x,adj)
        x=F.relu(self.bn2(self.conv2(x)))
        return x

class GCN_shape(nn.Module):
    def __init__(self):
        super(GCN_shape, self).__init__()
        self.conv1 = torch.nn.Conv1d(128, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)
        self.conv4=torch.nn.Conv1d(1024, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2=torch.nn.BatchNorm1d(512)
        self.bn3=torch.nn.BatchNorm1d(1024)
        self.bn4=torch.nn.BatchNorm1d(1024)
    def forward(self, x,adj):
        x=torch.bmm(x,adj)
        x = F.relu(self.bn1(self.conv1(x)))
        x=torch.bmm(x,adj)
        x=F.relu(self.bn2(self.conv2(x)))
        x=torch.bmm(x,adj)
        x=F.relu(self.bn3(self.conv3(x)))
        x=torch.bmm(x,adj)
        x=F.relu(self.bn4(self.conv4(x)))
        return x

class PASRNet(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(PASRNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = PointNetfeat(output_dim=256)          
        self.encoder_shape = PointNetfeat(output_dim=1024)   
        self.mlp512 = nn.Sequential(
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.ReLU()
        )
        self.mlp1024 = nn.Sequential(
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU()
        )
        self.encoder2=GCN_Our()
        self.encoder3=GCN_Our2()
        self.encoder4=GCN_Our3()
        self.GCN_tem=GCN_shape()

        self.encoder_rec=Feature_extract_Rec()

        self.decoder_pose = PointGenCon(bottleneck_size = 2*self.bottleneck_size)

        self.decoder_shape = PointGenCon_shape(bottleneck_size = 2*self.bottleneck_size)

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        point_set = mesh.vertices
        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.prop = pointcloud_processor.get_vertex_normalised_area(mesh)
        assert (np.abs(np.sum(self.prop) - 1) < 0.001), "Propabilities do not sum to 1!)"
        self.prop = torch.from_numpy(self.prop).cuda().unsqueeze(0).float()

        self.rig=self.getrig()
        self.rig=self.rig.unsqueeze(0).expand(1,self.rig.size(0),self.rig.size(1)).contiguous()#B,128,14

        self.adj=self.getadj()  
        self.adj=self.adj.unsqueeze(0).expand(1,self.adj.size(0),self.adj.size(1))
        self.adj=self.adj.transpose(2,1)
        self.temadj=self.gettemadj()

        
    def norm(self,adj):
        adj += np.eye(adj.shape[0]) 
        degree = np.array(adj.sum(1)) 
        degree = np.diag(np.power(degree, -0.5))
        return degree.dot(adj).dot(degree)
    
    def gettemadj(self):
        adjmat=torch.zeros([6890,6890],dtype=torch.float)
        for face_temp in  self.mesh.faces:
            A=face_temp[0]
            B=face_temp[1]
            C=face_temp[2]
            adjmat[A,B]=1
            adjmat[B,A]=1
            adjmat[A,C]=1
            adjmat[C,A]=1
            adjmat[C,B]=1
            adjmat[B,C]=1
        adj = self.norm(adjmat)
        adj = torch.tensor(adj, dtype=torch.float).cuda()
        return adj

    def getadj(self): 
        adjmat = np.zeros((14, 14)) 
        for edge_num in range(14):
            edge=edge_index[edge_num]
            for idx in edge:
                adjmat[edge_num][idx]=1
                adjmat[idx][edge_num]=1
        adj = self.norm(adjmat)
        adj = torch.tensor(adj, dtype=torch.float).cuda()
        return adj
    
    def getrig(self):
        rig_txt=np.load("rig14_txt.npy")
        rig_temp=torch.from_numpy(rig_txt)
        rig_temp=rig_temp.cuda()
        return rig_temp.transpose(1,0)
    def getInverposemodel(self,Input,Rota):

        InverseTem=Gen_InversePose(Rota,Input)
        return InverseTem
    def getInverposemodel_end(self,Inverse_temp,Pamt,inverse_label_index):
        InverseTem=RotaList_inverse(Inverse_temp,Pamt,inverse_label_index)
        return InverseTem
    def getRota(self,stmodel,Pamt):
        Rota = Pamt[:, :, [0, 1, 2, 3]]
        roat = Rota * Rota
        roat = roat.sum(2).unsqueeze(2)
        Rota_aq = torch.sqrt(roat)
        Rota = Rota / Rota_aq
        Rota_list=GetRota_List(stmodel,Rota)
        return Rota_list
    def getSTmodel(self,Pamt):              
        Tran = Pamt[:, 13, [4, 5, 6]]
        Scale = Pamt[:, :, [7,8,9]]
        template_points = self.vertex.clone()
        template_points = template_points.unsqueeze(0).expand(Pamt.size(0), template_points.size(0), template_points.size(1)) 
        template_points = Variable(template_points, requires_grad=False)
        template_points = template_points.cuda()
        template_points_temp = template_points.clone().detach()
        T=Tran
        T = T.unsqueeze(1)
        pointsReconstructed_edge=Scalelist(template_points_temp,Scale,0) + T
        return pointsReconstructed_edge

    def encoder_pasr(self,x):
        rig=self.rig.expand(x.size(0),self.rig.size(1),self.rig.size(2))
        adj=self.adj.expand(x.size(0),self.adj.size(1),self.adj.size(2))
        
        X_Fea_temp=self.encoder2(rig,adj) #B，256，14
        input_Fea = self.encoder(x)#B，256
        input_Fea256=input_Fea
        y = input_Fea256.unsqueeze(2).expand(input_Fea256.size(0),input_Fea256.size(1), X_Fea_temp.size(2)).contiguous()

        y = torch.cat((X_Fea_temp, y), 1).contiguous() #512，14
        X_Fea_temp2=self.encoder3(y,adj)
        input_Fea512=self.mlp512(input_Fea256)
        y = input_Fea512.unsqueeze(2).expand(input_Fea512.size(0),input_Fea512.size(1), X_Fea_temp2.size(2)).contiguous()
        y = torch.cat( (X_Fea_temp2, y), 1).contiguous() 
        X_Fea_temp3=self.encoder4(y,adj)
        input_Fea1024=self.mlp1024(input_Fea512)
        y = input_Fea1024.unsqueeze(2).expand(input_Fea1024.size(0),input_Fea1024.size(1), X_Fea_temp3.size(2)).contiguous()
        y = torch.cat((X_Fea_temp3, y), 1)
        pose_parameter=self.decoder_pose(y)
        st_template=self.getSTmodel(pose_parameter.transpose(2,1))
        Rota_list=self.getRota(st_template,pose_parameter.transpose(2,1))
        
        st_template_clone=st_template.clone().detach()
        pose_model=RotaList_eachpart(st_template,Rota_list)

        #shape
        tem_adj=self.temadj.unsqueeze(0).expand(x.size(0),self.temadj.size(0),self.temadj.size(1))
        pose_Fea=torch.ones((pose_model.size(0),128,pose_model.size(1)),device='cuda')

        for rig_n in range(14):
            x_per=self.encoder_rec(pose_model[:,Rig_list[rig_n]])
            pose_Fea[:,:,Rig_list[rig_n]]=x_per
        pose_Fea=self.GCN_tem(pose_Fea,tem_adj)

        F_input=self.encoder_shape(x)

        F_input=F_input.unsqueeze(-1).expand(F_input.size(0),F_input.size(1),pose_Fea.size(2))
        shape_fea=torch.cat((pose_Fea,F_input),dim=1)
        
        return y,shape_fea

    def forward(self, x,poseif):
        rig=self.rig.expand(x.size(0),self.rig.size(1),self.rig.size(2))
        adj=self.adj.expand(x.size(0),self.adj.size(1),self.adj.size(2))
        
        X_Fea_temp=self.encoder2(rig,adj) #B，256，14
        input_Fea = self.encoder(x)#B，256
        input_Fea256=input_Fea
        y = input_Fea256.unsqueeze(2).expand(input_Fea256.size(0),input_Fea256.size(1), X_Fea_temp.size(2)).contiguous()

        y = torch.cat((X_Fea_temp, y), 1).contiguous() #512，14
        X_Fea_temp2=self.encoder3(y,adj)
        input_Fea512=self.mlp512(input_Fea256)
        y = input_Fea512.unsqueeze(2).expand(input_Fea512.size(0),input_Fea512.size(1), X_Fea_temp2.size(2)).contiguous()
        y = torch.cat( (X_Fea_temp2, y), 1).contiguous() #1024，14
        X_Fea_temp3=self.encoder4(y,adj)#B,1024,14
        input_Fea1024=self.mlp1024(input_Fea512)
        y = input_Fea1024.unsqueeze(2).expand(input_Fea1024.size(0),input_Fea1024.size(1), X_Fea_temp3.size(2)).contiguous()
        y = torch.cat((X_Fea_temp3, y), 1)
        pose_parameter=self.decoder_pose(y)
        st_template=self.getSTmodel(pose_parameter.transpose(2,1))
        Rota_list=self.getRota(st_template,pose_parameter.transpose(2,1))
        
        st_template_clone=st_template.clone().detach()
        if poseif:
            pose_model=RotaList_eachpart(st_template,Rota_list)
            return pose_model


        #shape
        pose_model=RotaList_eachpart(st_template,Rota_list)
        tem_adj=self.temadj.unsqueeze(0).expand(x.size(0),self.temadj.size(0),self.temadj.size(1))
        pose_Fea=torch.ones((pose_model.size(0),128,pose_model.size(1)),device='cuda')

        for rig_n in range(14):
            x_per=self.encoder_rec(pose_model[:,Rig_list[rig_n]])
            pose_Fea[:,:,Rig_list[rig_n]]=x_per
        pose_Fea=self.GCN_tem(pose_Fea,tem_adj)
        #
        F_input=self.encoder_shape(x)
        #
        F_input=F_input.unsqueeze(-1).expand(F_input.size(0),F_input.size(1),pose_Fea.size(2))
        shape_fea=torch.cat((pose_Fea,F_input),dim=1)
        shape_add=self.decoder_shape(shape_fea)

        return pose_model,shape_add.transpose(2,1)

    def decode(self,pose_fea,shape_fea,x):
        pose_parameter=self.decoder_pose(pose_fea)
        st_template=self.getSTmodel(pose_parameter.transpose(2,1))
        Rota_list=self.getRota(st_template,pose_parameter.transpose(2,1))
        InverseTem14=self.getInverposemodel(Rota_list,x.clone().detach())
        inverse_dis_list=torch.zeros((InverseTem14.size(0),InverseTem14.size(1),InverseTem14.size(2))).cuda()
        for part_label_id in range(14):
            dist1, dist2, idx1, idx2 = distChamfer(InverseTem14[:,part_label_id], st_template[:,Rig_list[part_label_id]])
            #dist1=dist1/partscale[part_label_id]
            inverse_dis_list[:,part_label_id]=dist1  
        _,inverse_label_index=torch.min(inverse_dis_list,dim=1)
        Inverse_temp=x.clone().detach()
        Inverse_temp=Variable(Inverse_temp, requires_grad=False)
        Inverse_temp_end=self.getInverposemodel_end(Inverse_temp,Rota_list,inverse_label_index)
        pose_model=RotaList_eachpart(st_template,Rota_list)
        shape_add=self.decoder_shape(shape_fea)
        
        return pose_model,shape_add.transpose(2,1),inverse_label_index,Inverse_temp_end,st_template



if __name__ == '__main__':
   
    print('testing PointSenGet...')
    sim_data = Variable(torch.rand(1, 4, 192, 256))
    model = Hourglass()
    out = model(sim_data)
    print(out.size())
