# -*- coding: utf-8 -*-
"""
Created on 2025-03-24 18:50:58

@author: Robbie
"""
import numpy as np
from functools import lru_cache
from pyvisita import non_rigid_icp
import open3d as o3d #http://www.open3d.org/docs/release/tutorial/Basic/mesh.html
import scipy.io as sio
#https://anaconda.org/conda-forge/suitesparse


def face_correction(vertices_frontal,vertices_sidel,threshold=0.05,radius_feature=0.01,
                    nrom_frontal=None,nrom_sidel=None,method='norm_icp',flag_rotmatrix=False):
    #Num_Max=max([vertices_frontal.shape[0],vertices_sidel.shape[0]])
    #Num_Min=min([vertices_frontal.shape[0],vertices_sidel.shape[0]])
    #index = np.random.choice(np.arange(Num_Max), size=Num_Min, replace=False)
    
    pcd_frontal =o3d.geometry.PointCloud()#o3d.geometry.PointCloud()#
    pcd_frontal.points = o3d.utility.Vector3dVector(vertices_frontal)
    if nrom_frontal is not None:
        pcd_frontal.normals = o3d.utility.Vector3dVector(nrom_frontal)
    
    pcd_sidel = o3d.geometry.PointCloud()#o3d.geometry.PointCloud()#
    pcd_sidel.points = o3d.utility.Vector3dVector(vertices_sidel)
    if nrom_sidel is not None:
        pcd_sidel.normals  = o3d.utility.Vector3dVector(nrom_sidel)
    
    trans_init = np.asarray([[1,0,0,0], # 4x4 identity matrix，这是一个转换矩阵，
                             [0,1,0,0], # 象征着没有任何位移，没有任何旋转，我们输入
                             [0,0,1,0], # 这个矩阵为初始变换
                             [0,0,0,1]])
    if method=='norm_icp':  
        threshold = threshold#005#0.005
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd_sidel,
                                                pcd_frontal,
                                                threshold, trans_init)#,
                                                #o3d.cpu.pybind.pipelines.registration.TransformationEstimationPointToPoint(),
                                                #o3d.cpu.pybind.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000))
    elif method=='feature_icp': #not work now 
         #o3d.visualization.draw_geometries([pcd_sidel])
         radius_feature=0.01
         pcd_frontal_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_frontal, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
         pcd_sidel_fpfh =  o3d.pipelines.registration.compute_fpfh_feature(pcd_sidel, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=500))
         
         threshold = threshold#0.005
         reg_p2p =  o3d.pipelines.registration.registration_fast_based_on_feature_matching(pcd_sidel,
                                                                                 pcd_frontal,
                                                                                 pcd_sidel_fpfh,
                                                                                 pcd_frontal_fpfh,
                                                                                 o3d.registration.FastGlobalRegistrationOption(
                                                                                 maximum_correspondence_distance=threshold))
    print(reg_p2p.transformation)
    #source = o3d.io.read_point_cloud("./FLORENCE/subject_12/Model/frontal1/ply/110112131702.ply") #source 为需要配准的点云
    vertices_sidelT=(np.c_[vertices_sidel,np.zeros((len(vertices_sidel),1))]).dot(reg_p2p.transformation.T)
    
    if flag_rotmatrix:
        return vertices_sidelT[:,0:3],reg_p2p.transformation.T
    else:
        return  vertices_sidelT[:,0:3]


def smootherstep(x, x_min, x_max):
    y = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return 6 * (y ** 5) - 15 * (y ** 4) + 10 * (y ** 3)


def generate_data_weights(template, nosetip, ear_idx,r_mid=0.075, r_width=0.15,
                          y_pen=1.4, w_inner=1, w_outer=0):
    r_min = r_mid - (r_width / 2)
    r_max = r_mid + (r_width / 2)
    w_range = w_inner - w_outer
    x = np.sqrt(np.sum((template - nosetip) ** 2 *
                       np.array([1, y_pen, 1]), axis=1))
    weight=((1 - smootherstep(x, r_min, r_max))[:, None] * w_range + w_outer).T
    #print(weight.shape)
    weight[:,ear_idx]=np.median(weight,1).reshape(-1,1)
    return weight


def generate_data_weights_per_iter(template, nosetip, ear_idx,r_width, w_min_iter,
                                   w_max_iter, r_mid=0.075, y_pen=1.4):
    # Change in the data term follows the same pattern that is used for the
    # stiffness weights
    stiffness_weights = np.array([50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2])
    s_iter_range = stiffness_weights[0] - stiffness_weights[-1]
    w_iter_range = w_max_iter - w_min_iter
    m = w_iter_range / s_iter_range
    c = w_max_iter - m * stiffness_weights[0]
    w_outer = m * stiffness_weights + c
    w_inner = 1
    return generate_data_weights(template, nosetip, ear_idx,w_inner=w_inner,
                                 w_outer=w_outer, r_width=r_width, r_mid=r_mid,
                                 y_pen=y_pen)

@lru_cache()
def load_template():
    C = sio.loadmat('./model_data/BFM_2009_Model/BFM.mat')
    S_vertices =((np.reshape(C['model']['shapeMU'][0,0],(-1,3)))/1.0e6)
    Nose_idx=np.argmax(S_vertices[:,2])
    nosetip=np.repeat(S_vertices[Nose_idx,:].reshape(1,3),S_vertices.shape[0],0)
    
    regions_idx=np.loadtxt('./model_data/BFM_2009_Model/face_Region_idx.txt').astype(int)
    #column- 0: inner face; 1: nose; 2:eye, 3: mouth; 4: inner eye,5:ear
    ear_idx=np.where(regions_idx[:,5]==1)[0]
    return S_vertices,nosetip,ear_idx

@lru_cache()
def data_Faceweights_generate():
    w_max_iter = 0.5
    w_min_iter = 0.0
    
    #ply
    r_width = 0.18058   
    r_mid =   0.18058/2
    
    #obj
    #r_width = 0.18058/3*2
    #r_mid = 0.18058  
    
    y_pen =1.7
    template,nosetip,ear_idx= load_template()
    return generate_data_weights_per_iter(template,
                                          nosetip,#template.landmarks['nosetip'],
                                          ear_idx,
                                          r_width=r_width,
                                          r_mid=r_mid,
                                          w_min_iter=w_min_iter,
                                          w_max_iter=w_max_iter,
                                          y_pen=y_pen)                                                          
    


def data_Headweights_generate():
    num_iterate=8
    #BFM2019_HeadReg_idx=np.load("./model_data/BFM_2019_Model/BFM2019_HeadRegion_idx.npy") 
    # head_regionidx=np.loadtxt("./model_data/BFM_2019_Model/BFM2019_HeadRegion_idx.txt").astype(int)
    # Head_idx=np.load("./model_data/SymHead_Chinese_Model/Head_Regions_idx.npy")
    # BFM2019_HeadReg_idx=np.c_[Head_idx,head_regionidx]
    # np.save("./model_data/SymHead_Chinese_Model/BFM_2019_SymmetricalHead_Regionidx.npy",BFM2019_HeadReg_idx)
    BFM2019_HeadReg_idx=np.load("./model_data/SymHead_Chinese_Model/SymHead_RegisterRegion_idx.npy")
    #back head, hair regions (Parth/Fiona),top head
    
    num_points=BFM2019_HeadReg_idx.shape[0]#56804
    hair_fullidx=np.where(BFM2019_HeadReg_idx[:,1]==1)[0] #hair regions
    #tophead_idx=np.where(BFM2019_HeadReg_idx[:,2]==1)[0]  #top head 
    #idx=np.r_[tophead_idx,hair_fullidx]
    back_idx=np.where(BFM2019_HeadReg_idx[:,0]==0)[0]  #back head
    
    data_weights=np.ones((num_points,num_iterate))
    data_weights[hair_fullidx,:]=0.2
    
    #Fiona,Parth
    back_speed=np.linspace(1,4,num_iterate)
    hair_top_speed=np.linspace(1,8,num_iterate)
    # start_w1=5
    # start_w2=40
    # back_speed=np.linspace(start_w1,start_w1+num_iterate,num_iterate)
    # hair_top_speed=np.linspace(start_w2,start_w2+num_iterate,num_iterate)
    for i in range(num_iterate):
        data_weights[back_idx,i]=data_weights[back_idx,i]/back_speed[i]
        data_weights[hair_fullidx,i]=data_weights[hair_fullidx,i]/hair_top_speed[i]
    return data_weights.T


def correspond_mesh(source,target,group,
                    landmark_weights = [10, 5, 2, 0, 0, 0., 0., 0.],
                    stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2],
                    PerNrom_type=None,Para_Nrom=None,
                    data_weights=None,
                    flag_data_weights=True,
                    datatype="face",
                    verbose=False):
    #landmark_weights = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05]
    if flag_data_weights:
        if data_weights is None:
            print("data_weights_generate is used")
            if datatype=="face":
                Para_Nrom=0.3 if Para_Nrom is None else Para_Nrom

                aligned = non_rigid_icp(source, target, landmark_group=group,
                                landmark_weights=landmark_weights,
                                stiffness_weights=stiffness_weights,
                                data_weights=data_Faceweights_generate(), 
                                verbose=verbose,
                                PerNrom_type=PerNrom_type,Para_Nrom=Para_Nrom)
            elif datatype=="head":
                aligned = non_rigid_icp(source, target, landmark_group=group,
                            landmark_weights=landmark_weights,
                            stiffness_weights=stiffness_weights,
                            data_weights=data_Headweights_generate(), 
                            verbose=verbose,
                            PerNrom_type=PerNrom_type,Para_Nrom=Para_Nrom)
            
        else:
            aligned = non_rigid_icp(source, target, landmark_group=group,
                            landmark_weights=landmark_weights,
                            stiffness_weights=stiffness_weights,
                            data_weights=data_weights, 
                            verbose=verbose,
                            PerNrom_type=PerNrom_type,Para_Nrom=Para_Nrom)
                
    else:
        aligned = non_rigid_icp(source, target, landmark_group=group,
                            landmark_weights=landmark_weights,
                            stiffness_weights=stiffness_weights,
                            verbose=verbose,
                            PerNrom_type=PerNrom_type,Para_Nrom=Para_Nrom)   
    return aligned
