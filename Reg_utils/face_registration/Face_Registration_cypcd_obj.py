#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 21:57:36 2020

@author: peter
"""
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pycpd import RigidRegistration,AffineRegistration,gaussian_kernel, DeformableRegistration
from probreg import cpd,filterreg,bcpd #https://pypi.org/project/probreg/
import copy
# https://github.com/siavashk/pycpd/blob/master/testing/rigid_test.py

import numpy as np
import open3d as o3d #http://www.open3d.org/docs/release/tutorial/Basic/mesh.html
from skimage import io
import scipy.io as sio
import matplotlib.pyplot as plt
from menpo.shape import TriMesh,PointCloud
from scipy.spatial import Delaunay
from pycpd import RigidRegistration

from Rec_utils.visualize import plot_mlabvertex,plot_2mlabvertex,plot_mlabfaceerror
from Rec_utils.plyio import read_ply, save_ply
from Rec_utils.transform import angle2matrix
from Rec_utils.mesh import crop_mesh
from Reg_utils.face_corespond import face_correction,correspond_mesh
from Rec_utils.landmark import landmark3d_detect
from Reg_utils.uvmap_processing import Vertices2Mapuv
from Rec_utils.fitting import fit_shaperror
#===================================================Parameters===================================

Threshold_rnose=0.17*1000
Theshold_dtriangles=0.005

#===============================================Source Face============================================
#C = sio.loadmat('./model_data/BFM_Children_Model/BFM_Children.mat')

S_vertices,S_colors,S_triangles= read_ply('./model_data/BFM_Children_Model/BFM_Children_mean_L1.ply')

# S_vertices0,S_colors0,S_triangles0= read_ply('./model_data/BFM_Children_Model/BFM_Children_mean.ply')
# landmark_idx0=idx=np.array([21873, 22149, 21653, 21036, 43236, 44918, 46166, 47135, 47914,
#                               48695, 49667, 50924, 52613, 33678, 33005, 32469, 32709, 38695,
#                             39392, 39782, 39987, 40154, 40893, 41059, 41267, 41661, 42367,
#                               8161,  8177,  8187,  8192,  6515,  7243,  8204,  9163,  9883,
#                               2215,  3886,  4920,  5828,  4801,  3640, 10455, 11353, 12383,
#                               14066, 12653, 11492,  5522,  6025,  7495,  8215,  8935, 10395,
#                               10795,  9555,  8836,  8236,  7636,  6915,  5909,  7384,  8223,
#                               9064, 10537,  8829,  8229,  7629])#bfm.model['kpt_ind']

# landmark_earidx=[np.array([19790, 19225, 20038]), np.array([34776, 35523, 34490])]

# landmark_idx0=np.r_[landmark_idx0,landmark_earidx[0],landmark_earidx[1]]

# landmark_idx=np.array([ ((S_vertices-x.reshape(1,3))**2).sum(1).argmin() for x in  S_vertices0[landmark_idx0,:] ])
landmark_idx=np.array([2976, 2993, 2973, 2845, 4596, 4676, 4763, 4870, 5034, 5176, 5312,
                    5391, 5505, 3464, 3334, 3234, 3237, 4096, 4146, 4191, 4225, 4271,
                    4381, 4421, 4454, 4504, 3120, 1072, 1076, 1079, 1083,  676,  907,
                    1164, 1370, 1495,  104,  264,  391,  509,  351,  230, 1619, 1778,
                    1920, 2048, 1923, 1802,  500,  648,  874, 1130, 1259, 1666, 1715,
                    1424, 1269, 1139,  950,  773,  531,  913, 1097, 1293, 1584, 1263,
                    1100,  915, 2592, 2492, 2661, 3588, 3772, 3562])

S_vertices=S_vertices/1000
plot_mlabvertex(S_vertices,S_colors,S_triangles,S_vertices[landmark_idx,:])

source_points =o3d.geometry.PointCloud()#o3d.geometry.PointCloud()#
source_points.points = o3d.utility.Vector3dVector(S_vertices)

#===============================================target Face============================================
#filepath="/media/peter/Data/Children_Face_Databse/Supply_Data/subject"
filepath="./Data/supply_Children_Database/"
file_num='016'

# filepath="/media/peter/Data/Children_Face_Databse/Supply_Data/"
# file_num='subject2'

mesh_face=o3d.io.read_triangle_mesh(filepath+file_num+"/0_SFusion.obj")
T_vertices0=np.asarray(mesh_face.vertices)
T_triangles0=np.asarray(mesh_face.triangles)
nrom0=np.asarray(mesh_face.vertex_normals)
triangle_uvs=np.asarray(mesh_face.triangle_uvs)
img = io.imread(filepath+file_num+"/0_SFusion_0.jpg")
h,w,_=img.shape
triangle_uvs[:,0]=triangle_uvs[:,0]*h
triangle_uvs[:,1]=h-triangle_uvs[:,1]*h
triangle_uvs=triangle_uvs.astype(int)

T_colors0=np.repeat(np.random.randn(1,3)*255,len(T_vertices0),0).astype(np.uint8) 
T_colors0[T_triangles0.flatten(),:]=img[triangle_uvs[:,1],triangle_uvs[:,0],:].astype(float)


rotate_x=0
rotate_y=90
if rotate_y!=0:
    angle0=np.array([0,rotate_y, 0])
    T_vertices0=T_vertices0.dot(angle2matrix(angle0).T)
    
if rotate_x!=0:
    angle0=np.array([rotate_x,0, 0])
    T_vertices0=T_vertices0.dot(angle2matrix(angle0).T) 
    
plot_mlabvertex(T_vertices0,T_colors0,T_triangles0)


Nose_idx=np.argmax(T_vertices0[:,2])
T_vertices0=T_vertices0-T_vertices0[Nose_idx,:].reshape(1,3)

#T_vertices2,rotmatrix=face_correction(S_vertices,T_vertices0,threshold=50,flag_rotmatrix=True)
#plot_mlabvertex(T_vertices2,T_colors0,T_triangles0)

plot_mlabvertex(T_vertices0,T_colors0,T_triangles0)

dis_nose=np.sqrt(np.sum(T_vertices0*T_vertices0,1))
Sub_vertices=T_vertices0[dis_nose<=Threshold_rnose,:]

Sub_vertices[Sub_vertices[:,1]>50,0]=0
Sub_vertices[Sub_vertices[:,1]<-20,0]=0

Ear_idx=np.argmax(np.abs(Sub_vertices[:,0]))

Threshold_ear=np.sqrt(np.sum(Sub_vertices[Ear_idx,:]**2))*1.03

weight1=np.array([1,2.3,1]).reshape(1,3)
dis_nose1=np.sqrt(np.sum(T_vertices0*T_vertices0*weight1,1))

weight2=np.array([1,1.3,1]).reshape(1,3)
dis_nose2=np.sqrt(np.sum(T_vertices0*T_vertices0*weight2,1))

dis_nose1[T_vertices0[:,1]<0]=dis_nose2[T_vertices0[:,1]<0]
face_index=np.where(dis_nose1<=Threshold_ear)[0]

T_vertices1,T_colors1,T_triangles1=crop_mesh(face_index,T_vertices0,T_colors0,T_triangles0)
plot_mlabvertex(T_vertices1,T_colors1,T_triangles1)


# ============================================Face Pose Corection ==========================================

T_vertices2,rotmatrix=face_correction(S_vertices,T_vertices1,threshold=50,flag_rotmatrix=True)

plot_mlabvertex(T_vertices2,T_colors1,T_triangles1)#,azimuth=0, elevation=-220)

#save_ply(T_vertices2,T_colors1,T_triangles1,outdir+'Children_sub016n.ply')

voxel_size=4
T_points =o3d.geometry.PointCloud()#o3d.geometry.PointCloud()#
T_points.points = o3d.utility.Vector3dVector(T_vertices2)

target_points=T_points.voxel_down_sample(voxel_size)    
T_sampoints=np.asarray(target_points.points)

print(T_sampoints.shape)

plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,T_sampoints)#,azimuth=0, elevation=-220)

#==================================================registration===================================================

# affine cpd registration
tf_param1, _, _ = cpd.registration_cpd(source_points, target_points, #tf_init_params={},
                                      tf_type_name='affine')#lmd=0.01,beta=3,

result = copy.deepcopy(source_points)
result.points = tf_param1.transform(result.points)

R_sampoints1=np.asarray(result.points)

plot_mlabvertex(R_sampoints1,S_colors,S_triangles)

plot_2mlabvertex(T_vertices2,T_colors1,T_triangles1,R_sampoints1,S_colors,S_triangles,R_sampoints1[landmark_idx,:])#,azimuth=0, elevation=-220)


#nonrigid cpd registration
source_points2 =o3d.geometry.PointCloud()
source_points2.points = o3d.utility.Vector3dVector(R_sampoints1)

tf_param1, _, _ = cpd.registration_cpd(source_points2, target_points, 
                                       lmd=0.8,beta=10,tf_type_name='nonrigid')#

result2 = copy.deepcopy(source_points2)
result2.points = tf_param1.transform(result2.points)

R_sampoints2=np.asarray(result2.points)

plot_mlabvertex(R_sampoints2,S_colors,S_triangles,R_sampoints2[landmark_idx,:])#,azimuth=0, elevation=-220)

plot_2mlabvertex(T_vertices2,T_colors1,T_triangles1,R_sampoints2,S_colors,S_triangles,R_sampoints2[landmark_idx,:])#,azimuth=0, elevation=-220)

#plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,R_sampoints2)#,azimuth=0, elevation=-220)
# Y=S_sampoints.copy() #target points template
# X=T_sampoints.copy() #source points
# reg = DeformableRegistration(**{'X': X, 'Y': Y,'alpha':0.1,'beta':6})
# R_sampoints1, (W, G) = reg.register()
# plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,R_sampoints1)#,azimuth=0, elevation=-220)

# tf_param2=filterreg.registration_filterreg(source_points, target_points)
# result.points = tf_param2.transform(result.points)
# R_sampoints2=np.asarray(result.points)
# plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,R_sampoints2)#,azimuth=0, elevation=-220)

# tf_param=bcpd.registration_bcpd(source_points, target_points)
# result.points = tf_param.transform(result.points)
# R_sampoints3=np.asarray(result.points)
# plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,R_sampoints3)#,azimuth=0, elevation=-220)

# # #------------------------------------------------Fitting Error--------------------------------------------
# F_landmark=np.vstack((F_vertices[X_ind,:],F_vertices[S_Ear_idx1,:],F_vertices[S_Ear_idx2,:]))
# source1=TriMesh(F_vertices, F_triangles)
# source1.landmarks[group]=PointCloud(F_landmark)

# dist_error,N_vertices,tri_indices =fit_shaperror(source1,target,flag_Near_vertices=True)

# print("Mean Error:", dist_error.mean())

# plot_mlabfaceerror(F_vertices,dist_error,F_triangles)

# plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,N_vertices[land_idx,:]) 

# # plot_2mlabvertex(Near_vertices,S_colors,F_triangles,
# #                  F_vertices,S_colors,F_triangles,F_landmarker)

# save_ply(F_vertices, S_colors*255,  F_triangles,'./Data/Fitting_Faces/Fitting_Face'+file_num+'.ply')
# save_ply(T_vertices2,T_colors1,T_triangles1,'./Data/Corrected_Faces/Corrected_Face'+file_num+'.ply')

# #save_ply(F_vertices, S_colors*255,  F_triangles,filepath+file_num+'/fitting_face.ply')
# #save_ply(T_vertices2,T_colors1,T_triangles1,filepath+file_num+'/Corrected_face.ply')
# #save_ply(T_vertices0r[:,0:3],T_colors0,T_triangles0,filepath+file_num+'/Corrected_head.ply')
    
# # plot_mlabfaceerror(S_vertices,data_weights[7,:],S_triangles,
# #                    azimuth=0, elevation=-220,colormap_range=np.array([0., 1.]))