# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:48:22 2025

@author: Robbie
"""
#from matplotlib import cm,colors
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from menpo.shape import TriMesh,PointCloud
from scipy.spatial import Delaunay

from face_utils.reconstruction.visualize import (
    plot_mlabvertex,
    plot_2mlabvertex,
    plot_mlabfaceerror,
)
from face_utils.reconstruction.plyio import read_ply, write_ply
from face_utils.reconstruction.transform import angle2matrix
from face_utils.reconstruction.mesh import crop_mesh
from ..face_corespond import face_correction, correspond_mesh
from face_utils.reconstruction.landmark import landmark3d_detect
from ..uvmap_processing import Vertices2Mapuv
from face_utils.reconstruction.fitting import fit_shaperror

plt.rcParams["font.size"]=12
#===================================================Parameters===================================

Threshold_rnose=0.17*1000
Theshold_dtriangles=0.005

#===============================================Source Face============================================
#C = sio.loadmat('./model_data/BFM_Children_Model/BFM_Children.mat')
C = sio.loadmat('./model_data/BFM_2009_Model/BFM.mat')
S_vertices =((np.reshape(C['model']['shapeMU'][0,0],(-1,3)))/1.0e3)
S_colors = np.reshape(C['model']['texMU'][0,0],(-1,3))
S_triangles=(C['model']['tri'][0,0]).T-1
S_colors = S_colors/np.max(S_colors)
X_ind = C['model']['kpt_ind'][0,0].flatten()-1

land_select=np.arange(17,68)

S_Ear_idx1=S_vertices[:,0].argmax()
S_Ear_idx2=S_vertices[:,0].argmin()

S_landmarks=S_vertices[np.r_[X_ind[land_select],S_Ear_idx1,S_Ear_idx2],:]
plot_mlabvertex(S_vertices,S_colors,S_triangles,S_landmarks)

#===============================================target Face============================================
#for idx in range(90,92):
idx="054"
filenum=str(idx)
filename='./Data/Children_head_Database/scanned_data/'+filenum+'.ply'
T_vertices0,T_colors0,T_triangles0= read_ply(filename)
T_vertices0=T_vertices0*1000  #unit: mm

angle0=np.array([0,0, 0])
if (angle0==0).sum()!=0:
    T_vertices0=T_vertices0.dot(angle2matrix(angle0).T)

Nose_idx=np.argmin(T_vertices0[:,2])
T_vertices0[:,0:2]=T_vertices0[:,0:2]-T_vertices0[Nose_idx,0:2].reshape(1,2)
T_vertices0[:,2]=T_vertices0[Nose_idx,2]-T_vertices0[:,2]

plot_mlabvertex(T_vertices0,T_colors0,T_triangles0)


dist_mean=np.sqrt(np.sum((T_vertices0-T_vertices0.mean(0).reshape(1,3))**2,1))
head_v=T_vertices0[dist_mean<200,:]

Nose_idx=np.argmax(head_v[:,2])
T_vertices0=T_vertices0-head_v[Nose_idx,:].reshape(1,3)

#T_vertices2,rotmatrix=face_correction(S_vertices,T_vertices0,threshold=50,flag_rotmatrix=True)
#plot_mlabvertex(T_vertices2,T_colors0,T_triangles0)

#plot_mlabvertex(T_vertices0,T_colors0,T_triangles0)

dis_nose=np.sqrt(np.sum(T_vertices0*T_vertices0,1))
Sub_vertices=T_vertices0[dis_nose<=Threshold_rnose,:]

Sub_vertices[Sub_vertices[:,1]>50,0]=0
Sub_vertices[Sub_vertices[:,1]<-20,0]=0

Ear_idx=np.argmax(np.abs(Sub_vertices[:,0]))
Threshold_ear=np.sqrt(np.sum(Sub_vertices[Ear_idx,:]**2))*1.12

weight1=np.array([1,2.3,1]).reshape(1,3)
dis_nose1=np.sqrt(np.sum(T_vertices0*T_vertices0*weight1,1))

weight2=np.array([1,1.3,1]).reshape(1,3)
dis_nose2=np.sqrt(np.sum(T_vertices0*T_vertices0*weight2,1))

dis_nose1[T_vertices0[:,1]<0]=dis_nose2[T_vertices0[:,1]<0]
face_index=np.where(dis_nose1<=Threshold_ear)[0]


face_index=np.where(dis_nose1<=Threshold_ear)[0]

T_vertices1,T_colors1,T_triangles1=crop_mesh(face_index,T_vertices0,T_colors0,T_triangles0)
plot_mlabvertex(T_vertices1,T_colors1,T_triangles1)
# ============================================Face Pose Corection ==========================================

T_vertices2,rotmatrix=face_correction(S_vertices,T_vertices1,threshold=100,flag_rotmatrix=True)

plot_mlabvertex(T_vertices2,T_colors1,T_triangles1)

# #========================================================= Head Correcttion============================================
T_vertices0r=(np.c_[T_vertices0,np.zeros((len(T_vertices0),1))]).dot(rotmatrix)
#plot_mlabvertex(T_vertices0r,T_colors0,T_triangles0) 

#
#==============================================Face  landmark detection===============================

# T_landmark,T_landmark_idx=landmark3d_detect(T_vertices2,T_colors1,T_triangles1,
#                                             method='face_alignment',angle=[0, 0, 0], 
#                                             img_h=500,img_w=500,
#                                             flag_show=True,flag_imgadjust=True)

#==============================================Face  landmark detection===============================
detect_method="face_alignment" #face_alignment
T_landmark,T_landmark_idx=landmark3d_detect(T_vertices2,T_colors1,T_triangles1,
                                             #S_vertices,(S_colors*255).astype(np.uint8),S_triangles,
                                            method=detect_method,angle=[0, 0, 0], 
                                            img_h=500,img_w=500,
                                            flag_show=True,flag_imgadjust=True)


if detect_method=="baidu":


    X_ind=np.array([ 1963,  3629,  4533,  5310,  6093,  5063,  4289,  3386,  4410,
                    38612, 39460, 39842, 40043,  6684,  5806,  4775,  3227, 10208,
                    10969, 11739, 12512, 14457, 12914, 12012, 11238, 11874,  9568,
                    41003, 41206, 41558, 42327, 12885, 11467, 10433,  7081,  6854,
                    6621,  5491,  7121,  9401, 11021,  9988,  9495,  9122,  8190,
                    5780,  6406,  8215, 10019, 10666,  9195,  8235,  7275,  6782,
                    8223,  9665,  9189,  8229,  7270, 38697, 40237, 40814, 42280,
                    2858,  4144,  5049,  5699,  5579,  4677,  3902,  2738,  3635,
                    5185, 10584, 11224, 12125, 13288, 13557, 12399, 11626, 10722,
                    11231, 12648,  8167,  8180,  8188,  6517,  7355,  9034, 10007,
                    8203,  6168, 10407,  5643,  6026,  7015,  7615,  8815,  9535,
                    10524, 10786, 10163,  9675,  8835,  8475,  7875,  7635,  6794,
                    6302,  6541,  6288,  7384,  7743,  8703,  9184,  9668, 10032,
                    10035,  9669,  8829,  8589,  7869,  7629,  6788,  6422])
    
   
    S_landmarks=S_vertices[np.r_[X_ind,S_Ear_idx1,S_Ear_idx2],:]
    
    land_select=np.array([ 13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
                        65,  66,  67,  68,  69,  70,  71,  84,  85,  86,  87,  88,  89,
                        90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102,
                        103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                        116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
                        129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                        142, 143, 144, 145, 146, 147, 148, 149])
    
else:
        X_ind = C['model']['kpt_ind'][0,0].flatten()-1
        land_select=np.arange(17,68)
        S_landmarks=S_vertices[np.r_[X_ind[land_select],S_Ear_idx1,S_Ear_idx2],:]

#T_landmark2,T_landmark_idx2=landmark3d_detect(T_vertices2,T_colors1,T_triangles1,
#                                               method='face_alignment',angle=[0, 60, 0], 
#                                               img_h=500,img_w=500,flag_show=True)
plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,T_landmark)

#==============================================mesh delaunay ===============================

Nose_idx=T_vertices2[:,2].argmax()
Sub_vertices=T_vertices2.copy()

Sub_vertices[Sub_vertices[:,1]>50,0]=0
Sub_vertices[Sub_vertices[:,1]<-20,0]=0

Ear_idx=[Sub_vertices[:,0].argmax(),Sub_vertices[:,0].argmin()]
mapuv_info=Vertices2Mapuv(T_vertices2,T_triangles1,T_colors1,landmark_idx=None,
                          Nose_idx=Nose_idx,Ear_idx=Ear_idx,
                          flag_show=True,flag_select=False)

uvmap_Delaunay= Delaunay(mapuv_info.uvmap[:,0:2])
T_triangles2=uvmap_Delaunay.simplices


T_landmarks=T_vertices2[np.r_[T_landmark_idx[land_select],Ear_idx]]

plot_mlabvertex(T_vertices2,T_colors1,T_triangles2,T_landmarks)

#==============================================Face Registration======================================

group='lsfm'
target=TriMesh(T_vertices2, T_triangles2)
target.landmarks[group]=PointCloud(T_landmarks) #attribtion:points

source=TriMesh(S_vertices, S_triangles)
source.landmarks[group]=PointCloud(S_landmarks)

#plot_2mlabvertex(T_vertices2,T_colors1,T_triangles2,
#                 S_vertices,S_colors,S_triangles,T_landmarks)

aligned =correspond_mesh(source,target,group,
                         landmark_weights=[10,5,2,1,0.25,0.1, 0.0,0.0],
                         stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2],
                         PerNrom_type="Constant",
                         flag_data_weights=True,
                         verbose=True)


F_vertices=aligned.points
F_triangles=aligned.trilist
land_idx=np.r_[C['model']['kpt_ind'][0,0].ravel(),S_Ear_idx1,S_Ear_idx2]
F_landmarker=F_vertices[land_idx,:]


plot_mlabvertex(F_vertices,S_colors,F_triangles,F_landmarker) 

plot_mlabvertex(T_vertices2,T_colors1,T_triangles2,F_landmarker) 


plot_2mlabvertex(T_vertices2,T_colors1,T_triangles2,
                 F_vertices,S_colors,F_triangles,
                 F_landmarker)
     
# #------------------------------------------------Fitting Error--------------------------------------------
F_landmark=np.vstack((F_vertices[X_ind,:],F_vertices[S_Ear_idx1,:],F_vertices[S_Ear_idx2,:]))
source1=TriMesh(F_vertices, F_triangles)
source1.landmarks[group]=PointCloud(F_landmark)

dist_error,Near_vertices,tri_indices =fit_shaperror(source1,target,flag_Near_vertices=True)

print("Mean Error:", dist_error.mean())

plot_mlabfaceerror(F_vertices,dist_error,F_triangles)

# plot_2mlabvertex(Near_vertices,S_colors,F_triangles,
#                  F_vertices,S_colors,F_triangles,F_landmarker)

save_ply(F_vertices, S_colors*255,  F_triangles,'./Data/Children_head_Database/Fitting_Faces/Fitting_Face'+filenum+'.ply')
save_ply(T_vertices2,T_colors1,T_triangles1,'./Data/Children_head_Database/Corrected_Faces/Corrected_Face'+filenum+'.ply')
save_ply(T_vertices0r[:,0:3],T_colors0,T_triangles0,'./Data/Children_head_Database/Corrected_Heads/Corrected_Head'+filenum+'.ply')
