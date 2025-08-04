# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:33:49 2020

@author: Peter_Zhang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from menpo.shape import TriMesh,PointCloud
from skimage import io 
import open3d as o3d #http://www.open3d.org/docs/release/tutorial/Basic/mesh.html
from scipy.spatial import Delaunay
import os, fnmatch
import glob

from face_utils.reconstruction.transform import angle2matrix, procrustes_alignment
from face3d.morphable_model import MorphabelModel
from face_utils.reconstruction.mesh import crop_mesh
from face_utils.reconstruction.landmark import landmark3d_detect
from face_utils.reconstruction.plyio import read_VRML, read_ply, save_ply
from face_utils.reconstruction.visualize import (
    plot_mlabvertex,
    plot_2mlabvertex,
    plot_mlabfaceerror,
)
from ..face_corespond import face_correction, correspond_mesh
from ..uvmap_processing import Vertices2Mapuv
from face_utils.reconstruction.fitting import fit_shaperror
#===================================================Parameters===================================

Threshold_rnose=0.17*1000
Theshold_dtriangles=0.005
outpath="./Data/Adult_Head_Database/"

#===============================================Source Face============================================
#C = sio.loadmat('./model_data/BFM_Children_Model/BFM_Children.mat')
C = sio.loadmat('./model_data/BFM_2009_Model/BFM.mat')
S_vertices =((np.reshape(C['model']['shapeMU'][0,0],(-1,3)))/1.0e3)
#S_vertices=(C['model']['shapeMU'][0,0].flatten()+C['model']['expPC'][0,0][:,0]*1.0).reshape(-1,3)/1000

S_colors = np.reshape(C['model']['texMU'][0,0],(-1,3))
S_triangles=(C['model']['tri'][0,0]).T-1
S_colors = S_colors/np.max(S_colors)
X_ind = C['model']['kpt_ind'][0,0].flatten()-1

S_vertices=S_vertices-S_vertices[S_vertices[:,2].argmax(),:].reshape(1,3)

#land_select=np.r_[np.arange(17,68),15,1]

#S_Ear_idx1=S_vertices[:,0].argmax()
#S_Ear_idx2=S_vertices[:,0].argmin()
land_idx=np.array([10892,  5491, 10793,  5908, 47904, 48691, 49085, 46598, 46937,
                   14194,  2598, 41914, 39208, 40553, 30334, 24403, 11896,  4303,
                    9229,  7189,  8191,  8191, 11747,  4537,  8274,  8322, 32853,
                   21627, 15004,  2509, 14825,  2071])


S_landmarks=S_vertices[land_idx,:]
plot_mlabvertex(S_vertices,S_colors,S_triangles,S_landmarks)

group='lsfm'

source=TriMesh(S_vertices, S_triangles)
source.landmarks[group]=PointCloud(S_landmarks)

path_landmark="/media/peter/Data/Face_TShape&Image_Database/HeadChinese/"
df_landmark=pd.read_csv(path_landmark+"landmark_info.csv",header=[0,1])  #df_landmark.columns
filenames=df_landmark['File name'].values.flatten()

from collections import Counter
a=np.array([df_landmark.columns[i][0]  for i in range(len(df_landmark.columns))])
land_items=[item for item, count in Counter(a).items() if count > 1]
#===============================================target Face============================================
path_lists0=['/media/peter/CHDI0480LV(Mac)',
            '/media/peter/CHDI0480LV(Mac)1',
            '/media/peter/CHDI0480LV(Mac)2',
            '/media/peter/CHDI1571LV(Mac)',
            '/media/peter/CHDI1571LV(Mac)1',
            '/media/peter/CHDI1571LV(Mac)2',
            '/media/peter/CHDI1571LV(Mac)3',
            '/media/peter/CHDI1571LV(Mac)4',
            '/media/peter/CHDI1571LV(Mac)5'
            ]

gender_type=["Female/","Male/"]
age_type=["18-30/","31-50/","51-70+/"]
X,Y=np.meshgrid(gender_type,age_type)
XY=[ x+y for x,y in zip(X.flatten(),Y.flatten())]

size_type=["/Headsize Large/","/Headsize Medium/","/Headsize Small/"]
X,Y=np.meshgrid(XY,size_type)
XYZ=[ x+y for x,y in zip(Y.flatten(),X.flatten())]

X,Y=np.meshgrid(path_lists0,XYZ)
path_lists=[ x+y for x,y in zip(X.flatten(),Y.flatten())]


filePath1 ="./Data/Adult_Head_Database/Fitting_Face/"
file_fittlists=np.array([x[0:6] for x in os.listdir(filePath1)])
    
#path='/media/peter/CHDI0480HV(Mac)/Headsize Large/Male/31-50/'
for path in path_lists:
    file_lists=glob.glob(path+"*.wrl")
    
    #fielnum=11
    for fielnum in np.arange(0,len(file_lists)):
        filename=file_lists[fielnum][-10:-4]
        if sum(file_fittlists==filename)==1:
            continue
        print(path,fielnum)
        print(filename)
        
        #filename='b0559f'
        #path='/media/peter/CHDI1571FV(Mac)2//Headsize Medium/Female/51-70+/'
        T_vertices0,T_colors0,T_triangles0= read_VRML(path+filename)
        
        #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0)
        #===========================================maunal landmark===============================================
        try:
            row_idx=np.where(filenames==filename)[0][0]
        except:
            continue
        
        manual_land=df_landmark.loc[row_idx,land_items]
        
        manual_land=pd.to_numeric(manual_land, downcast='float').values.reshape(-1,3)
        
        manual_land=manual_land[:,[1,2,0]]
        #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0,manual_land)
        
        Tragion_idx=np.array([ np.sum((T_vertices0-x.reshape(1,3))**2,1).argmin()  for x in manual_land])
        
        #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0,T_vertices0[Tragion_idx,:])
        
        #===========================================rotation===============================================
        rotate_x=90
        rotate_y=90
        if rotate_y!=0:
            angle0=np.array([0,rotate_y, 0])
            T_vertices0=T_vertices0.dot(angle2matrix(angle0).T)
            
        if rotate_x!=0:
            angle0=np.array([rotate_x,0, 0])
            T_vertices0=T_vertices0.dot(angle2matrix(angle0).T) 
            
        
        #Nose_idx=np.argmax(T_vertices0[:,2])
        T_vertices0=T_vertices0-T_vertices0[Tragion_idx[20],:].reshape(1,3)
        
        
        #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0,T_vertices0[Tragion_idx,:])
        
        #dis_nose=np.sqrt(np.sum(T_vertices0*T_vertices0,1))
        #Sub_vertices=T_vertices0[dis_nose<=Threshold_rnose,:]
        
        #Sub_vertices[Sub_vertices[:,1]>50,0]=0
        #Sub_vertices[Sub_vertices[:,1]<-20,0]=0
        
        #Ear_idx=np.argmax(np.abs(Sub_vertices[:,0]))
        Threshold_ear=np.max([np.sqrt(np.sum(T_vertices0[Tragion_idx[26],:]**2)),
                              np.sqrt(np.sum(T_vertices0[Tragion_idx[27],:]**2))])*1.35
        
        weight1=np.array([1,2.3,1]).reshape(1,3)
        dis_nose1=np.sqrt(np.sum(T_vertices0*T_vertices0*weight1,1))
        
        weight2=np.array([1,1.3,1]).reshape(1,3)
        dis_nose2=np.sqrt(np.sum(T_vertices0*T_vertices0*weight2,1))
        
        dis_nose1[T_vertices0[:,1]<0]=dis_nose2[T_vertices0[:,1]<0]
        face_index=np.where(dis_nose1<=Threshold_ear)[0]
        
        T_vertices1,T_colors1,T_triangles1,Tragion_idx1=crop_mesh(face_index,T_vertices0,T_colors0,T_triangles0,X_ind0=Tragion_idx)
        #plot_mlabvertex(T_vertices1,T_colors1,T_triangles1,T_vertices1[Tragion_idx1,:])
        
        # ============================================Face Pose Corection ==========================================
        T_vertices2,rotmatrix=face_correction(S_vertices,T_vertices1,threshold=50,flag_rotmatrix=True)
        
        #plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,T_vertices2[Tragion_idx1[1:2],:])#,azimuth=0, elevation=-220)
        
        # #========================================================= Head Correcttion============================================
        T_vertices0r=(np.c_[T_vertices0,np.zeros((len(T_vertices0),1))]).dot(rotmatrix)
        #plot_mlabvertex(T_vertices0r,T_colors0,T_triangles0)#,azimuth=0, elevation=-220)
        
        save_ply(T_vertices0r[:,0:3],T_colors0*255,T_triangles0,outpath+'Corrected_Head/'+filename+'.ply')
        
        #==============================================Face  landmark detection===============================
        
        # T_landmark,T_landmark_idx=landmark3d_detect(T_vertices2,T_colors1,T_triangles1,
        #                                             method='face_alignment',angle=[0, 0, 0], 
        #                                             img_h=500,img_w=500,
        #                                             flag_show=True,flag_imgadjust=True)
        
        #plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,T_landmark)
            
        #==============================================mesh delaunay ===============================
        # Nose_idx=T_vertices2[:,2].argmax()
        # Sub_vertices=T_vertices2.copy()
        
        # Sub_vertices[Sub_vertices[:,1]>50,0]=0
        # Sub_vertices[Sub_vertices[:,1]<-20,0]=0
        
        # Ear_idx=[Sub_vertices[:,0].argmax(),Sub_vertices[:,0].argmin()]
        # mapuv_info=Vertices2Mapuv(T_vertices2,T_triangles1,T_colors1,landmark_idx=None,
        #                           Nose_idx=Nose_idx,Ear_idx=Ear_idx,
        #                           flag_show=True,flag_select=False)
        
        # uvmap_Delaunay= Delaunay(mapuv_info.uvmap[:,0:2])
        # T_triangles2=uvmap_Delaunay.simplices
        
        T_landmarks=T_vertices2[Tragion_idx1,:]
        
        #plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,T_landmarks)#,azimuth=0, elevation=-220)
        
        #==============================================Face Registration======================================
        
        
        target=TriMesh(T_vertices2, T_triangles1)
        target.landmarks[group]=PointCloud(T_landmarks) #attribtion:points
        
        
        aligned =correspond_mesh(source,target,group,
                                 landmark_weights=[10,5,2,1,0.5,0.0, 0.0,0.0],
                                 stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2],
                                 PerNrom_type="Percentage",
                                 flag_data_weights=True,
                                 verbose=True)
        
        F_vertices=aligned.points
        F_triangles=aligned.trilist
        land_idx=np.r_[C['model']['kpt_ind'][0,0].ravel()]
        F_landmarker=F_vertices[land_idx,:]
        
        #plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,F_landmarker) 
        #plot_2mlabvertex(T_vertices2,T_colors1,T_triangles1,F_vertices,S_colors,F_triangles,F_landmarker)
        
         
        #plot_mlabvertex(F_vertices,S_colors,F_triangles,F_landmarker)#,azimuth=0, elevation=-220) 
        
        # #------------------------------------------------Fitting Error--------------------------------------------
        # F_landmark=F_vertices[X_ind,:]
        # source1=TriMesh(F_vertices, F_triangles)
        # source1.landmarks[group]=PointCloud(F_landmark)
        
        # dist_error,N_vertices,tri_indices =fit_shaperror(source1,target,flag_Near_vertices=True)
        
        # print("Mean Error:", dist_error.mean())
        
        #plot_mlabfaceerror(F_vertices,dist_error,F_triangles)
        
        # plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,N_vertices[land_idx,:]) 
        
        save_ply(F_vertices, S_colors*255,  F_triangles,outpath+'temp_fitting_face/'+filename+'.ply')
        save_ply(T_vertices2,T_colors1*255,T_triangles1,outpath+'Corrected_Face/'+filename+'.ply')
    


