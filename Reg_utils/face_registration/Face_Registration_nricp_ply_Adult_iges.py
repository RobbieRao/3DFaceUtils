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
import menpo3d.io.input.mesh as ia

from Rec_utils.transform import angle2matrix,procrustes_alignment,P2sRt
from face3d.morphable_model import MorphabelModel
from Rec_utils.mesh import crop_mesh
from  Rec_utils.landmark import landmark3d_detect
from Rec_utils.plyio import read_VRML,read_ply, write_ply
from Rec_utils.visualize import plot_mlabvertex,plot_2mlabvertex,plot_mlabfaceerror
from Reg_utils.face_corespond import face_correction,correspond_mesh
from Reg_utils.uvmap_processing import Vertices2Mapuv
from Rec_utils.fitting import fit_shaperror
#===================================================Parameters===================================

Threshold_rnose=0.17*1000
Theshold_dtriangles=0.005
outpath="./Data/Chinese_TestingData/"

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
                    14194,  2598, 41914, 39208, 40553, 52401, 43128, 11896,  4303,
                    9229,  7189,  8191,  8191, 11747,  4537,  8274,  8322, 32853,
                    21627, 15004,  2509, 14825,  2071])

S_landmarks=S_vertices[land_idx,:]
plot_mlabvertex(S_vertices,S_colors,S_triangles,S_landmarks)

group='lsfm'

source=TriMesh(S_vertices, S_triangles)
source.landmarks[group]=PointCloud(S_landmarks)

path_landmark="./Data/Adult_Head_Database/Tina/"
df_landmark=pd.read_csv(path_landmark+"landmark_info.csv",header=[0,1])  #df_landmark.columns
filenames=df_landmark['File name'].values.flatten()

from collections import Counter
a=np.array([df_landmark.columns[i][0]  for i in range(len(df_landmark.columns))])
land_items=[item for item, count in Counter(a).items() if count > 1]

#path_landmark="/media/peter/Data/Face_TShape&Image_Database/HeadChinese/"

#===============================================target Face============================================

#filename='b0062m'
path="./Data/Chinese_TestingData/all/"#'./Data/Adult_Head_Database/Tina/supply_data/'

# file_lists=[os.listdir("./Data/Chinese_TestingData/"+path+"/")  for path in ["16f","25f","50f","16m","25m","50m"]]
# df_file=pd.DataFrame(dict(file_name=np.array(file_lists).flatten(),
#                           groups=np.repeat(['Female 18-', 'Female 18-50' ,'Female 50+' ,'Male 18-' ,'Male 18-50','Male 50+'],10)
#                       ))
# df_file.to_csv("./Data/Chinese_TestingData/test_file_info.csv",index=False)

file_lists=sorted(glob.glob(path+"*.ply"))

num_file=len(file_lists)

for fielnum in range(22,num_file):
    print(fielnum)
    
    filename=file_lists[fielnum][-10:-4]
    try:    
        # mesh_head=ia.ply_importer(path+filename+'.ply')
        # T_vertices0=mesh_head.points#read_ply(filePath1+i)
        # T_triangles0=mesh_head.trilist
        # T_colors0=np.repeat(np.random.rand(1,3),len(T_vertices0),0).astype(float) 
        T_vertices0,T_colors0,T_triangles0=read_ply(path+filename+'.ply')
    except:
        continue
    #T_vertices0,T_colors0,T_triangles0= read_VRML(path+filename)
    
    #filename='b0559f'
    #path='/media/peter/CHDI1571FV(Mac)2//Headsize Medium/Female/51-70+/'
    #file_lists=glob.glob(path+"*.wrl")
    
    #fielnum=11
    
    #for fielnum in np.arange(3,len(file_lists)):
        #print(fielnum)
    #filename='l0581f'#file_lists[fielnum][-10:-4]
    
    #T_vertices0,T_colors0,T_triangles0= read_VRML(path+filename)
    
    #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0)
    #===========================================maunal landmark===============================================
    
    row_idx=np.where(filenames==filename)[0][0]
    
    
    manual_land=df_landmark.loc[row_idx,land_items]
            
    manual_land=pd.to_numeric(manual_land, downcast='float').values.reshape(-1,3)
            
    
    #manual_land=manual_land[:,[1,2,0]]
    manual_land=manual_land[:,[0,1,2]]
    #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0,manual_land)
    
    Tragion_idx=np.array([ np.sum((T_vertices0-x.reshape(1,3))**2,1).argmin()  for x in manual_land])
    
    #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0,T_vertices0[Tragion_idx,:])
    
    #===========================================rotation===============================================
    # rotate_x=90
    # rotate_y=90
    # if rotate_y!=0:
    #     angle0=np.array([0,rotate_y, 0])
    #     T_vertices0=T_vertices0.dot(angle2matrix(angle0).T)
        
    # if rotate_x!=0:
    #     angle0=np.array([rotate_x,0, 0])
    #     T_vertices0=T_vertices0.dot(angle2matrix(angle0).T) 
        
    
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
    print(P2sRt(rotmatrix))
    #plot_mlabvertex(T_vertices2,T_colors1,T_triangles1,T_vertices2[Tragion_idx1,:])#,azimuth=0, elevation=-220)
    
    # #========================================================= Head Correcttion============================================
    T_vertices0r=(np.c_[T_vertices0,np.zeros((len(T_vertices0),1))]).dot(rotmatrix)
    #plot_mlabvertex(T_vertices0r,T_colors0,T_triangles0)#,azimuth=0, elevation=-220)
    
    
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
    
    write_ply(outpath+'Fitting_Face/'+filename+'.ply',F_vertices, S_colors*255,  F_triangles)
    #save_ply(T_vertices2,T_colors1*255,T_triangles1,outpath+'Corrected_Face/'+filename+'.ply')
    write_ply(outpath+'Corrected_Head/'+filename+'.ply',T_vertices0r[:,0:3],T_colors0*255,T_triangles0)
    
    # F_vertices, S_colors,  F_triangles=read_ply(outpath+'temp_fitting_face/'+filename+".ply")
    # plot_mlabvertex(F_vertices, S_colors,  F_triangles)

# import os
# os.system("/home/peter/Documents/Chinese_Face_Model/Chinese_Face_model_Specificity_Analysis.py")