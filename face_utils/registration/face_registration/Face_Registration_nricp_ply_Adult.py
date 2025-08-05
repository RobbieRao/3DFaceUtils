# -*- coding: utf-8 -*-
"""
Created on 2025-04-04 17:48:22

@author: Robbie
"""
#from matplotlib import cm,colors
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from menpo.shape import TriMesh,PointCloud
from scipy.spatial import Delaunay
import pandas as pd
import menpo3d.io.input.mesh as ia
import os

from face_utils.reconstruction.visualize import (
    plot_mlabvertex,
    plot_2mlabvertex,
    plot_mlabfaceerror,
)
from face_utils.reconstruction.plyio import read_ply, save_ply
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

#land_select=np.arange(17,68)

#S_Ear_idx1=S_vertices[:,0].argmax()
#S_Ear_idx2=S_vertices[:,0].argmin()

S_landmarks=S_vertices[X_ind,:]
plot_mlabvertex(S_vertices,S_colors,S_triangles,S_landmarks)

#===============================================target Face============================================
#df_files=pd.read_csv("./Data/Adult_Head_Database/excluded_faces.csv")
#file_lists=df_files["name"][df_files["excluded"]==1.0].values

filePath1 ="./Data/Adult_Head_Database/Tina/temp_fitting_face/"
filePath2 ="./Data/Adult_Head_Database/Tina/Corrected_Face/"

file_lists=os.listdir(filePath1)  

for idx in range(len(file_lists)):
    #for idx in range(90,92):
    #idx=1
    print(idx)
    filenum=file_lists[idx][0:6]
    #filename='./Data/Children_head_Database/'+filenum+'.ply'
    
    F_mesh=ia.ply_importer(filePath1+filenum+'.ply')#.points#read_ply(filePath1+i)
    F_vertices0=F_mesh.points
    F_triangles0=F_mesh.trilist
    #F_colors0=np.repeat([100,100,100],len(F_vertices0)).reshape(-1,3)/255
    #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0)#,S_landmarks)
    
    T_mesh=ia.ply_importer(filePath2+filenum+'.ply')#.points#read_ply(filePath1+i)
    T_vertices0=T_mesh.points
    T_triangles0=T_mesh.trilist
    T_colors0=np.repeat([100,100,100],len(T_vertices0)).reshape(-1,3)/255
    
    T_landmarks=F_vertices0[X_ind,:]
    #plot_2mlabvertex(T_vertices0,T_colors0,T_triangles0,
    #                F_vertices0,S_colors,F_triangles0,T_landmarks)
    
    #==============================================Face Registration======================================
    
    group='lsfm'
    target=TriMesh(T_vertices0, T_triangles0)
    target.landmarks[group]=PointCloud(T_landmarks) #attribtion:points
    
    source=TriMesh(S_vertices, S_triangles)
    source.landmarks[group]=PointCloud(S_landmarks)
    
    #plot_2mlabvertex(T_vertices2,T_colors1,T_triangles2,
    #                 S_vertices,S_colors,S_triangles,T_landmarks)
    
    aligned =correspond_mesh(source,target,group,
                                     landmark_weights=[10,5,2,1,0.5,0.0, 0.0,0.0],
                                     stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2],
                                     PerNrom_type="Percentage",
                                     flag_data_weights=True,
                                     verbose=True)
    
    F_vertices=aligned.points
    #F_triangles=aligned.trilist
    # F_landmarker=F_vertices[X_ind,:]
    
    #plot_mlabvertex(F_vertices,S_colors,F_triangles,F_landmarker) 
    
    #plot_mlabvertex(T_vertices0,T_colors0,T_triangles0,F_landmarker) 
    
    
    #plot_2mlabvertex(T_vertices0,T_colors0,T_triangles0,  F_vertices,S_colors,S_triangles,F_vertices[X_ind,:])
         
    # #------------------------------------------------Fitting Error--------------------------------------------
    #F_landmark=np.vstack((F_vertices[X_ind,:],F_vertices[S_Ear_idx1,:],F_vertices[S_Ear_idx2,:]))
    # source1=TriMesh(F_vertices, F_triangles)
    # #source1.landmarks[group]=PointCloud(F_landmark)
    
    # dist_error,Near_vertices,tri_indices =fit_shaperror(source1,target,flag_Near_vertices=True)
    
    # print("Mean Error:", dist_error.mean())
    
    # plot_mlabfaceerror(F_vertices,dist_error,F_triangles)
    
    # plot_2mlabvertex(Near_vertices,S_colors,F_triangles,
    #                  F_vertices,S_colors,F_triangles,F_landmarker)
    
    save_ply(F_vertices, S_colors*255,  S_triangles,'./Data/Adult_Head_Database/Tina/temp_fitting_face2/'+filenum+'.ply')
