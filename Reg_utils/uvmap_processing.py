# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:03:45 2020

@author: Peter_Zhang
"""
import numpy as np
#from time import time
import math
import matplotlib.pyplot as plt
from typing import NamedTuple
from Rec_utils.visualize import plot_mlabvertex,plot_matuvmap


def Vertices2Mapuv(vertices,triangles=None,texture=None,landmark_idx=None,
                   Nose_idx=None,Ear_idx=None,flag_show=True,flag_select=True):
    
    #uvmap: NX3-U,V,R
    if texture is not None:
        assert vertices.shape[0] == texture.shape[0]
    else:
        texture=np.repeat(np.array([100,100,100]).reshape(1,3),len(vertices),0).astype(np.uint8)
        
    threshold_y=0.025*1000
    vertice_temp=vertices[(vertices[:,1]<=threshold_y*1.5) & (vertices[:,1]>=-threshold_y),:]
    if Ear_idx is None:
        #BFM 2009 reference
        Ear_value11=np.array([ 0.08989184,  0.02110429, -0.14616016]).reshape(1,3)*1000
        Ear_value12=np.array([-0.09017756,  0.0210322 , -0.14674583]).reshape(1,3)*1000
        Ear_idx11=np.argmin(np.sum((vertices-Ear_value11)**2,1))
        Ear_idx12=np.argmin(np.sum((vertices-Ear_value12)**2,1))
           
        Ear_value21=vertice_temp[np.argmax(vertice_temp[:,0]),:]
        Ear_idx21=np.where((Ear_value21[0]==vertices[:,0]) &(Ear_value21[1]==vertices[:,1]))[0][0]
        #Ear_idx22=np.argmin(vertices[(vertices[:,1]<=threshold_y) & (vertices[:,1]>=-threshold_y),0])
        Ear_value22=vertice_temp[np.argmin(vertice_temp[:,0]),:]
        Ear_idx22=np.where((Ear_value22[0]==vertices[:,0]) &(Ear_value22[1]==vertices[:,1]))[0][0]
        
        Threshold_ear=0.02*1000
        Ear_idx1=Ear_idx21 if np.sqrt(np.sum((Ear_value11-Ear_value21)**2))<Threshold_ear else Ear_idx11 
        Ear_idx2=Ear_idx22 if np.sqrt(np.sum((Ear_value12-Ear_value22)**2))<Threshold_ear else Ear_idx12
    else:
        Ear_idx1=Ear_idx[0]
        Ear_idx2=Ear_idx[1]
    
    ver_center=(vertices[Ear_idx1,:]+vertices[Ear_idx2,:])/2
    
    if Nose_idx is None:
        Nose_idx=np.argmin(np.sum(vertices*vertices,axis=1))#np.argmax(vertices[:,2])
    #print(Nose_idx)
    #average_z=vertices[[Nose_idx,Ear_idx1,Ear_idx2],2].mean()/2
    
    points_trans=vertices-np.repeat(ver_center.reshape((1,3)),vertices.shape[0],0)
    
    U=[math.atan2(x[2],x[0])*180/np.pi for x in points_trans]
    U=np.array([180-np.abs(x)+180 if x<=-90 else x for x in U ])-90.0
    V=points_trans[:,1]
    R=np.array([math.sqrt(x[2]*x[2]+x[0]*x[0]) for x in points_trans]) 
    
    #=======================calculate U_scalse=============================================
    
    dist_earuv=np.abs(U[Ear_idx1]-U[Ear_idx2])
    R1=np.sqrt(np.sum((vertices[Ear_idx1,:]-ver_center)*
                      (vertices[Ear_idx1,:]-ver_center)))
    R2=np.sqrt(np.sum((vertices[Ear_idx2,:]-ver_center)*
                      (vertices[Ear_idx2,:]-ver_center)))
    dist_ear=2*np.mean([R1,R2])*np.pi/360*dist_earuv 
    U_scalse=dist_earuv/dist_ear
   
    uvmap=np.vstack((U/U_scalse,V,R)).T
    
    uvmap_center=uvmap[Nose_idx,:].copy()
    uvmap_center[1]=uvmap_center[1]-0.015
    #uvmap_center[1]=uvmap_center[1]-(uvmap_center[1]-uvmap[vertices[:,2]>average_z,1].min())/5
    
    uvmap[:,0]=uvmap[:,0]-uvmap_center[0]
    uvmap[:,1]=uvmap[:,1]-uvmap_center[1]
    
    
    landmarkers=uvmap[landmark_idx,:]
    
    #flag_select=True
    
    if flag_select: #以耳为半径截取圆形区域
        Threshold_ear1=np.sqrt(np.sum((uvmap[Ear_idx1,0:2]-uvmap[Nose_idx,0:2])**2))
    
        Threshold_ear2=np.sqrt(np.sum((uvmap[Ear_idx2,0:2]-uvmap[Nose_idx,0:2])**2))
    
        Threshold_ear=max([Threshold_ear1,Threshold_ear2])

        dist_center=np.sqrt(np.sum((uvmap[:,0:2]-uvmap[Nose_idx,0:2])**2,1))

        dis_index=np.arange(len(dist_center))[dist_center<=Threshold_ear]
        
        uvmap=uvmap[dis_index,:]
        texture=texture[dis_index,:]
        
        vertices=vertices[dis_index,:]
        if landmark_idx is not None:
            landmark_idx=np.array([np.sum((uvmap-x.reshape(1,3).repeat(len(uvmap),0))**2,1).argmin()
                               for x in landmarkers])
        
        tri_index=np.ones((len(dist_center)))*-1
        tri_index[dis_index]=np.arange(len(dis_index))
        
        triangles=triangles[(dist_center[triangles[:,0]]<=Threshold_ear) & 
                          (dist_center[triangles[:,1]]<=Threshold_ear) & 
                          (dist_center[triangles[:,2]]<=Threshold_ear),:]
        triangles=np.array([tri_index[triangles[:,0]],
                       tri_index[triangles[:,1]],
                       tri_index[triangles[:,2]]]).astype(int).T
        
    
    if flag_show:
        if landmark_idx is not None:
            landmarkers=uvmap[landmark_idx,:]
            idx=uvmap[:,2].argsort()
            plot_matuvmap(uvmap[idx,:],texture[idx,:],landmarkers,flag_equal=True)
        else:
            idx=uvmap[:,2].argsort()
            plot_matuvmap(uvmap[idx,:],texture[idx,:],None,flag_equal=True)
            plot_matuvmap(uvmap[idx,:],colors=None,landmarker=None,cmap=plt.cm.gray,flag_equal=True)
    
    class mapuv_info(NamedTuple):
        uvmap:float
        vertices:float
        triangles:int
        texture:float
        landmark_idx:int
        ver_center:float
        U_scalse:float
        uvmap_center:float
    mapuv_info=mapuv_info(uvmap,vertices,triangles,texture,
                          landmark_idx,
                          ver_center,U_scalse,uvmap_center)  
    
    return mapuv_info#,U_scalse,ver_center


def VMapuv2Image(uvmap, triangles, colors,landmark_idx=None, 
                  img_h=256, img_w=256, flag_show=True):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
    Returns:
        image: [h, w, 4]. colors+depth
    '''
    assert uvmap.shape[0] == colors.shape[0]
    
    c=3
    uv_center=np.array([0.0,0.0])#np.ptp(uvmap[:,0:2],0)/2+np.min(uvmap[:,0:2],0)
    
    I_scale_w=img_w/2/np.min((np.abs(np.min(uvmap[:,0],0)-uv_center[0]),np.max(uvmap[:,0],0)-uv_center[0]))
    
    I_scale_h=I_scale_w*1.45
    #I_scale_h=img_h/2/np.mean((np.abs(np.min(uvmap[:,1],0)-uv_center[1]),np.max(uvmap[:,1],0)-uv_center[1]))
    #print(I_scale_h/I_scale_w)
    
    vertices=np.zeros_like(uvmap)
    vertices[:,0]=(uvmap[:,0]-uv_center[0])*I_scale_w+img_w/2
    vertices[:,1]=img_h-((uvmap[:,1]-uv_center[1])*I_scale_h+img_h/2)
    vertices[:,2]=uvmap[:,2]
    
    if landmark_idx is not None:
        landmarker=vertices[landmark_idx,:]
    else:
        landmarker=None
    # initial 
    uvimg = np.zeros((img_h, img_w, c+1))
    depth_buffer = np.zeros([img_h, img_w]) - 999999.
    depth_vertices=np.zeros(vertices.shape[0])
    triangles=triangles[np.argsort((uvmap[triangles[:,0],2]+uvmap[triangles[:,1],2]+uvmap[triangles[:,2],2])),:]
   
    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), img_w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), img_h-1)

        if umax<umin or vmax<vmin:
            #print(umax,umin,vmax,vmin)
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]

                if point_depth >depth_buffer[v, u]:
                    depth_vertices[tri]=1
                    depth_buffer[v, u] = point_depth
                    uvimg[v, u, 0:3] = w0*colors[tri[0], :] + w1*colors[tri[1], :] + w2*colors[tri[2], :]
                    uvimg[v, u, 3] =   w0* uvmap[tri[0], 2] + w1*uvmap[tri[1], 2]  + w2*uvmap[tri[2], 2]
    
    if flag_show:
        img_depth=uvimg[:,:,3]/np.max(uvimg[:,:,3])
        img_depth[img_depth==0]=np.NaN
        
        fig = plt.figure(figsize=(8,8),dpi =100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(uvimg[:,:,0:3])
        ax2.imshow(img_depth,cmap=plt.cm.seismic)
        if landmark_idx is not None:
            ax1.scatter(landmarker[:,0],landmarker[:,1],
                        marker='o',c='red',edgecolors='black',s=10,linewidth=0.1)
            ax2.scatter(landmarker[:,0],landmarker[:,1],
                        marker='o',c='black',edgecolors='white',s=10,linewidth=0.1)
        #fig.colorbar(cntr,ax=ax2,label="R")
        plt.show()
        
#        mlab.figure(figure="Coressponding Face",fgcolor=(1., 1., 1.), bgcolor=(0, 0, 0))
#        x, y = np.mgrid[0:img_h, 0:img_w]
#        mlab.surf(x,y,uvimg[:,:,3], warp_scale='auto')
#        mlab.colorbar()
#        mlab.show()
    
    class imguv_info(NamedTuple):
        uvimg:float
        landmarker:float
        I_scale_w:float
        I_scale_h:float
        uv_center:float
        
        
    imguv_info=imguv_info(uvimg,landmarker,I_scale_w,I_scale_h,uv_center)
    
    return imguv_info