# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:53:40 2020

@author: Peter_Zhang
"""
#import os
#os.environ["QT_API"] = "pyqt"
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
#from pylab import savefig
import random

import pandas as pd
import colorsys
from pandas.api.types import CategoricalDtype
import skimage.io as io  
from skimage import transform
from math import floor
#from matplotlib import cm#,colors
plt.rcParams["font.size"]=12 
#conda install -c conda-forge/label/cf202003 mayavi


def make_video(files,x_range,y_range,fps=1, img_size0=1000,img_size1=1200,outfile='./Figures/movie.mp4'):
    import imageio
   
    range_max=np.max(np.r_[x_range,y_range])
    img_range=[]
    for img_path in files:
        img = io.imread(img_path).astype(float)/255
        img_i,img_j=np.where(img[:,:,0]<1)    
        img_range.append(np.max([img_j.ptp(),img_i.ptp()]))
    print("image max range:",np.max(img_range))    
    
    images=[]
    for i,img_path in zip(range(len(files)),files):
        #img=imageio.imread(folder+file)
        img = io.imread(img_path)
        img_i,img_j=np.where(img[:,:,0]>0)
        
        mesh_range=np.max([x_range[i],y_range[i]])
        img_scale=mesh_range/range_max
        #print(img_scale)
        img_extract=img[img_i.min():img_i.max(),img_j.min():img_j.max(),:]
        
        img_extract=transform.resize(img_extract, (img_size0,int(img_j.ptp()/img_i.ptp()*img_size0)))
        img_rescale=transform.rescale(img_extract, (img_scale,img_scale,1))
        
        #print(img_path,img_extract.shape)
        #print(img_rescale.shape)
        img_h,img_w,_=img_rescale.shape
        img_centerh=int(img_h/2)
        img_centerw=int(img_w/2)
        
        img_new=np.zeros((img_size1,img_size1,3))
        img_new[int(img_size1/2)-img_centerh:int(img_size1/2)-img_centerh+img_h,
                int(img_size1/2)-img_centerw:int(img_size1/2)-img_centerw+img_w,:]=img_rescale
    
        images.append((img_new*255).astype(np.uint8))
    
    imageio.mimwrite(outfile, images, fps=fps)    
    print("Video is produced sucessfully")
    
def floored_percentage(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}%'.format(digits, floor(val) / 10 ** digits)
#RGB2Hex
def RGB2Hex(red,green,blue): return '#%02x%02x%02x' % (red,green,blue)  

def Subjects_AgeSex_Display(df_ages,df_sex,xrange=None,yrange=None,figsize=[7,5]):
    df_sex=np.array([np.char.lower(x) for x in df_sex])
    df_ages=df_ages.astype(int)
    
    df_agesf=df_ages[df_sex=="f"]
    df_agesm=df_ages[df_sex=="m"]
    
    if xrange is None:
        xrange=[df_ages.min(),df_ages.max()]
        
    histf=plt.hist(df_agesf,bins=np.arange(xrange[0],xrange[1]+1))
    histm=plt.hist(df_agesm,bins=np.arange(xrange[0],xrange[1]+1))
    
    num_female=len(df_agesf)
    num_male=len(df_agesm)
    num=num_female+num_male
    percef=floored_percentage(num_female/num, 2)
    percem=floored_percentage(num_male/num, 2)
    print("num_female:",num_female,"num_male:",num_male)
    
    
    #background information: sex and ages             
    fig=plt.figure(figsize=(figsize[0],figsize[1]),dpi=300)
    #plt.hist(df_agesf,bins=np.arange(15,100),edgecolor='k',stacked=True)
    #plt.hist(df_agesm,bins=np.arange(15,100),edgecolor='k',stacked=True)
    plt.axes([0.1,0.1,0.8,0.8])
    plt.bar(histf[1][:-1],histf[0],facecolor="#FD8D62",edgecolor='k',width=1,linewidth=0.5,label='Female')
    plt.bar(histf[1][:-1],histm[0],facecolor="#66C2A5",edgecolor='k',width=1,linewidth=0.5,bottom=histf[0],label='Male')
    plt.xlabel("Ages")
    plt.ylabel("Count")#" between average and standard faces")
    plt.xlim(xrange[0]-1,xrange[1])
    plt.xticks(range(3,18), range(3,18))#, rotation='vertical')
    
    if yrange is not None:
        plt.ylim(yrange[0],yrange[1])
        
    plt.axes([0.6,0.65,0.2,0.2])
    plt.pie(x=[num_female,num_male],colors=["#FD8D62","#66C2A5"],
            labels=["Female\n"+percef,"Male\n"+percem],startangle=90)
    
    plt.show()

    
def Mesh_Groups_Display(x_range,y_range,imgs_plusredus,img_size=None,num_pca=5,range_max=None,display_type=None,layout="rectangle",flag_bg="w"):
    
    if layout=="matrix":
       x_range=x_range.reshape(2,-1) 
       x_range=np.r_[x_range[0,0],x_range[:,1:5].flatten()]
       
       y_range=y_range.reshape(2,-1) 
       y_range=np.r_[y_range[0,0],y_range[:,1:5].flatten()]
       
       imgs_plusredus=imgs_plusredus.reshape(2,-1) 
       imgs_plusredus=np.r_[imgs_plusredus[:1,0],imgs_plusredus[:,1:5].flatten()]
       #print(imgs_plusredus.shape)
    
    x_range=np.array(x_range)
    y_range=np.array(y_range)
    if range_max is None:
        range_max=np.max([x_range.max(),y_range.max()])
    
    img_range=[]
    for img_path in imgs_plusredus:
        img = io.imread(img_path).astype(float)/255
        if flag_bg=="b":   
            img_i,img_j=np.where(img[:,:,0]>0)   
        else:
            img_i,img_j=np.where(img[:,:,0]<1)    
        img_range.append(np.max([img_j.ptp(),img_i.ptp()]))
    print("image max range:",np.max(img_range))    
    
    if img_size is None:   
        img_size=int(np.ceil(np.max(img_range)/100)*100)#900
    #elif img_size<np.max(img_range):
    #    img_size=int(np.ceil(np.max(img_range)/100)*100)#900
    
    num_sub=len(imgs_plusredus)
    num_row=int(num_sub/num_pca) if num_sub/num_pca<=int(num_sub/num_pca) else int(num_sub/num_pca)+1
    
    if layout=="rectangle":
        img_all=np.ones((img_size*num_row,img_size*num_pca,3))
        for img_path,i in zip(imgs_plusredus,range(num_sub)):
            img = io.imread(img_path).astype(float)/255
            if flag_bg=="b":   
                img_i,img_j=np.where(img[:,:,0]>0)
            else:
                img_i,img_j=np.where(img[:,:,0]<1)
            
            mesh_range=np.max([x_range[i],y_range[i]])
            img_scale=mesh_range/range_max
            #print(img_scale)
            img_extract=img[img_i.min():img_i.max(),img_j.min():img_j.max(),:]
            
            if img_j.ptp()<img_i.ptp():
                img_extract=transform.resize(img_extract, (img_size,int(img_j.ptp()/img_i.ptp()*img_size)))
            else:
                img_extract=transform.resize(img_extract, (int(img_i.ptp()/img_j.ptp()*img_size),img_size))
            #img_extract=transform.resize(img_extract, (img_size,int(img_j.ptp()/img_i.ptp()*img_size)))
            img_rescale=transform.rescale(img_extract, (img_scale,img_scale,1))
            
            #print(img_path,img_extract.shape)
            #print(img_rescale.shape)
            img_h,img_w,_=img_rescale.shape
            img_centerh=int(img_h/2)
            img_centerw=int(img_w/2)
            
            if flag_bg=="b": 
                img_new=np.zeros((img_size,img_size,3))
            else:
                img_new=np.ones((img_size,img_size,3))
                
            img_new[int(img_size/2)-img_centerh:int(img_size/2)-img_centerh+img_h,
                    int(img_size/2)-img_centerw:int(img_size/2)-img_centerw+img_w,:]=img_rescale
            #plt.imshow(img_new)
            #plt.show()
            #img_news.append(img_new)
            k=np.mod(i,num_pca)
            j=int(np.ceil((i+1)/num_pca))
            
            img_all[img_size*(j-1):img_size*j,img_size*k:img_size*(k+1),:]=img_new
            
            if (display_type=="Mean_plus_PCA") & (k==0) & (j==2):
                img_all[:,img_size*k:img_size*(k+1),:]=1.
                img_all[img_size*(j-1)-int(img_size/2):img_size*j-int(img_size/2),img_size*k:img_size*(k+1),:]=img_new
                
    elif layout=="matrix":
        print("PCA result: mean+first 4 PCs") 
        img_all=np.zeros((img_size*3,img_size*3,3))
        img_y=[1,0,0,0,1,2,2,2,1]#[1,0,2,0,2,0,2,1,1]
        img_x=[1,0,1,2,2,2,1,0,0]#[1,0,2,1,1,2,0,2,0]
        
        num_sub=9
        for img_path,i in zip(imgs_plusredus,range(num_sub)):
            img = io.imread(img_path).astype(float)/255
            if flag_bg=="b":   
                img_i,img_j=np.where(img[:,:,0]>0)   
            else:
                img_i,img_j=np.where(img[:,:,0]<1)    
            
            mesh_range=np.max([x_range[i],y_range[i]])
            img_scale=mesh_range/range_max
            #print(img_scale)
            
            img_extract=img[img_i.min():img_i.max(),img_j.min():img_j.max(),:]
            
            if img_j.ptp()<img_i.ptp():
                img_extract=transform.resize(img_extract, (img_size,int(img_j.ptp()/img_i.ptp()*img_size)))
            else:
                img_extract=transform.resize(img_extract, (int(img_i.ptp()/img_j.ptp()*img_size),img_size))
            img_rescale=transform.rescale(img_extract, (img_scale,img_scale,1))
            
            #print(img_path,img_extract.shape)
            #print(img_rescale.shape)
            img_h,img_w,_=img_rescale.shape
            img_centerh=int(img_h/2)
            img_centerw=int(img_w/2)
            
            if flag_bg=="b": 
                img_new=np.zeros((img_size,img_size,3))
            else:
                img_new=np.ones((img_size,img_size,3))
            img_new[int(img_size/2)-img_centerh:int(img_size/2)-img_centerh+img_h,
                    int(img_size/2)-img_centerw:int(img_size/2)-img_centerw+img_w,:]=img_rescale
           
            img_all[img_size*img_y[i]:img_size*(img_y[i]+1),img_size*img_x[i]:img_size*(img_x[i]+1),:]=img_new
            
        
    fig, ax = plt.subplots(figsize=(3*num_pca,3),dpi =300)
    plt.imshow(img_all)
    #plt.xticks([])
    #plt.yticks([])
    plt.show()
    
    return img_all
    
   
def Motion_colorCircle(Vs=0.6,Ss_max=10,Ss_step=0.1,method="plotnine"):
    #Color Motion Legend: Motion_colorCircle()
    #Vs=0.6#np.repeat(0.6,len(Rs))
    Hs=np.arange(0,360,1)
    Ss=np.arange(0,Ss_max,Ss_step)/Ss_max
    #Ss=Ss/Ss.max()
    
    RGBs=[]
    x=[]
    y=[]
    for h in Hs:
        for s in Ss:
            rgb=(np.array(colorsys.hsv_to_rgb(h/360,s,Vs))*255).astype(np.uint8)
            RGBs.append( "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2]))
            x.append(s*np.cos(h*np.pi/180))
            y.append(s*np.sin(h*np.pi/180))   
    df_colors=pd.DataFrame(dict(x=x,y=y,RGB=RGBs))
    
    if method=="plotnine":
        from plotnine import ggplot,geom_point,scale_fill_manual,coord_fixed,aes
        df_colors['order'] = range(len(df_colors))
        df_colors['order'] = df_colors['order'].astype(CategoricalDtype(categories=df_colors['order'],ordered=True))
        
        base_plot=(ggplot(df_colors,aes(x='x',y='y',fill='order'))+
                    geom_point(shape='o',color='none',show_legend=False)+
                    scale_fill_manual(values=df_colors['RGB'])+
                    coord_fixed(ratio = 1)
                    )
        print(base_plot)
    else:
        plt.figure(figsize=(6,6), dpi=300)
        plt.scatter(x,y,s=3,c=RGBs,marker='o')
        plt.axis('square')
        plt.show()
        
def Caculate_motion(dist_vertices,max_R=10,Vs_range=[0.6,0.9]):
    R=np.sqrt(dist_vertices[:,0]**2+dist_vertices[:,1]**2)
    
    dist_vertices[dist_vertices[:,0]==0.,0]=0.0000000001
    angle=np.arctan(dist_vertices[:,1]/dist_vertices[:,0])
    
    angle_direction=np.array([ np.pi if x<0 else 0 for x in dist_vertices[:,0]]) 
    angle=angle+angle_direction
    
    Vs=dist_vertices[:,2]/max_R
    Vs[Vs<Vs_range[0]]=Vs_range[0]
    Vs[Vs>Vs_range[1]]=Vs_range[1]
    
    Ss=R/max_R
    Ss[Ss>1.0]=1.0
    motion_color=np.array([colorsys.hsv_to_rgb(h, s, v) for h,s,v in zip(angle/(2*np.pi),Ss,Vs)])
    
    #motion_color=np.array([colorsys.hsv_to_rgb(h, s, 0.6) for h,s in zip(angle/(2*np.pi),R/max_R)])
    return motion_color


def compute_cumulative_error(errors, bins):
    r"""
    Computes the values of the Cumulative Error Distribution (CED).

    Parameters
    ----------
    errors : `list` of `float`
        The `list` of errors per image.
    bins : `list` of `float`
        The values of the error bins centers at which the CED is evaluated.

    Returns
    -------
    ced : `list` of `float`
        The computed CED.
    """
    n_errors = len(errors)
    return [np.count_nonzero([errors <= x]) / n_errors for x in bins]

def cumulative_error_plot(errors,error_range=None,bins=10,fillcolor="skyblue",
                          xlabel="Mean per-vertex registering error (mm)",
                          ylabel="Proportion of subject",flag_plot=True):
    from matplotlib.ticker import PercentFormatter
    
    errors=np.array(errors)
    if error_range is None:
        error_range = [errors.min(), errors.max(), errors.ptp()/bins]
    x_axis = np.arange(error_range[0], error_range[1], error_range[2])
    y_cumsum = compute_cumulative_error(errors, x_axis)
    df_xy = pd.DataFrame({'x':x_axis,'y':y_cumsum})
        
    # Line_plot1=(ggplot(df_xy, aes('x', 'y') )+
    #   geom_line( size=0.5)+
    #   #geom_point(shape='o',size=3,color="black",fill="#F78179") +
    #   #xlim(0,2.5)+
    #   ##xlab("Mean per-vertex reconstruction error (mm)")+
    #   ylab("Proportion of subjects")+
    #   theme(legend_position="none",
    #         aspect_ratio =1,
    #         figure_size = (4, 4),
    #         dpi = 100))
    # print(Line_plot1)
    if flag_plot:
        fig = plt.figure(figsize=(4,4), dpi=300)
        # plt.rcParams["font.size"]=15 
        plt.fill_between(df_xy.x,df_xy.y,color=fillcolor, alpha=0.85)
        plt.plot(df_xy.x,df_xy.y,color='k',linewidth=1.5,alpha=1.)#,label="full face")
        #plt.plot(df_errors_inner.x,df_errors_inner.y,color='b',linewidth=1.5,alpha=1.,label="inner face")
        #plt.xlim(0,7)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.legend(loc='lower right')
        
        plt.show()

    return df_xy

#matplotlib-二维散点
def plot_matuvmap(uvmap,colors=None,landmarker=None,cmap=plt.cm.Spectral_r,flag_equal=True,xlabel="xlabel",ylabel="ylabel",colorbar_label="colorbar_label"):
    
    fig=plt.figure(figsize=(5.8,5),dpi =300)
    if colors is not None:
        color_hex=[RGB2Hex(x[0],x[1],x[2]) for x in colors.astype(np.uint8)]
        plt.scatter(uvmap[:,0],uvmap[:,1],edgecolors='none',s=1.0,c=color_hex)
    else:
        scatter=plt.scatter(uvmap[:,0],uvmap[:,1],c=uvmap[:,2],
                    edgecolors='none',s=1.0,cmap=cmap)
        fig.colorbar(scatter,label=colorbar_label)
        
    if landmarker is not None:
        if len(landmarker.shape)==1:
            plt.scatter(uvmap[landmarker,0],uvmap[landmarker,1],
                    marker='o',c='red',edgecolors='black',s=20,linewidth=1)
        elif len(landmarker.shape)==2:
            plt.scatter(landmarker[:,0],landmarker[:,1],
                    marker='o',c='red',edgecolors='black',s=20,linewidth=1)
    if flag_equal:
        plt.axis('equal')
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.rcParams["font.size"]=15
    plt.show()

#matplotlib-三维散点
def plot_matscatter(points,colors,randselect=True,Num_select=5000):
    if randselect:
        slice_idx = random.sample(range(points.shape[0]), Num_select)
    else:
        slice_idx =range(points.shape[0])
    
    color_hex=[RGB2Hex(x[0],x[1],x[2]) for x in colors[slice_idx,:].astype(np.uint8)]
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d') 
    ax.scatter(points[slice_idx,0],points[slice_idx,1],points[slice_idx,2], 
               marker='.',s=15,c=color_hex)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    

#mlab-三维人脸    
def plot_mlabpoints(points,colors=None,landmarks=None,scale_factor=1, panel_XYZ=None,
                    opacity=1,azimuth=0, elevation=180,xyz_ranges=None,title=None,
                    out_file=None,figsize=(1000,1000)):
    '''
    Plot vertices and triangles from a PlyData instance. Assumptions:
        `ply' has a 'vertex' element with 'x', 'y', and 'z'
            properties;
        `ply' has a 'face' element with an integral list property
            'vertex_indices', all of whose elements have length 3.
    '''
    N=points.shape[0]
    ones = np.ones(N)
    scalars = np.arange(N) # Key point: set an integer for each point
    
    mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1.,1.,1.),size=figsize)
    
    if colors is not None:
        if (colors.dtype=='<f4') | (isinstance(colors[0,0],float)):
            #print(1)
            colors=np.hstack(((colors*255).astype(np.uint8),255.*np.ones((points.shape[0],1)))) 
        else:
            colors=np.hstack((colors.astype(np.uint8),255.*np.ones((colors.shape[0],1)))) 
        
  
    nodes=mlab.quiver3d(points[:,0], points[:,1], -points[:,2],ones, ones, ones, scalars=scalars, mode='sphere',scale_factor=scale_factor)
    #nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.glyph.color_mode = 'color_by_scalar' 
    #nodes.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, colors.shape[0])
    if  colors is not None:
        nodes.module_manager.scalar_lut_manager.lut.number_of_colors  = points.shape[0]
        nodes.module_manager.scalar_lut_manager.lut.table = colors
    
    if landmarks is not None:
        N=landmarks.shape[0]
        ones = np.ones(N)
        scalars = np.arange(N) # Key point: set an integer for each point
        nodes=mlab.quiver3d(landmarks[:,0], landmarks[:,1], -landmarks[:,2],ones, ones, ones, mode='sphere',scale_factor=scale_factor*3)

    if panel_XYZ is not None:
        mlab.surf(panel_XYZ[0],panel_XYZ[1],panel_XYZ[2],color=(0.7, 0.7, 0.7),#colormap='RdYlBu', 
                         warp_scale=0.3, representation='surface', line_width=0.5)#wireframe

    mlab.view(azimuth=azimuth, elevation=elevation,focalpoint=[ 0, 0, 0])
    
    if title is not None:
        mlab.title(title)
    if xyz_ranges is not None:
        mlab.axes(ranges=xyz_ranges,y_axis_visibility=True)
    mlab.draw()
    if out_file is not None:
        mlab.savefig(filename=out_file)
    mlab.show()

    
#mlab-三维人脸    
def plot_mlabvertex(vertex,colors,triangles,landmarker=None,lines=None,plane=None,
                    opacity=1,azimuth=0, elevation=180,scale_factor=0.02,
                    representation='surface',xyz_ranges=None,title=None,bgcolor=(1.,1.,1.),
                    out_file=None,figsize=(1000,1000)):
    '''
    Plot vertices and triangles from a PlyData instance. Assumptions:
        `ply' has a 'vertex' element with 'x', 'y', and 'z'
            properties;
        `ply' has a 'face' element with an integral list property
            'vertex_indices', all of whose elements have length 3.
    '''
    mlab.figure(fgcolor=(0., 0., 0.), bgcolor=bgcolor,size=figsize)
    
    if colors is not None:
        if (colors.dtype=='<f4') | (isinstance(colors[0,0],float)):
            #print(1)
            colors=np.hstack(((colors*255).astype(np.uint8),255.*np.ones((colors.shape[0],1)))) 
        else:
            colors=np.hstack((colors.astype(np.uint8),255.*np.ones((colors.shape[0],1)))) 
        
        mesh = mlab.triangular_mesh(vertex[:,0], vertex[:,1], -vertex[:,2], triangles,
                                    scalars=np.arange(colors.shape[0]), 
                                    opacity=opacity,representation=representation)
        mesh.module_manager.scalar_lut_manager.lut.number_of_colors = colors.shape[0]
        mesh.module_manager.scalar_lut_manager.lut.table = colors
        
    else:
        mesh = mlab.triangular_mesh(vertex[:,0], vertex[:,1], -vertex[:,2], triangles, 
                                    opacity=opacity,
                                    colormap="bone",
                                    representation=representation)
    
    if landmarker is not None:
        landmarker=landmarker.reshape(-1,3)
        mlab.points3d(landmarker[:,0], landmarker[:,1], -landmarker[:,2],
                      color=(1,1,0.2), mode='sphere', scale_factor=scale_factor*vertex.max())
    
    if lines is not None:
        for i in range(lines.shape[0]):
            mlab.plot3d(lines[i,:,0], lines[i,:,1], -lines[i,:,2], color=(0,0,0), tube_radius=0.1)
        
    # if plane is not None:
    #     mlab.surf(plane[0],plane[1],-plane[2],warp_scale=1, color=(0.1,0.2,0.3))#
        
    if plane is not None:
        mlab.surf(plane[0],plane[1],plane[2],color=(0.7, 0.7, 0.7),#colormap='RdYlBu', 
                         warp_scale=0.3, representation='surface', line_width=0.5)#wireframe    
       
    mlab.view(azimuth=azimuth, elevation=elevation,focalpoint=[ 0, 0, 0])
    
    if title is not None:
        mlab.title(title)
    if xyz_ranges is not None:
        mlab.axes(ranges=xyz_ranges,y_axis_visibility=True)
    mlab.draw()
    
    if out_file is not None:
        mlab.savefig(filename=out_file)
    
    mlab.show()
    
    
#mlab-三维人脸    
#reference:https://stackoverflow.com/questions/52114655/mayavi-fixed-colorbar-for-all-scenes
def plot_mlabfaceerror(vertex,dist_error,triangles,
                       F_vertex=None,F_colors=None,F_triangles=None,landmarker=None, colormap="RdYlBu",reverse_lut=True,
                      opacity=1,azimuth=0, elevation=180,colormap_range=None,colorbar=True,
                       fgcolor=(0, 0, 0), bgcolor=(1., 1., 1.),title=None,
                      representation='surface',out_file=None,figsize=(1000,1000)):
    
    #print("Peter===================================================================================")                      
    figure=mlab.figure(figure="Fittting Error Analysis",fgcolor=fgcolor, bgcolor=bgcolor,size=figsize)
    #camera_light3 = figure.scene.light_manager.lights[2]
    #camera_light3.intensity = 0.25
    
    s=mlab.triangular_mesh(vertex[:,0], vertex[:,1], -vertex[:,2], triangles,
                                colormap=colormap,
                                scalars=dist_error, 
                                opacity=1.,representation=representation)
    s.module_manager.scalar_lut_manager.reverse_lut = reverse_lut
    if colormap_range is not None:
        s.module_manager.scalar_lut_manager.data_range = colormap_range
    if landmarker is not None:
        landmarker=landmarker.reshape(-1,3)
        mlab.points3d(landmarker[:,0], landmarker[:,1], -landmarker[:,2],
                      color=(1,0,0), mode='sphere', scale_factor=0.015*vertex.max())
        
        
    if F_vertex is not None:
        if F_colors is not None:
            if (F_colors.dtype=='<f4') | (isinstance(F_colors[0,0],float)):
                F_colors=np.hstack(((F_colors*255).astype(np.uint8),255.*np.ones((F_colors.shape[0],1)))) 
            else:
                F_colors=np.hstack((F_colors.astype(np.uint8),255.*np.ones((F_colors.shape[0],1)))) 
                
            mesh = mlab.triangular_mesh(F_vertex[:,0], F_vertex[:,1], -F_vertex[:,2], 
                                    F_triangles,
                                    scalars=np.arange(F_colors.shape[0]), 
                                    transparent=True,#alpha=0.9,
                                    opacity=opacity,representation="surface")
            
            mesh.module_manager.scalar_lut_manager.lut.number_of_colors = F_colors.shape[0]
            mesh.module_manager.scalar_lut_manager.lut.table = F_colors
        else:
            mlab.triangular_mesh(F_vertex[:,0], F_vertex[:,1], -F_vertex[:,2], F_triangles, 
                                    opacity=opacity,
                                    colormap="bone",
                                    representation="surface")
    
    #print(figure.scene.light_manager)
    #l0 = s.scene.light_manager.lights[0]
        
    if colorbar:
        mlab.colorbar()
    mlab.view(azimuth=azimuth, elevation=elevation,focalpoint=[ 0, 0, 0])

    mlab.draw()
    
    if title is not None:
        mlab.title(title)
    if out_file is not None:
        mlab.savefig(filename=out_file)
        
    mlab.show()


    
    
def plot_2mlabvertex(S_vertex,S_colors,S_triangles,
                     F_vertex,F_colors,F_triangles,
                     F_landmarker=None,azimuth=0, elevation=180,
                     representation='wireframe',opacity=1,title=None,scale_factor=None,
                     out_file=None,figsize=(1000,1000)):
    mlab.figure(figure="Coressponding Face",fgcolor=(0., 0., 0.), bgcolor=(1.0, 1.0, 1.0),size=figsize)
    
    if S_colors is not None:
        if (S_colors.dtype=='<f4') | (isinstance(S_colors[0,0],float)):
            S_colors=np.hstack(((S_colors*255).astype(np.uint8),255.*np.ones((S_colors.shape[0],1)))) 
        else:
            S_colors=np.hstack((S_colors.astype(np.uint8),255.*np.ones((S_colors.shape[0],1))))
            
        mesh = mlab.triangular_mesh(S_vertex[:,0], S_vertex[:,1], -S_vertex[:,2], 
                                S_triangles,
                                scalars=np.arange(S_colors.shape[0]), 
                                opacity=1,representation='surface')
        mesh.module_manager.scalar_lut_manager.lut.number_of_colors = S_colors.shape[0]
        mesh.module_manager.scalar_lut_manager.lut.table = S_colors
    else:
        mlab.triangular_mesh(S_vertex[:,0], S_vertex[:,1], -S_vertex[:,2], S_triangles, 
                                    opacity=1,
                                    colormap="bone",
                                    representation='surface')
    
    #mlab.triangular_mesh(S_vertex[:,0], S_vertex[:,1], -S_vertex[:,2], S_triangles,
    #                     color=(1, 0, 0),opacity=1,representation='wireframe')
    #plot_vertex(vertices_fitted,std_colors,triangles_fitted,None,1)
    # if representation=='wireframe':
    #     mlab.triangular_mesh(F_vertex[:,0], F_vertex[:,1],-F_vertex[:,2],
    #                      F_triangles,
    #                      color=(1, 1, 1),transparent=True,opacity=0.1,representation='wireframe')
        
        
#else:
    if F_colors is not None:
        if (F_colors.dtype=='<f4') | (isinstance(F_colors[0,0],float)):
            F_colors=np.hstack(((F_colors*255).astype(np.uint8),255.*np.ones((F_colors.shape[0],1)))) 
        else:
            F_colors=np.hstack((F_colors.astype(np.uint8),255.*np.ones((F_colors.shape[0],1)))) 
            
        mesh = mlab.triangular_mesh(F_vertex[:,0], F_vertex[:,1], -F_vertex[:,2], 
                                F_triangles,
                                scalars=np.arange(F_colors.shape[0]), 
                                #transparent=True,#alpha=0.9,
                                opacity=opacity,representation=representation)
        mesh.module_manager.scalar_lut_manager.lut.number_of_colors = F_colors.shape[0]
        mesh.module_manager.scalar_lut_manager.lut.table = F_colors
    else:
        mlab.triangular_mesh(F_vertex[:,0], F_vertex[:,1], -F_vertex[:,2], F_triangles, 
                                opacity=opacity,
                                colormap="bone",
                                representation=representation)
        
    if F_landmarker is not None:
        F_landmarker=F_landmarker.reshape(-1,3)
        scale_factor=0.015*S_vertex.max() if scale_factor is None else scale_factor
        mlab.points3d(F_landmarker[:,0], F_landmarker[:,1], -F_landmarker[:,2],
                       color=(1,0,0), mode='sphere', scale_factor=scale_factor)
    mlab.view(azimuth=azimuth, elevation=elevation,focalpoint=[ 0, 0, 0])
    
    if title is not None:
        mlab.title(title)
        
    mlab.draw()
    if out_file is not None:
        mlab.savefig(filename=out_file)
    mlab.show()


def plot_2dlandmark(img,df_pos):
     fig = plt.figure(figsize=(4*3,4), dpi=80)
     ax1 = fig.add_subplot(121)
     ax2 = fig.add_subplot(122)
     ax1.imshow(img)
     ax2.imshow(img)
     if len(df_pos)==68:
         LEFT_EYE_POINTS = list(range(42, 48))
         RIGHT_EYE_POINTS = list(range(36, 42))
         LEFT_BROW_POINTS = list(range(22, 27))
         RIGHT_BROW_POINTS = list(range(17, 22))
         NOSE_POINTS = list(range(27, 36))
         MOUTH_POINTS = list(range(48, 67))
         JAW_PROFILE=list(range(0,17))
         plt_groups=[JAW_PROFILE,
                    LEFT_EYE_POINTS,RIGHT_EYE_POINTS,
                    LEFT_BROW_POINTS,RIGHT_BROW_POINTS,
                    NOSE_POINTS,
                    MOUTH_POINTS]
         for group in plt_groups:
             ax2.plot(df_pos[group,0],df_pos[group,1],linewidth=0.75,zorder=1,c='k')
             ax2.scatter(df_pos[group,0],df_pos[group,1],
                            c='r',s=10, linewidths=0.5, edgecolors="k",alpha=1,marker="o",zorder=2)
     else:
         ax2.scatter(df_pos[:,0],df_pos[:,1],
                            c='r',s=10, linewidths=0.5, edgecolors="k",alpha=1,marker="o",zorder=2)
               