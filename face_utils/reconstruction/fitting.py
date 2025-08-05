# -*- coding: utf-8 -*-
"""
Created on 2025-06-06 23:00:09

@author: Robbie
"""

import numpy as np
import pandas as pd
from .transform import (
    similarity_transform,
    estimate_affine_matrix_3d22d,
    P2sRt,
    angle2matrix,
    matrix2angle,
    estimate_affine_matrix_3d23d,
)
from .light import get_normal, sh9
from .render import render_colors_ras, barycentricReconstruction
from .mesh import generate_texture

import matplotlib.pyplot as plt
from matplotlib import cm#,colors
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator
#from scipy.optimize import lsq_linear
from scipy.linalg import block_diag
from typing import NamedTuple
from scipy.optimize import minimize, check_grad, least_squares, nnls, lsq_linear
 
from menpo.transform import Translation, UniformScale, AlignmentSimilarity
plt.rcParams["font.size"]=12 


# ---------------- fit 
def fit_3dpoints(x, X_ind, model,lamb_sp=300,  max_iter = 4):#lamb_ep=30000,
    '''
    Args:
        x: (n, 3) image points
        X_ind: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    '''
    #n_ep = model['expPC'].shape[1]
    #n_tex_para = self.model['texMU'].shape[1]
    n_sp = model['shapePC'].shape[1]
    
    #x = x.copy().T
    
    #-- init
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    #ep = np.zeros((n_ep, 1), dtype = np.float32)
    
    #-------------------- estimate
    X_ind=X_ind.astype(int)
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')
    #print(valid_ind)

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :]
    #expPC = model['expPC'][valid_ind, :n_ep]
    
    #sp = estimate_3dshape(x, shapeMU, shapePC, model['shapeEV'][:n_sp,:],lamb =lamb_sp)# 400)
    
    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) #+ expPC.dot(ep)
        X = np.reshape(X, [int(len(X)/3), 3])
        
        #print(X,x)
        #----- estimate pose
        P = estimate_affine_matrix_3d23d(X,x)
        s, R, t = P2sRt(P)
        #print(s,R,t)
        rx, ry, rz = matrix2angle(R)
        #print(rx, ry, rz )
        #x_pose=np.hstack((x.T, np.ones([x.shape[1],1]))).dot(P.T)
        # print('Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t[0], t[1]))
        #----- estimate shape
        # expression
        #shape = shapePC.dot(sp)
        #shape = np.reshape(shape, [int(len(shape)/3), 3]).T
        #ep = estimate_expression(x, shapeMU, expPC, model['expEV'][:n_ep,:], shape, s, R, t[:2], lamb =lamb_ep)# 200000)
        
        # shape
        #expression = expPC.dot(ep)
        #expression = np.reshape(expression, [int(len(expression)/3), 3]).T
           
        sp = estimate_3dshape(x.T, shapeMU, shapePC, model['shapeEV'][:n_sp,:],s, R, t, lamb =lamb_sp)# 400)
    
    #X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
    #X = np.reshape(X, [int(len(X)/3), 3]).T
    #P1 = estimate_affine_matrix_3d23d(X.T,x.T)
    #s, R, t = P2sRt(P1)
    #P2 = estimate_affine_matrix_3d23d(x.T,X.T) 
    return sp,  s, R, t#ep,

def estimate_3dshape(x, shapeMU, shapePC, shapeEV, s=1, R=angle2matrix([0,0,0]), t3d=np.array([0,0,0]),lamb = 3000):
    '''
    Args:
        x: (3n, 1)
        shapeMU: (3n, 1)
        shapePC: (3n, n_sp)
        shapeEV: (n_sp, 1)
        lambda: regulation coefficient

    Returns:
        shape_para: (n_sp, 1) shape parameters(coefficients)
    '''
    x = x.copy()
    assert(shapeMU.shape[0] == shapePC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = shapePC.shape[1]
    #print(dof)
    n = x.shape[1]
    sigma = shapeEV
    t3d = np.array(t3d)
    #P = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]], dtype = np.float32)
    A = s*R#P.dot(R)
    
    # --- calc pc
    pc_3d = np.resize(shapePC.T, [dof, n, 3]) # 199 x n x 3
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d.dot(A.T.copy()) # 199 x n x 3
    
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 199
    
    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # expression
    #exp_3d = expression
    # 
    b = A.dot(mu_3d) + np.tile(t3d[:, np.newaxis], [1, n]) # 3 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1
    #print(mu_3d.shape)
    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para

def estimate_shape(x, shapeMU, shapePC, shapeEV, expression, s, R, t2d, lamb = 3000):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        shapeMU: (3n, 1)
        shapePC: (3n, n_sp)
        shapeEV: (n_sp, 1)
        expression: (3, n)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient

    Returns:
        shape_para: (n_sp, 1) shape parameters(coefficients)
    '''
    x = x.copy()
    assert(shapeMU.shape[0] == shapePC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = shapePC.shape[1]

    n = x.shape[1]
    sigma = shapeEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    # --- calc pc
    pc_3d = np.resize(shapePC.T, [dof, n, 3]) # 199 x n x 3
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d.dot(A.T.copy()) # 199 x n x 2
    
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 199

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # expression
    exp_3d = expression
    # 
    b = A.dot(mu_3d + exp_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para

def estimate_expression(x, shapeMU, expPC, expEV, shape, s, R, t2d, lamb = 2000):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        shapeMU: (3n, 1)
        expPC: (3n, n_ep)
        expEV: (n_ep, 1)
        shape: (3, n)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient

    Returns:
        exp_para: (n_ep, 1) shape parameters(coefficients)
    '''
    x = x.copy()
    assert(shapeMU.shape[0] == expPC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = expPC.shape[1]

    n = x.shape[1]
    sigma = expEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    A = s*P.dot(R)

    # --- calc pc
    pc_3d = np.resize(expPC.T, [dof, n, 3]) 
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d.dot(A.T) 
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # expression
    shape_3d = shape
    # 
    b = A.dot(mu_3d + shape_3d) + np.tile(t2d[:, np.newaxis], [1, n]) # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    exp_para = np.dot(np.linalg.inv(equation_left), equation_right)
    
    return exp_para


# ---------------- fit 
def fit_points(x, X_ind, model,lamb_sp=300, lamb_ep=30000, max_iter = 4):
    '''
    Args:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    '''
    n_ep = model['expPC'].shape[1]
    #n_tex_para = self.model['texMU'].shape[1]
    n_sp = model['shapePC'].shape[1]
    
    x = x.copy().T
    
    #-- init
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep = np.zeros((n_ep, 1), dtype = np.float32)

    #-------------------- estimate
    X_ind=X_ind.astype(int)
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X)/3), 3]).T
        
        #----- estimate pose
        P = estimate_affine_matrix_3d22d(X.T, x.T)
        s, R, t = P2sRt(P)
        rx, ry, rz = matrix2angle(R)
        # print('Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t[0], t[1]))
        #----- estimate shape
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape)/3), 3]).T
        ep = estimate_expression(x, shapeMU, expPC, model['expEV'][:n_ep,:], shape, s, R, t[:2], lamb =lamb_ep)# 200000)
        
        # shape
        expression = expPC.dot(ep)
        expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        sp = estimate_shape(x, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t[:2], lamb =lamb_sp)# 400)


    return sp, ep, s, R, t



def estimate_2shape(x1, shapeMU1, shapePC1, expression1, s1, R1, t2d1,
                    x2, shapeMU2, shapePC2, expression2, s2, R2, t2d2,
                    shapeEV, lamb = 3000,p=1):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        shapeMU: (3n, 1)
        shapePC: (3n, n_sp)
        shapeEV: (n_sp, 1)
        expression: (3, n)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lambda: regulation coefficient

    Returns:
        shape_para: (n_sp, 1) shape parameters(coefficients)
    '''
    sigma = shapeEV
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    
    x1 = x1.copy()
    assert(shapeMU1.shape[0] == shapePC1.shape[0])
    assert(shapeMU1.shape[0] == x1.shape[1]*3)
    
    x2 = x2.copy()
    assert(shapeMU2.shape[0] == shapePC2.shape[0])
    assert(shapeMU2.shape[0] == x2.shape[1]*3)

    #img1------------------------------------------------------------------------------------
    dof1 = shapePC1.shape[1]
    n1 = x1.shape[1]
    t2d1 = np.array(t2d1)
    A1 = s1*P.dot(R1)
    # --- calc pc
    pc_3d1 = np.resize(shapePC1.T, [dof1, n1, 3]) # 199 x n x 3
    pc_3d1 = np.reshape(pc_3d1, [dof1*n1, 3]) 
    pc_2d1 = pc_3d1.dot(A1.T.copy()) # 199 x n x 2   
    pc1 = np.reshape(pc_2d1, [dof1, -1]).T # 2n x 199

    # --- calc b
    # shapeMU
    mu_3d1 = np.resize(shapeMU1, [n1, 3]).T # 3 x n
    # expression
    exp_3d1 = expression1
    # 
    b1 = A1.dot(mu_3d1 + exp_3d1) + np.tile(t2d1[:, np.newaxis], [1, n1]) # 2 x n
    b1 = np.reshape(b1.T, [-1, 1]) # 2n x 1

    #img2------------------------------------------------------------------------------------
    dof2 = shapePC2.shape[1]
    n2 = x2.shape[1]
    t2d2 = np.array(t2d2)
    A2 = s2*P.dot(R2)
    # --- calc pc
    pc_3d2 = np.resize(shapePC2.T, [dof2, n2, 3]) # 199 x n x 3
    pc_3d2 = np.reshape(pc_3d2, [dof2*n2, 3]) 
    pc_2d2 = pc_3d2.dot(A2.T.copy()) # 199 x n x 2   
    pc2 = np.reshape(pc_2d2, [dof2, -1]).T # 2n x 199

    # --- calc b
    # shapeMU
    mu_3d2 = np.resize(shapeMU2, [n2, 3]).T # 3 x n
    # expression
    exp_3d2 = expression2
    # 
    b2 = A2.dot(mu_3d2 + exp_3d2) + np.tile(t2d2[:, np.newaxis], [1, n2]) # 2 x n
    b2 = np.reshape(b2.T, [-1, 1]) # 2n x 1

   #image1-2-merging--------------------------------------------------------------------------
    equation_left = p*np.dot(pc1.T, pc1) + np.dot(pc2.T, pc2) + lamb * np.diagflat(1/sigma**2)
    # --- solve
    #equation_left = equation_left1+equation_left2
    x1 = np.reshape(x1.T, [-1, 1])
    x2 = np.reshape(x2.T, [-1, 1])
    equation_right = p*np.dot(pc1.T, x1 - b1)+ np.dot(pc2.T, x2 - b2)

    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para

# ---------------- fit 
def fit_points2img(x1,X_ind1, x2, X_ind2, model, lamb_sp = 300*2,lamb_ep = 30000, max_iter = 4, p=1):
    '''
    Args:
        x: (n, 2) image points
        X_ind: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
    Returns:
        sp: (n_sp, 1). shape parameters
        ep: (n_ep, 1). exp parameters
        s, R, t
    '''
    n_ep = model['expPC'].shape[1]
    n_sp = model['shapePC'].shape[1]
    #-- init
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep1 = np.zeros((n_ep, 1), dtype = np.float32)
    ep2 = np.zeros((n_ep, 1), dtype = np.float32)
    
    #-------------------- estimate img1------------------------
    x1 = x1.copy().T
    X_ind_all = np.tile(X_ind1[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')
    shapeMU1 = model['shapeMU'][valid_ind, :]
    shapePC1 = model['shapePC'][valid_ind, :n_sp]
    expPC1 = model['expPC'][valid_ind, :n_ep]
    
    
   #-------------------- estimate img2------------------------
    x2 = x2.copy().T
    X_ind_all = np.tile(X_ind2[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')
    shapeMU2 = model['shapeMU'][valid_ind, :]
    shapePC2 = model['shapePC'][valid_ind, :n_sp]
    expPC2 = model['expPC'][valid_ind, :n_ep]

    for i in range(max_iter):
        #image1------------------------------------------------------
        X1 = shapeMU1 + shapePC1.dot(sp) + expPC1.dot(ep1)
        X1 = np.reshape(X1, [int(len(X1)/3), 3]).T 
        #----- estimate pose
        P1 = estimate_affine_matrix_3d22d(X1.T, x1.T)
        s1, R1, t1 = P2sRt(P1)
        rx1, ry1, rz1 = matrix2angle(R1)
        
        #image2------------------------------------------------------
        X2 = shapeMU2 + shapePC2.dot(sp) + expPC2.dot(ep2)
        X2 = np.reshape(X2, [int(len(X2)/3), 3]).T 
        #----- estimate pose
        P2 = estimate_affine_matrix_3d22d(X2.T, x2.T)
        s2, R2, t2 = P2sRt(P2)
        rx2, ry2, rz2 =matrix2angle(R2)
        # print('Iter:{}; estimated pose: s {}, rx {}, ry {}, rz {}, t1 {}, t2 {}'.format(i, s, rx, ry, rz, t[0], t[1]))
        #----- estimate shape
        # expression
        shape1 = shapePC1.dot(sp)
        shape1 = np.reshape(shape1, [int(len(shape1)/3), 3]).T
        
        shape2 = shapePC2.dot(sp)
        shape2 = np.reshape(shape2, [int(len(shape2)/3), 3]).T
        
        #ep = estimate_2expression(x1, shapeMU1, expPC1, shape1, s1, R1, t1[:2],
        #                          x2,  shapeMU2, expPC2, shape2, s2, R2, t2[:2],
        #                         model['expEV'][:n_ep,:], lamb = 200000*2,p=p)
        ep1 = estimate_expression(x1, shapeMU1, expPC1, model['expEV'][:n_ep,:], 
                                  shape1, s1, R1, t1[:2], lamb = lamb_ep)
        
        ep2 = estimate_expression(x2, shapeMU2, expPC2, model['expEV'][:n_ep,:], 
                                  shape2, s2, R2, t2[:2], lamb = lamb_ep)
        # shape
        expression1 = expPC1.dot(ep1)
        expression1 = np.reshape(expression1, [int(len(expression1)/3), 3]).T
        
        expression2 = expPC2.dot(ep2)
        expression2 = np.reshape(expression2, [int(len(expression2)/3), 3]).T
        sp = estimate_2shape(x1, shapeMU1, shapePC1, expression1, s1, R1, t1[:2], 
                             x2, shapeMU2, shapePC2, expression2, s2, R2, t2[:2], 
                             model['shapeEV'][:n_sp,:],lamb = lamb_sp,p=p)

    return sp, [ep1,ep2], [s1,s2], [R1,R2], [t1,t2]


## TODO. estimate light(sh coeff)
def fit_light(image, vertices,model, lamb =70000, max_iter = 10, flag_show=True):
    '''
    Args:
        x: (2, n). image points (to be fitted)
        textMU: (3n, 1)
        textPC: (3n, n_tx)
        textEV: (n_tx, 1)
        expression: (3, n)
        lambda: regulation coefficient

    Returns:
        text_para: (n_tex\, 1) shape parameters(coefficients)
    '''
    textMU=model['texMU']
    textPC=model['texPC']
    textEV=model['texEV']
    triangles=model['tri']
    #assert(textMU.shape[0] == textPC.shape[0])
    #assert(textPC.shape[1] == textEV.shape[1])
    
    [h, w, c] = image.shape
    texture=textMU.reshape(-1,3)
    #vertices[:,2]=-vertices[:,2]
    
    face_region_idx=np.loadtxt('./model_data/face_region_idx.txt').astype(int)
    face_center_idx=np.where(face_region_idx[:,0]==1)[0]
    vertices2=vertices[face_center_idx,:]
    texture2=texture[face_center_idx,:]
    
    tri_index=np.ones((len(triangles)))*-1
    tri_index[face_center_idx]=np.arange(len(face_center_idx))
    triangles1=triangles.copy()
    for i in range(3):
        triangles1[:,i]=tri_index[triangles1[:,i]]
    
    df_tri=pd.DataFrame(triangles1,columns=list('ABC'))
    #通过~取反，选取不包含数字1的行
    triangles2=np.array(df_tri[~df_tri['A'].isin([-1]) & ~df_tri['B'].isin([-1]) & ~df_tri['C'].isin([-1])]).astype(int)

    
    rendering,pixelCoord,pixelFaces,pixelBarycentricCoords=render_colors_ras(vertices2,triangles2,texture2,h,w)
    #vertices[:,2]=-vertices[:,2]
    
    imgReconstruction = barycentricReconstruction(texture2.T, pixelFaces, pixelBarycentricCoords, triangles2)
    rendering = np.zeros(image.shape)
    rendering[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction
    
    X_ind_all = np.tile(face_center_idx[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')
    
    textPC=textPC[valid_ind,:]
    n_text = textPC.shape[1]
    n_pixls=pixelFaces.size
    recons_textPC=np.empty((n_pixls*3,n_text))
    for ii in range(3):
        pixelVertices = triangles2[pixelFaces, :]*3+ii
        colorMat = textPC.T[:, pixelVertices.flat].reshape((n_text, 3,n_pixls), order = 'F')
        recons_textPC[ii*n_pixls:(ii+1)*n_pixls,:]=np.einsum('ij,kji->ik', pixelBarycentricCoords, colorMat)
        
    img_pixels=image[pixelCoord[:, 0], pixelCoord[:, 1]]
    img_pixels_flatten=np.r_[img_pixels[:,0],img_pixels[:,1],img_pixels[:,2]].reshape(-1,1)*255
    
    recons_textMU=np.r_[imgReconstruction[:,0],imgReconstruction[:,1],imgReconstruction[:,2]].reshape(-1,1)
   
    #===================================estimate inital texCoef===================================
    # --- solve
    equation_left =  np.dot(recons_textPC.T, recons_textPC) + lamb * np.diagflat(1/textEV**2)
    equation_right = np.dot(recons_textPC.T, img_pixels_flatten-recons_textMU)
    tex_para = np.dot(np.linalg.inv(equation_left), equation_right)
   
    # surface normal
    vertexNorms = -get_normal(vertices, triangles)
    B = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
        
   
#    class img_info(NamedTuple):
#        h:int
#        w:int
#        img_pixels: float
#        pixelCoord:int
#        center_idx: int
#        pixelFaces: int
#        pixelBarycentricCoords: float
#        triangles: int
#        B:float
#        
#    #img_info={"pixelFaces":pixelFaces}
#    img_info=img_info(h,w,img_pixels,pixelCoord,
#                      face_center_idx,pixelFaces,
#                      pixelBarycentricCoords,triangles2,B)
    # Set the number of faces for stochastic optimization
#    numRandomFaces = 10000
#    texCoef=np.zeros_like(model['texEV']).flatten()    
#    # Do some cycles of nonlinear least squares iterations, using a new set of random faces each time for the optimization objective
#    cost = np.zeros(10)
#    for i in range(10):
#        randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
#        initTex = least_squares(textureResiduals, texCoef, jac = textureJacobian, 
#                                args =(img_info,model, (200, 1), randomFaces), 
#                                loss = 'soft_l1')
#        texCoef = initTex['x']
#        cost[i] = initTex.cost
#    print(texCoef)        
#    #tex_para=np.zeros_like(model['texEV'])
#    tex_para=texCoef.reshape(-1,1)
    colors=generate_texture(model, tex_para,sh_para=None)
    #print(tex_para.flatten())
    
    imgReconstruction = barycentricReconstruction(colors[face_center_idx,:].T, pixelFaces, pixelBarycentricCoords, triangles2)
    reconstruction2 = np.zeros(rendering.shape)
    reconstruction2[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction
    
    image2=np.zeros_like(image)
    image2[pixelCoord[:, 0], pixelCoord[:, 1]]=image[pixelCoord[:, 0], pixelCoord[:, 1]]

    #===================================estimate inital shCoef===================================

    # Initialize an array to store the barycentric reconstruction of the nine spherical harmonics. 
    #The first dimension indicates the color (RGB).
    I = np.empty((3, pixelFaces.size, 9))
    sh_para = np.empty((9, 3))
    #img_pixels1=image[pixelCoord[:, 0], pixelCoord[:, 1]]
    for c in range(3): 
        I[c, ...] = barycentricReconstruction(B[:,face_center_idx] * colors.T[c, face_center_idx], 
                                               pixelFaces, pixelBarycentricCoords, triangles2)
        sh_para[:, c] = np.dot(np.linalg.inv(I[c, ...].T.dot(I[c, ...])),I[c, ...].T.dot(img_pixels[:, c]))
        
        #np.linalg.lstsq(I[c, ...], img_pixels1[:, c],rcond=-1)[0]#lsq_linear(I[c, ...], img_pixels1[:, c]).x
   
    #textureWithLighting = generate_texture(model, tex_para,sh_para,vertices)

    #imgReconstruction = barycentricReconstruction(textureWithLighting[face_center_idx,:].T, pixelFaces, pixelBarycentricCoords, triangles2)
    #reconstruction3 = np.zeros(rendering.shape)
    #reconstruction3[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction
    
    colors = generate_texture(model,tex_para,sh_para=None) 
    #loss=np.zeros(max_iter+1)
    #loss[0]=np.mean(np.sqrt(np.sum((imgReconstruction*255-img_pixels*255)**2,1)))
    
#    if flag_show:
#        fig = plt.figure(figsize=(4*3,4), dpi=80)
#        ax1 = fig.add_subplot(131)
#        ax2 = fig.add_subplot(132)
#        ax3 = fig.add_subplot(133)
#        ax1.imshow(image2)
#        ax2.imshow(reconstruction2)
#        ax3.imshow(reconstruction3)
    
    #=======================Optimization simultaneously over the texture and lighting parameters=========================
#    #Optimization simultaneously over the texture and lighting parameters
    #texture_display=[]
    #recons_textSh=np.empty((n_pixls*3,1))

    
    # Jointly optimize the texture and spherical harmonic lighting coefficients
    # Set the number of faces for stochastic optimization
#    numRandomFaces = 5000
#    cost = np.zeros(10)
#    #print(tex_para,sh_para)
#    texParam = np.r_[tex_para.flatten(), sh_para.flatten()]
#    for i in range(10):
#        randomFaces = np.random.randint(0, pixelFaces.size, numRandomFaces)
#        initTexLight = least_squares(textureLightingResiduals, texParam, jac = textureLightingJacobian, 
#                                     args = (img_info,model, (100, 1), randomFaces), 
#                                     loss = 'soft_l1', max_nfev = 100)
#        texParam = initTexLight['x']
#        cost[i] = initTexLight.cost
#        
#    #print(cost)
#    
#    numTex=tex_para.size   
#    tex_para = texParam[:numTex].reshape(-1,1)
#    sh_para =  texParam[numTex:].reshape(9, 3)
#    print(tex_para.flatten())
    #print(tex_para,sh_para)
    #colors = generate_texture(model,tex_para,sh_para=None)    
     #texture = generateTexture(vertexCoords, texParam2, m)
#    for i in range(max_iter):  #max_iter
#         mat_sh= np.empty((3, len(vertices))) 
#         for c in range(3):
#             mat_sh[c, :] = np.dot(sh_para[:, c], B)
#             
#         mat_sh=mat_sh.T.reshape(-1,1) 
#         
#         for ii in range(3):
#             pixelVertices = triangles2[pixelFaces, :]*3+ii
#             colorMat = mat_sh.T[:, pixelVertices.flat].reshape((1, 3,n_pixls), order = 'F')
#             recons_textSh[ii*n_pixls:(ii+1)*n_pixls,:]=np.einsum('ij,kji->ik', pixelBarycentricCoords, colorMat)
#         
#         recons_textSh=np.maximum(recons_textSh,0)
#         recons_textMU1=recons_textMU*recons_textSh
#         recons_textPC1=recons_textPC*recons_textSh.repeat(n_text,1)
#         #img_pixels_sh=img_pixels_flatten/recons_textSh
#         #img_pixels_sh=np.maximum(np.minimum(img_pixels_sh,255.0),0.0)#np.maximum(img_pixels_sh,0.0)#
#         #recons_textSh=np.maximum(recons_textSh,0)
#         recons_textMU1=np.minimum(recons_textMU1,255)
#         
#         equation_left = np.dot(recons_textPC1.T, recons_textPC1)+lamb*np.diagflat(1/textEV**2)#/np.dot(recons_textSh.T,recons_textSh)
#         equation_right = np.dot(recons_textPC1.T, img_pixels_flatten-recons_textMU1)
#         tex_para = np.dot(np.linalg.inv(equation_left), equation_right)
#         
#         #print(tex_para.T)
#         colors = generate_texture(model,tex_para,sh_para=None)
#         #texture_display.append(colors)
#         fit_colors=np.zeros_like(img_pixels)
#         for c in range(3): 
#             I[c, ...] = barycentricReconstruction(B[:,face_center_idx] * colors.T[c, face_center_idx], pixelFaces, pixelBarycentricCoords, triangles2)
#             sh_para[:, c] =np.dot(np.linalg.inv(I[c, ...].T.dot(I[c, ...])),I[c, ...].T.dot(img_pixels[:, c]))
#            #np.linalg.lstsq(I[c, ...], img_pixels1[:, c],rcond=-1)[0]# lsq_linear(I[c, ...], img_pixels1[:, c]).x
#             fit_colors[:,c]=I[c, ...].dot(sh_para[:, c])
#             
#         loss[i+1]=np.mean(np.sqrt(np.sum(((img_pixels*255-fit_colors*255)**2),1)))
    
    #print('loss:',np.round(loss,2)) 
    
    if flag_show:
        textureWithLighting = generate_texture(model, tex_para,sh_para,vertices) 
        
        
        imgReconstruction5 = barycentricReconstruction(textureWithLighting[face_center_idx,:].T, pixelFaces, pixelBarycentricCoords, triangles2)
        reconstruction5 = np.zeros(rendering.shape)
        reconstruction5[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction5
    
        imgReconstruction = barycentricReconstruction(colors[face_center_idx,:].T, pixelFaces, pixelBarycentricCoords, triangles2)
        reconstruction4= np.zeros(rendering.shape)
        reconstruction4[pixelCoord[:, 0], pixelCoord[:, 1], :] = imgReconstruction
         
        mat_sh= np.empty((3, len(vertices)))
        for c in range(3):
            mat_sh[c, :] = np.dot(sh_para[:, c], B)
        
        fig = plt.figure(figsize=(4*4,4), dpi=300)
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)
        ax1.imshow(image2)
        image2[pixelCoord[:, 0], pixelCoord[:, 1]]=image[pixelCoord[:, 0], pixelCoord[:, 1]]/barycentricReconstruction(mat_sh[:,face_center_idx], pixelFaces, pixelBarycentricCoords, triangles2)
        image2=np.maximum(np.minimum(image2,1.0),0.0)
      
        ax2.imshow(image2)
        ax3.imshow(reconstruction4)
        ax4.imshow(reconstruction5)
        
        rendering,pixelCoord2,pixelFaces,pixelBarycentricCoords=render_colors_ras(vertices,triangles,textureWithLighting,h,w)
        imgReconstruction = barycentricReconstruction(textureWithLighting.T, pixelFaces, pixelBarycentricCoords, triangles)
        
        fig = plt.figure(figsize=(5*3,4), dpi=80)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(image)
        image[pixelCoord[:, 0],  pixelCoord[:, 1]]=imgReconstruction5
        ax2.imshow(image)
        image[pixelCoord2[:, 0], pixelCoord2[:, 1]]=imgReconstruction
        ax3.imshow(image)
        
        plt.savefig('yangzi.tif')
        
    return colors


line2angle=lambda ba,bc: np.degrees(np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))))


def fit_shaperror(source,target,flag_Near_vertices=True,flag_direction=False):
    #lm_align = AlignmentSimilarity(source.landmarks[group],
    #                           target.landmarks[group]).as_non_alignment()
    #source = lm_align.apply(source)
    points = source.points.copy()#trilist,source.trilist.copy()
    
    #if group is not None:
    #    landmarker=source.landmarks[group].points.copy()
        
    tr = Translation(-1 * source.centre())
    sc = UniformScale(1.0 / np.sqrt(np.sum(source.range() ** 2)), 3)
    prepare = tr.compose_before(sc)

    source = prepare.apply(source)
    target = prepare.apply(target)
    # store how to undo the similarity transform
    restore = prepare.pseudoinverse()
    
    target_vtk = trimesh_to_vtk(target)
    closest_points_on_target = VTKClosestPointLocator(target_vtk)

    #target_tri_normals = target.tri_normals()
    #
    U, tri_indices = closest_points_on_target(source.points.copy())
    Near_vertices=restore.apply(U)

    #fface.plot_2mlabvertex(vertices,colors,triangles,
    #                   Near_vertices,S_colors,F_triangles,
    #                   F_landmarker)
    #fface.plot_mlabvertex(Near_vertices,S_colors,F_triangles,F_landmarker) 

    dist_SF0=np.sqrt(np.sum((points-Near_vertices)**2,1))

    #pal_color=cm.get_cmap( 'seismic',101)(np.linspace(0, 1, 101))*255
    min_depth=np.min(dist_SF0)
    max_depth=np.max(dist_SF0)
    print("Max Error:",max_depth," Min Error:",min_depth," Mean Error:",dist_SF0.mean())
    #dist_SF=(dist_SF0-min_depth)/(max_depth-min_depth)*100
    #print(dist_SF)
    #detal_color=pal_color[dist_SF.astype(int),0:3].astype(np.uint8)
    #detal_color=np.hstack((detal_color,255.*np.ones((len(detal_color),1))))  # NX4 
#    if flag_show:
#        if group is not None:
#            plot_mlabfaceerror(points,dist_SF,trilist,landmarker)
#        else:
#            plot_mlabfaceerror(points,dist_SF,trilist)
    
    if flag_direction:
        # center=points.mean(0).reshape(1,3)
        # dist_target=np.sum((points-center)**2,1)
        # dist_source=np.sum((Near_vertices-center)**2,1)
        # dist_SF0[dist_target>dist_source]=-dist_SF0[dist_target>dist_source]
            
        bc=source.vertex_normals()#get_normal(input_head,triangles)
        ba=points-Near_vertices
        angles=np.array([line2angle(x,y)  for x,y in zip(ba,bc)])
        dist_SF0[angles<90]=-dist_SF0[angles<90]

    if flag_Near_vertices:
        return dist_SF0,Near_vertices,tri_indices 
    else:
        return dist_SF0
    
    
def textureLightingResiduals(texParam, img_info, model,  w = (1, 1), randomFaces = None):
    """
    Energy formulation for fitting texture and spherical harmonic lighting coefficients
    """
    texEV=model['texEV'].flatten() 
    numTex=len(texEV)
    texCoef = texParam[:numTex]
    shCoef = texParam[numTex:].reshape(9, 3)
    texture = generate_texture(model,tex_para=texCoef.reshape(-1,1),sh_para=shCoef,sh=img_info.B)#generateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)
    
    img_pixels_rec = barycentricReconstruction(texture[img_info.center_idx,:].T, 
                                               img_info.pixelFaces, 
                                               img_info.pixelBarycentricCoords, 
                                               img_info.triangles)
    #renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    #renderObj.resetFramebufferObject()
    #renderObj.render()
    #rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]
    #=================show rec_img=====================================
    #rec_img = np.zeros((img_info.h,img_info.w,3))
    #rec_img[img_info.pixelCoord[:, 0], img_info.pixelCoord[:, 1], :] = img_pixels_rec
    #plt.figure(figsize=(4,4), dpi=80)
    #plt.imshow(rec_img)
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        #pixelCoord = pixelCoord[randomFaces, :]
    else:
        numPixels = len(img_info.img_pixels)
    
    img_pixels=img_info.img_pixels[randomFaces,:]
    img_pixels_rec=img_pixels_rec [randomFaces,:]
    #rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    #img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    return np.r_[w[0] / numPixels *255* (img_pixels - img_pixels_rec).flatten('F'),
                 w[1] * texCoef ** 2 / texEV]


def textureLightingJacobian(texParam, img_info, model,  w = (1, 1), randomFaces = None):
    #texParam, img, vertexCoord, sh, model, renderObj, w = (1, 1), randomFaces = None):
    texEV = model['texEV'].flatten() 
    texEvec=model['texPC'].reshape((3, int(len(model['texMU'])/3), 199), order = 'F')
    numTex=len(texEV)
    texCoef = texParam[:numTex]
    shCoef = texParam[numTex:].reshape(9, 3)
    
    vertexColor = generate_texture(model,tex_para=texCoef.reshape(-1,1),sh_para=None)*255 
    #model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    #texture     = generate_texture(model,tex_para=texCoef,sh_para=shCoef)#enerateTexture(vertexCoord, np.r_[texCoef, shCoef.flatten()], model)
    #renderObj.updateVertexBuffer(np.r_[vertexCoord.T, texture.T])
    #renderObj.resetFramebufferObject()
    #renderObj.render()
    #pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[2:]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = img_info.pixelFaces[randomFaces]
        pixelBarycentricCoords = img_info.pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = img_info.pixelFaces.size
        pixelFaces = img_info.pixelFaces
        pixelBarycentricCoords = img_info.pixelBarycentricCoords
        
    pixelVertices =img_info.triangles[pixelFaces, :]
    
    pixelTexture = barycentricReconstruction( vertexColor[img_info.center_idx,:].T, pixelFaces, pixelBarycentricCoords, img_info.triangles)
    pixelSHBasis = barycentricReconstruction( img_info.B[:,img_info.center_idx],    pixelFaces, pixelBarycentricCoords,  img_info.triangles)
    J_shCoef = np.einsum('ij,ik->jik', pixelTexture, pixelSHBasis)
    
    #rec_img = np.zeros((img_info.h,img_info.w,3))
    #rec_img[img_info.pixelCoord[:, 0], img_info.pixelCoord[:, 1], :] = pixelTexture/255
    #plt.figure(figsize=(4,4), dpi=80)
    #plt.imshow(rec_img)
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        pixelTexEvecsCombo = barycentricReconstruction(texEvec[c,img_info.center_idx,:].T,                    pixelFaces, pixelBarycentricCoords, img_info.triangles)
        pixelSHLighting    = barycentricReconstruction(np.dot(shCoef[:, c], img_info.B)[img_info.center_idx], pixelFaces, pixelBarycentricCoords, img_info.triangles)
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = pixelSHLighting * pixelTexEvecsCombo[np.newaxis, ...]
    
    texCoefSide = np.r_[w[0] / numPixels * J_texCoef,             w[1] * np.diag(texCoef / texEV)]
    shCoefSide  = np.r_[w[0] / numPixels * block_diag(*J_shCoef), np.zeros((texCoef.size, shCoef.size))]
    
    return np.c_[texCoefSide, shCoefSide]



def textureResiduals(texCoef,img_info, model, renderObj, w = (1, 1), randomFaces = None):
    #vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texEV=model['texEV'].flatten() 
    vertexColor = generate_texture(model,tex_para=texCoef.reshape(-1,1),sh_para=None)*255 
    img_pixels_rec = barycentricReconstruction(vertexColor[img_info.center_idx,:].T, 
                                               img_info.pixelFaces, 
                                               img_info.pixelBarycentricCoords, 
                                               img_info.triangles)
    
    #renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T])
    #renderObj.resetFramebufferObject()
    #renderObj.render()
    #rendering, pixelCoord = renderObj.grabRendering(return_info = True)[:2]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        #pixelCoord = pixelCoord[randomFaces, :]
    else:
        #numPixels = pixelCoord.shape[0]
         numPixels = len(img_info.img_pixels)
    
    img_pixels=img_info.img_pixels[randomFaces,:]
    img_pixels_rec=img_pixels_rec [randomFaces,:]
    #rendering = rendering[pixelCoord[:, 0], pixelCoord[:, 1]]
    #img = img[pixelCoord[:, 0], pixelCoord[:, 1]]
    
    return np.r_[w[0] / numPixels * 255*(img_pixels_rec - img_pixels).flatten('F'), w[1] * texCoef ** 2 / texEV]

def textureJacobian(texCoef, img_info, model, renderObj, w = (1, 1), randomFaces = None):
    #vertexColor = model.texMean + np.tensordot(model.texEvec, texCoef, axes = 1)
    texEV=model['texEV'].flatten() 
    texEvec=model['texPC'].reshape((3, int(len(model['texMU'])/3), 199), order = 'F')
    #vertexColor = generate_texture(model,tex_para=texCoef.reshape(-1,1),sh_para=None)*255 
    #renderObj.updateVertexBuffer(np.r_[vertexCoord.T, vertexColor.T])
    #renderObj.resetFramebufferObject()
    #renderObj.render()
    #pixelFaces, pixelBarycentricCoords = renderObj.grabRendering(return_info = True)[2:]
    
    if randomFaces is not None:
        numPixels = randomFaces.size
        pixelFaces = img_info.pixelFaces[randomFaces]
        pixelBarycentricCoords = img_info.pixelBarycentricCoords[randomFaces, :]
    else:
        numPixels = img_info.pixelFaces.size
        pixelFaces = img_info.pixelFaces
        pixelBarycentricCoords = img_info.pixelBarycentricCoords
    
    pixelVertices =img_info.triangles[pixelFaces, :]
    
    J_texCoef = np.empty((pixelVertices.size, texCoef.size))
    for c in range(3):
        J_texCoef[c*numPixels: (c+1)*numPixels, :] = barycentricReconstruction(texEvec[c,img_info.center_idx,:].T, 
                                                     pixelFaces, pixelBarycentricCoords, img_info.triangles)
    
    return np.r_[w[0] / numPixels * J_texCoef, w[1] * np.diag(texCoef / texEV)]
