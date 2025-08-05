# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:05:34 2025

@author: Robbie
"""

'''
Functions about transforming mesh(changing the position: modify vertices).
1. forward: transform(transform, camera, project).
2. backward: estimate transform matrix from correspondences.

Preparation knowledge:
transform&camera model:
https://cs184.eecs.berkeley.edu/lecture/transforms-2
Part I: camera geometry and single view geometry in MVGCV
'''

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import numpy as np
import math
from math import cos, sin
import cv2  #cv2.__version__
from .load import load_BFM
from ..registration.model_transform import BFM20092Models

def face2head(BFM_vertices,model_head="ASymHead_Chinese_Shape_1862",
              model_face="ASymBFM_Chinese_Shape_1862", 
              num_PCs=300,
              Whf_title="AsymBFM2ASymHeadAdult_Regression_Matrix_global"):
    
    if BFM_vertices.ptp(0).max()<1000: BFM_vertices=BFM_vertices*1000
    face_vertices,_,f_triangles,face_idx=BFM20092Models(BFM_vertices,colors=None,model_type="SymHead")
    Whf=np.load("./model_data/BFM_Chinese_Model/"+Whf_title+".npy")    
    #num_PCs=300
    
    model1 = load_BFM("./model_data/SymHead_Chinese_Model/"+model_head+".mat",model_type="NoExpression")
    landmark_idx=model1["kpt_ind"]
    triangles=model1['tri']
    mu_texture=model1['texMU'].reshape(-1,3)
    shp_mu1=model1['shapeMU'].reshape(-1,3)
    shp_pc1=model1['shapePC'][:,:num_PCs]  #(170412, 1503)
    shp_ev1=model1['shapeEV'][:num_PCs]
    
    
    model2 = load_BFM("./model_data/BFM_Chinese_Model/"+model_face+".mat",model_type="NoExpression")
    shp_mu2=model2['shapeMU'].reshape(-1,3)
    shp_pc2=model2['shapePC'][:,:num_PCs]  #(159645, 1503)
    shp_ev2=model2['shapeEV'][:num_PCs]
    triangles2=model2['tri']
    mu_texture2=model2['texMU'].reshape(-1,3)
    
    input_face,tform=procrustes_alignment(shp_mu2,BFM_vertices, scaling=True, reflection='best')
    
    head = (shp_mu1+(shp_pc1@(Whf@(shp_pc2.T@((input_face.flatten()-shp_mu2.flatten()).reshape(-1,1))))).reshape(-1,3))/1000
    
    _,tform=procrustes_alignment(face_vertices/1000,head[face_idx,:], scaling=True, reflection='best')
    head2=head.dot(tform['rotation'])+tform['translation'].reshape(1,3)
    
    A=(np.hstack((head[face_idx,:],np.ones((len(face_idx),1))))).T
    x = np.linalg.lstsq( A.T,face_vertices/1000)[0].T
    s,R,t=P2sRt(x)
    head2=similarity_transform(head, s, R, t)
        
    nosetip=face_vertices[face_vertices[:,2].argmax(),:]
    weight0=np.sqrt(np.sum((face_vertices-nosetip.reshape(1,3))**2*np.array([1,1.7,1]),1))
    weight0=np.exp(weight0/weight0.max()*1.5).reshape(1,-1)
    weight1=np.repeat(1-(weight0-weight0.min())/(weight0.max()-weight0.min()),3,axis=0).T
    #plot_mlabfaceerror(face_vertices,weight1[:,0],f_triangles)
    
    head2[face_idx,:]=face_vertices*weight1/1000+(1-weight1)* head2[face_idx,:]
    
    return head2

def face_alignment(F_vertices,S_vertices0,scaling=True,headtype="face",symmetric=True):
    if headtype=="face":
        nose_centeridx=np.loadtxt("./model_data/BFM_2009_Model/nose_centeridx.txt").astype(int)
        pair_idxs=np.loadtxt('./model_data/BFM_2009_Model/face_Symmetry_idx.txt').astype(int)  
        nosetip_idx=8192
        
    elif headtype=="head":
        pair_idxs=np.load("./model_data/SymHead_Chinese_Model/SymHead_Symmetry_idx.npy").astype(int)
        nose_centeridx=np.load("./model_data/SymHead_Chinese_Model/SymHead_CenterLine_idx.npy").astype(int)
        nosetip_idx=41360
       
    S_vertices=S_vertices0-S_vertices0[nosetip_idx,:].reshape(1,3)
    
    A=(np.hstack((F_vertices,np.ones((len(F_vertices),1))))).T
    x = np.linalg.lstsq( A.T,S_vertices)[0].T
    s0,R,t=P2sRt(x)
    F_vertices=similarity_transform(F_vertices,s0,R,t)
        
    #==========================================================
    F_vertices=F_vertices-F_vertices[nosetip_idx,:].reshape(1,3)
        
    #plot_mlabvertex(F_vertices,S_colors,S_triangles)
    A=(np.hstack((F_vertices,np.ones((len(F_vertices),1))))).T
    x = np.linalg.lstsq( A.T,S_vertices)[0].T
    s,x_rotate,t=P2sRt(x)
    vertices_r=np.dot(F_vertices,(x_rotate.T))
    
    vertices_r2=vertices_r.copy()
    
    #------------------------------------------Symmetry Processing-----------------------------------------------
    if symmetric:
        vertices_r2[pair_idxs[:,0],1]=(vertices_r[pair_idxs[:,0],1]+vertices_r[pair_idxs[:,1],1])/2
        vertices_r2[pair_idxs[:,1],1]=(vertices_r[pair_idxs[:,0],1]+vertices_r[pair_idxs[:,1],1])/2
        
        vertices_r2[pair_idxs[:,0],2]=(vertices_r[pair_idxs[:,0],2]+vertices_r[pair_idxs[:,1],2])/2
        vertices_r2[pair_idxs[:,1],2]=(vertices_r[pair_idxs[:,0],2]+vertices_r[pair_idxs[:,1],2])/2
        
        vertices_r2[pair_idxs[:,0],0]= (vertices_r[pair_idxs[:,0],0]+np.abs(vertices_r[pair_idxs[:,1],0]))/2
        vertices_r2[pair_idxs[:,1],0]=-(vertices_r[pair_idxs[:,0],0]+np.abs(vertices_r[pair_idxs[:,1],0]))/2
        
        
        vertices_r2[nose_centeridx[:,0],:]=(vertices_r2[nose_centeridx[:,1],:]+vertices_r2[nose_centeridx[:,2],:])/2 
        vertices_r2[nose_centeridx[:,0],2]= np.max(np.vstack((vertices_r2[nose_centeridx[:,0],2],vertices_r[nose_centeridx[:,0],2])),0)
        
    input_face,tform=procrustes_alignment(S_vertices0, vertices_r2, scaling=scaling, reflection='best')
    #plot_mlabvertex(input_face,colors,triangles)
    return input_face,tform['scale']*s0

def generalized_procrustes_analysis(template,Matshapes,scaling=True, reflection='best'):
    '''
    Performs superimposition on a set of 
    shapes, calculates a mean shape 3nXM  
    Args:
        shapes(a list of 2nx1 Numpy arrays), shapes to
        be aligned
    Returns:
        mean(2nx1 NumPy array), a new mean shape
        aligned_shapes(a list of 2nx1 Numpy arrays), super-
        imposed shapes
    '''
    #initialize Procrustes distance
    current_distance = 0
    template0=template.copy()
    #initialize a mean shape
    
    num_shapes =Matshapes.shape[1]
    
    mean_shape = np.zeros(Matshapes.shape[0])

    #create array for new shapes, add 
    new_Matshapes = np.zeros(Matshapes.shape)
    jj=0
    while True:
        
        #superimpose all shapes to current mean
        for i in range(0, num_shapes):
            new_sh,_ = procrustes_alignment(template, Matshapes[:,i].reshape(-1,3),scaling=scaling, reflection=reflection)
            new_Matshapes[:,i] = new_sh.flatten()
        
        #calculate new mean
        #new_mean = np.mean(new_shapes, axis = 0)
        new_mean=new_Matshapes.mean(1).reshape(-1,3)#np.zeros(vertices[0].shape) 
        
        new_distance = np.sum(np.sqrt((new_mean[:,0] - template[:,0])**2 +
                                      (new_mean[:,1] - template[:,1])**2+ 
                                      (new_mean[:,2] - template[:,2])**2))
        
        #if the distance did not change, break the cycle
        if np.round(new_distance,5) == np.round(current_distance,5):
            break
        
        #align the new_mean to old mean
        new_mean,_ = procrustes_alignment(template0, new_mean,scaling=scaling, reflection=reflection)
        
        #update mean and distance
        template = new_mean
        current_distance = new_distance
        print(jj,new_distance)
        jj=jj+1
        
        if jj==100: break
    
    return new_Matshapes, template 

def procrustes_alignment(X, Y, scaling=True, reflection='best'):
    """
       该函数是matlab版本对应的numpy实现
       Outputs
       ------------
       d：the residual sum of squared errors, normalized according to a measure of the scale of X, ((X - X.mean(0))**2).sum()
       Z：the matrix of transformed Y-values
       tform：a dict specifying the rotation, translation and scaling that maps X --> Y
       """
    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection != 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        #d = 1 - traceTA ** 2
        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        #d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)
    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}
    
    # from menpo.shape import TriMesh,PointCloud
    # from menpo.transform import Translation, UniformScale, AlignmentSimilarity
    # lm_align = AlignmentSimilarity(PointCloud(X),PointCloud(Y)).as_non_alignment()
    # Z = lm_align.apply(PointCloud(Y)).points  
    return Z, tform


def model_correction(S_vertices,T_vertices=None,landmark_idx=None,S_landmark=None,T_landmark=None,nosetip_idx=8192,
                     model_type="BFM",return_Mrotate=False):
    # from menpo.shape import TriMesh, PointCloud
    # from menpo.transform import  AlignmentSimilarity

    # T_landmark=PointCloud(T_landmark)
    # S_landmark=PointCloud(S_landmark)
    
    # lm_align = AlignmentSimilarity(S_landmark,T_landmark,allow_mirror=False).as_non_alignment()
    # S_vertices = lm_align.apply(PointCloud(S_vertices))
    #return S_vertices.points
    if model_type=="BFM":
        S_vertices=S_vertices-S_vertices[nosetip_idx,:].reshape(1,3)
        T_vertices=T_vertices-T_vertices[nosetip_idx,:].reshape(1,3)
    
    if landmark_idx is not None:
        S_landmark=S_vertices[landmark_idx,:]
        T_landmark=T_vertices[landmark_idx,:]
    else:
        S_landmark=S_vertices.copy()
        T_landmark=T_vertices.copy()
        
    A=(np.hstack((S_landmark,np.ones((S_landmark.shape[0],1))))).T
    P = np.linalg.lstsq( A.T,T_landmark)[0].T
    #print(P)
    s,R,t=P2sRt(P)
    angle=np.array(matrix2angle(R))
    print(angle)
        
    #x_rotate=angle2matrix(angle)
    #print(x_rotate)
    S_vertices=np.dot(S_vertices,(R.T))#+t.reshape(1,3)
    
    #S_vertices=S_vertices-S_vertices[nosetip_idx,:].reshape(1,3)
    
    if return_Mrotate:
        return S_vertices,R.T#,t.reshape(1,3)
    else:
        return S_vertices

    
#旋转矩阵-欧拉角(yaw，pitch，roll)
#pitch()：俯仰，将物体绕X轴旋转（localRotationX）
#yaw()：航向，将物体绕Y轴旋转（localRotationY）
#roll()：横滚，将物体绕Z轴旋转（localRotationZ）
def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

## ------------------------------------------ 1. transform(transform, project, camera).
## ---------- 3d-3d transform. Transform obj in world space
def rotate(vertices, angles):
    ''' rotate vertices. 
    X_new = R.dot(X). X: 3 x 1   
    Args:
        vertices: [nver, 3]. 
        rx, ry, rz: degree angles
        rx: pitch. positive for looking down 
        ry: yaw. positive for looking left
        rz: roll. positive for tilting head right
    Returns:
        rotated vertices: [nver, 3]
    '''
    R = angle2matrix(angles)
    rotated_vertices = vertices.dot(R.T)

    return rotated_vertices

def similarity_transform(vertices, s, R, t3d=np.array([0,0,0])):
    ''' similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    Args:(float32)
        vertices: [nver, 3]. 
        s: [1,]. scale factor.
        R: [3,3]. rotation matrix.
        t3d: [3,]. 3d translation vector.
    Returns:
        transformed vertices: [nver, 3]
    '''
    t3d = np.squeeze(np.array(t3d, dtype = np.float32))
    transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    return transformed_vertices


## --------- 3d-2d project. from camera space to image plane
# generally, image plane only keeps x,y channels, here reserve z channel for calculating z-buffer.
def orthographic_project(vertices):
    ''' scaled orthographic projection(just delete z)
        assumes: variations in depth over the object is small relative to the mean distance from camera to object
        x -> x*f/z, y -> x*f/z, z -> f.
        for point i,j. zi~=zj. so just delete z
        ** often used in face
        Homo: P = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    Args:
        vertices: [nver, 3]
    Returns:
        projected_vertices: [nver, 3] if isKeepZ=True. [nver, 2] if isKeepZ=False.
    '''
    return vertices.copy()

def perspective_inverse(model_points,image_points,camera_matrix):
    image_points=image_points.astype(float)
    model_points=model_points.astype(float)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                              image_points,
                                                              camera_matrix,
                                                              dist_coeffs,
                                                              flags=cv2.SOLVEPNP_ITERATIVE) 
    rotation_matrix=cv2.Rodrigues(rotation_vector)[0] 
    
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
#    (vertices_pers2d, jacobian) = cv2.projectPoints(vertices,
#                                                  rotation_vector,
#                                                  translation_vector,
#                                                  camera_matrix,
#                                                  dist_coeffs)
#    fig = plt.figure(figsize=(4*3,4), dpi=80)
#    ax1 = fig.add_subplot(121)
#    ax2 = fig.add_subplot(122)
#    ax1.imshow(image)
#    ax1.scatter(lm[:,0],lm[:,1],
#             c='r',s=10, linewidths=0.5, edgecolors="k",alpha=1,marker="o",zorder=2)
#    ax2.imshow(image)
#    ax2.scatter(vertices_pers2d[model['kpt_ind'],:,0],vertices_pers2d[model['kpt_ind'],:,1],
#             c='r',s=10, linewidths=0.5, edgecolors="k",alpha=1,marker="o",zorder=2)      
#    
#    

    return rotation_matrix,translation_vector
        
def perspective_project(vertices,camera_matrix,rotation,translation,
                        fov=10.0,focal=None,image_size=256,#cam_pos=10.0,
                        aspect_ratio = 1., near = 0.1,far = 10.):
    
    if camera_matrix is None:
        if focal is None: 
            focal_length=image_size/2/np.tan(fov/2/180*np.pi)
        center =image_size/2 
        #cam_pos = 10
        camera_matrix = np.array([[focal_length, 0,            center],
                                  [0,            focal_length, center],
                                  [0,            0,            1]], dtype="double")# projection matrix
   
    
    vertices_r = vertices.dot(rotation.T)  # CHJ: R has been transposed
    vertices_t = vertices_r + translation.reshape([1, 3])
    #vertices_t[:,2] = cam_pos-vertices_t[:,2]
    
    aug_projection = vertices_t.dot(camera_matrix.T)

    #print(aug_projection)
    #exit()
    face_projection = aug_projection[:, 0:2] / aug_projection[:,2:]
    # CHJ_WARN: I do this for visualization
    z_buffer = aug_projection[:, 2:]# cam_pos -  # CHJ: same as the z of  face_shape_t

    return face_projection, z_buffer
    

# =============================================================================
# def perspective_project(vertices, fovy0, aspect_ratio = 1., near = 0.1, far = 1000.):
#     ''' perspective projection.
#     Args:
#         vertices: [nver, 3]
#         fovy: vertical angular field of view. degree.
#         aspect_ratio : width / height of field of view
#         near : depth of near clipping plane
#         far : depth of far clipping plane
#     Returns:
#         projected_vertices: [nver, 3] 
#     '''
#     fovy = np.deg2rad(fovy0)
#     fovy = near*np.tan(fovy/2)
#    # bottom = -top 
#     #right = top*aspect_ratio
#     #left = -right
# 
#     #-- homo
#     P = np.array([[fovy*aspect_ratio, 0,        0,                0],
#                  [0,                 fovy,      0,                0],
#                  [0,                 0, (far+near)/(near-far), 2*far*near/(near-far)],
#                  [0,                 0,         -1,               0]])
#     vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1)))) # [nver, 4]
#     projected_vertices = vertices_homo.dot(P.T)
#     projected_vertices = projected_vertices/projected_vertices[:,3:]
#     projected_vertices = projected_vertices[:,:3]
#     projected_vertices[:,2] = -projected_vertices[:,2]
# 
#     #-- non homo. only fovy
#     # projected_vertices = vertices.copy()
#     # projected_vertices[:,0] = -(near/right)*vertices[:,0]/vertices[:,2]
#     # projected_vertices[:,1] = -(near/top)*vertices[:,1]/vertices[:,2]
#     return projected_vertices
# =============================================================================


def estimate_affine_matrix_3d23d(X, Y):
    ''' Using least-squares solution 
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    '''
    X_homo = np.hstack((X, np.ones([X.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(X_homo, Y)[0].T # Affine matrix. 3 x 4
    return P
    
def estimate_affine_matrix_3d22d(X, x):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. in MVGCV 
        Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
        x_homo = X_homo.dot(P_Affine)
    Args:
        X: [n, 3]. corresponding 3d points(fixed)
        x: [n, 2]. n>=4. 2d points(moving). x = PX
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    '''
    X = X.T; x = x.T
    assert(x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert(n >= 4)

    #--- 1. normalization
    # 2d points
    mean = np.mean(x, 1) # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x**2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3,3), dtype = np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean*scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1) # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    #m = X_homo[:3,:] - X
    average_norm = np.mean(np.sqrt(np.sum(X**2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4,4), dtype = np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean*scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n*2, 8), dtype = np.float32);
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])
 
    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype = np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))
    return P_Affine
    
def P2sRt(P):
    ''' decompositing camera matrix P
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation. 
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t

#Ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert(isRotationMatrix)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz

def to_image(vertices, h, w, is_perspective = False):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis. 
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        projected_vertices: [nver, 3]  
    '''
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:,0] = image_vertices[:,0]*w/2
        image_vertices[:,1] = image_vertices[:,1]*h/2
    # move to center of image
    image_vertices[:,0] = image_vertices[:,0] + w/2
    image_vertices[:,1] = image_vertices[:,1] + h/2
    # flip vertices along y-axis.
    image_vertices[:,1] = h - image_vertices[:,1] - 1
    return image_vertices