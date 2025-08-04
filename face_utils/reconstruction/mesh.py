# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 00:37:40 2020

@author: Peter_Zhang
"""


from .light import get_normal, sh9
from .visualize import plot_mlabfaceerror
#from .fitting import fit_shaperror
#from .render import get_point_weight

from menpo.shape import TriMesh,PointCloud
from scipy import sparse#,linalg#,dok_matrix
from scipy.sparse.linalg import inv
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
#from pyntcloud import PyntCloud
#import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
#from stl import mesh
import trimesh
import open3d as o3d #http://www.open3d.org/docs/release/tutorial/Basic/mesh.html
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator
#from scipy.optimize import lsq_linear
from scipy.optimize import minimize, check_grad, least_squares, nnls, lsq_linear
from menpo.transform import Translation, UniformScale, AlignmentSimilarity

def fit_shaperror(source,target,flag_Near_vertices=True):
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
    
    if flag_Near_vertices:
        return dist_SF0,Near_vertices,tri_indices 
    else:
        return dist_SF0
    
def get_point_weight(point, tri_points):
    ''' Get the weights of the position
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster.(used)
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        w0: weight of v0
        w1: weight of v1
        w2: weight of v3
     '''
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2

def connect_tris(points):
    """    Takes a 2d point cloud as input   """
    if not isinstance(points, pd.DataFrame):  ## Vrrifies input is a dataframe. Converts if it isn't
        points = pd.DataFrame(points)
    points.loc()[:, 2] = pow(points.loc()[:, 0], 2) + pow(points.loc()[:,1], 2)  ## Performs parabolic lifting of 2D points to 3D

    #plt.scatter(points.loc()[:,0], points.loc()[:,1]) ## Make a plot of the original 2D points

    hull = ConvexHull(points)  ## Performs convex hull on the lifted 3D points

    points = hull.points[:, 0:-1]     ## Puts points in an easier format to work with

    xy_norm = (0, 0, 1) ## Standard normal vector of the XY Plane
        ## Find dot product of this and each simplex, and check that it falls between 0 and -1, exclusive
        ## This would mean the simplex is facing the origonal  coordinate hyperplane and its edges should be inlcuded in the dimensional reduction

    ## Create an array of the dot products of the XY-Plane Normal and al the Facet Normals
    idx = np.shape(hull.equations)[1] - 1
    dot_facets = np.dot(hull.equations[:,0:idx], xy_norm)

    ## Create a list of all the 3D simplices that have dot-product with XY plane between -1 and 0, exclusive
    bool_simplices = hull.simplices[((dot_facets < 0) & (dot_facets > -1)), :]


    ## Reduce list of triangles to only ones that contain the first point.
    ## This step assumes the local optimality of the triangulation given that the neighborhood is large enough
    bool_simplices = [[num for num in row] for row in bool_simplices if 0 in row]
    return bool_simplices


def points2mesh(points_xyz,neighbors_k=15,flag_process=True):

    arr = np.asarray(points_xyz)
    
    ## 1. Build BallTree
    neighbors = BallTree(arr, metric='braycurtis')

    ## Build KD-Tree
    dist, ind = neighbors.query(arr, k=neighbors_k)

    ## Declare PCA model
    pca = PCA(n_components = 2)
    
    ## Declare an array to store all the tiangles for the mesh
    triangles = []

    ##  Iterate through each point
    for i in range(len(arr)):
        #print("loop {}".format(i))

        ## Build list of X - nearest points
        ## PCA reduce to 2D
        temp = pca.fit_transform(arr[ind[i]])
        ## Pass 2-D points to twoD.py
        ## Returns all triangles that touch the point arr[1]
        _triangles = connect_tris(temp)

        for tri in _triangles:
            temp = [ind[i][num] for num in tri]
            #triangles.append([[item for item in arr[num]] for num in temp])
            triangles.append(temp)
    #wolf0 = mesh.Mesh(np.zeros(np.shape(triangles)[0], dtype=mesh.Mesh.dtype))
    triangles=np.array(triangles)
    #for idx, face in enumerate(triangles):
    #    wolf0.vectors[idx] = face

    #wolf0.save('ball_wolf_braycurtis.stl')
    #print("Completed Initial STL Generation")
    if flag_process:
        wolf = trimesh.Trimesh(np.asarray(points_xyz), np.asarray(triangles))
    #wolf = trimesh.load("ball_wolf_braycurtis.stl")
    #print("Beginning mesh repair.")

        print("Euler Number of mesh before repair = {}".format(wolf.euler_number))

        wolf.process()
        wolf.remove_degenerate_faces()
        wolf.remove_duplicate_faces()
        wolf.merge_vertices()
        wolf.remove_infinite_values()
        wolf.remove_unreferenced_vertices()
        wolf.fix_normals()
        wolf.fill_holes()
    
        print("Euler Number of mesh after repair = {}".format(wolf.euler_number))
    
        #wolf.export(file_obj="processed_wolf2.stl")
        points_xyz=np.array(wolf.vertices)
        triangles=np.array(wolf.faces)
    
    return points_xyz,triangles


def reduce_mesh(S_vertices,S_colors=None,S_triangles=None,num_points=4000,neighbors_k=15,flag_process=False):
    T_mesh =o3d.pybind.geometry.TriangleMesh()
    T_mesh.vertices = o3d.pybind.utility.Vector3dVector(S_vertices)
    T_mesh.triangles= o3d.pybind.utility.Vector3iVector(S_triangles)
    # T_mesh =o3d.cuda.pybind.geometry.TriangleMesh()
    # T_mesh.vertices = o3d.cuda.pybind.utility.Vector3dVector(S_vertices)
    # T_mesh.triangles= o3d.cuda.pybind.utility.Vector3iVector(S_triangles)
    ##T_mesh.PointCloud = o3d.utility.Vector3dVector(T_vertices0)
    
    target_points=T_mesh.sample_points_poisson_disk(number_of_points=num_points)
    T_sampoints=np.asarray(target_points.points)
    
    #print(T_sampoints.shape)

    S_vertices1,S_triangles1=points2mesh(T_sampoints,flag_process=True)
    
    if S_colors is not None:
        S_colors=S_colors.astype(float)
        target1=TriMesh(S_vertices, S_triangles)
        source1=TriMesh(S_vertices1, S_triangles1)
    
        dist_SF1,Near_vertices1,tri_indices1 =fit_shaperror(source1,target1,flag_Near_vertices=True)
    
    
        vert_distidx1=S_triangles[tri_indices1,:]
    
        vert_weight1=np.array([ get_point_weight(x,S_vertices[idx,:]) for idx,x in zip(S_triangles[tri_indices1,:],S_vertices1)])
    
    
        colorMat = S_colors.T[:, vert_distidx1.flat].reshape((S_colors.shape[1], 3, len(vert_distidx1)), order = 'F')
        S_colors1=np.einsum('ij,kji->ik', vert_weight1, colorMat)

    else:
        S_colors1=np.repeat([100,100,100],len(S_vertices1)).reshape(-1,3)/255
        
    return S_vertices1,S_colors1,S_triangles1


def generate_vertices(model, shape_para, exp_para=None):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1) 
        Returns:
            vertices: (nver, 3)
        '''
        if exp_para is not None:
            vertices = model['shapeMU'] + model['shapePC'].dot(shape_para) + model['expPC'].dot(exp_para)
        else:
            vertices = model['shapeMU'] + model['shapePC'].dot(shape_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T

        return vertices  
    
def generate_texture(model, tex_para,sh_para=None,vertices=None,sh=None,flag_shape=False):
    '''
    Args:
        tex_para: (n_tex_para, 1)
    Returns:
        colors: (nver, 3)
    '''
    colors = model['texMU'] + model['texPC'].dot(tex_para)##*model['texEV'])
    colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F')/255.  
    
    if flag_shape:
        #colors =np.array([1.0,153.0,252.0]).reshape(3,1).repeat(colors.shape[1],axis=1)/255
        colors =np.array([246.0,110.0,184.0]).reshape(3,1).repeat(colors.shape[1],axis=1)/255
    if sh_para is not None:
        
         if sh is None:
             vertexNorms = -get_normal(vertices, model['tri'])
             sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
         
         colors0=colors.copy()
         colors = np.empty((3, colors0.shape[1]))
         for c in range(3):
             colors[c, :] = np.dot(sh_para[:, c], sh) * colors0[c, :]
               
    colors=np.maximum(np.minimum(colors,1.0),0.0)
    return colors.T


def crop_mesh(mesh_idx,vertices, colors, triangles,X_ind0=None):
    
    import pandas as pd
    
    num_vert=len(vertices)
    
    tri_index=np.ones((num_vert))*-1
    
    tri_index[mesh_idx]=np.arange(len(mesh_idx))
    
    tem_triangles=triangles.astype(int).copy()
    for i in range(3):
        tem_triangles[:,i]=tri_index[tem_triangles[:,i]]
        
    df_tri=pd.DataFrame(tem_triangles,columns=list('ABC'))
    #通过~取反，选取不包含数字1的行
    crop_triangles=np.array(df_tri[~df_tri['A'].isin([-1]) & ~df_tri['B'].isin([-1]) & ~df_tri['C'].isin([-1])]).astype(int)

    crop_vertices=vertices[mesh_idx,:]
    crop_colors=colors[mesh_idx,:]
    
    if X_ind0 is not None:
        X_ind=(tri_index[X_ind0]).astype(int)
    
    if X_ind0 is not None:
        return crop_vertices,crop_colors,crop_triangles,X_ind
    else:
        return crop_vertices,crop_colors,crop_triangles

def build_laplacian( X, T ):
    
    #import scipy.special as sc
    nv=X.shape[0]
    #nf=T.shape[0]
    func_dist=lambda x: np.sqrt(np.sum(x*x,1))
    
    L1=func_dist(X[T[:,1],:]-X[T[:,2],:])
    L2=func_dist(X[T[:,0],:]-X[T[:,2],:])
    L3=func_dist(X[T[:,0],:]-X[T[:,1],:])
    #EL=np.c_[L1,L2,L3]
    A1=(L2*L2+L3*L3-L1*L1)/(2*L2*L3)
    A2=(L1*L1+L3*L3-L2*L2)/(2*L1*L3)
    A3=(L1*L1+L2*L2-L3*L3)/(2*L1*L2)
    A=np.arccos(np.c_[A1,A2,A3])   #np.arccos(0.3)
    
    I = np.r_[ T[ :, 0 ],T[ :,1],T[ :, 2]]
    J = np.r_[ T[ :, 1 ],T[ :,2],T[ :, 0]]
    #S=  np.r_[range(nf),range(nf),range(nf)]+1
    #E=np.zeros((nv,nv))
    #E[I,J]=S
    #E = sparse.coo_matrix((S,(I,J)),shape=(nv,nv))
    #I,J=np.where(E>0)
    #BV = dok_matrix((nv,1), dtype=np.float32)
    
    S = 0.5 * 1/np.tan(np.r_[ A[:, 2],A[ :, 0],A[ :, 1]])
    In = np.r_[ I, J, I, J]
    Jn = np.r_[ J, I, I, J]
    Sn = np.r_[-S,-S, S, S]
    W = sparse.csc_matrix((Sn,(In,Jn)),shape=(nv,nv))#sparse( In, Jn, Sn, nv, nv );
    return W

def mesoscopic_deformation( X,I,T,  params=None,flag_show=True):
        
    params_dt=0.8
    params_eta = .17
    params_step_size = .1;
    
    
    w_mu = .1;
    params_w_mu = w_mu;
    params_w_s = 1 - w_mu;
    
    nv=X.shape[0]
    W=build_laplacian( X, T )    
    
    #VV=np.linalg.inv(sparse.eye(X.shape[0]) + 10 * W).dot(X)
    VV=inv(sparse.eye(nv,format="csc") + 10 * W).dot(X)
    #np.linalg.inv((sparse.eye(X.shape[0]) + 10 * W).toarray()).dot(X)
    normals = get_normal(VV, T).T
    
    I = (np.mean(I, 1) - .5 ) * 2
    mu = inv(sparse.eye(nv,format="csc") + params_dt * W).dot(I)#( speye( size( W, 1 ) ) + params.dt * W )\I;
    mu = I + mu
        
    if flag_show:
        plot_mlabfaceerror(X,-mu,T) #Mesoscopic displacement.
    
    #regularizing displacement term====================================================
    delta_s = W.dot(VV)*normals.T 
    #==================================================================================
    
    
    II =  np.r_[ T[ :, 0 ],T[ :,1],T[ :, 2]]
    JJ =  np.r_[ T[ :, 1 ],T[ :,2],T[ :, 0]]
    #nrm2 =lambda x: np.sqrt(np.sum(x*x,1))# @( x )( sqrt( sum( x .^ 2, 2 ) ) );
    dst2 =lambda x,y:np.sqrt(np.sum((x-y)**2,1))# @( x, y )( nrm2( x - y ) );
    expdst2 = lambda x,y: np.exp(-np.sqrt(np.sum((x-y)**2,1)))#@( x, y )( exp(  - dst2( x, y ) ) );
    SS = np.c_[ expdst2( VV[T[ :, 1 ], : ], VV[T[ :, 0 ], : ] ),
                expdst2( VV[T[ :, 2 ], : ], VV[T[ :, 1 ], : ] ),
                expdst2( VV[T[ :, 2 ], : ], VV[T[ :, 0 ], : ] )]
    
    In = np.r_[ II,JJ ]
    Jn = np.r_[ JJ,II ]
    Sn = np.r_[ SS,SS ].T.flatten()
    WW = sparse.csc_matrix((Sn,(In,Jn)),shape=(nv,nv))#WW = sparse( In, Jn, Sn, nv, nv );
    
    a=np.array(np.sum( WW, 1)).flatten()
    a[a<=0.00000000001]=np.inf
    WW1 = sparse.diags(1/a).dot (WW)
    #print(WW1.toarray())
    
    ffunc = lambda x, y, n: ( 1 - np.abs( np.multiply(x-y,n).sum(axis=1) ) / dst2( x, y ))
    #@( x, y, n )( 1 - abs( dot( x - y, n, 2 ) ) ./ dst2( x, y ) );
    SS = np.c_[ffunc(VV[T[:,1],:],VV[T[:,0],:], normals[ :,T[:,1]].T ),
               ffunc(VV[T[:,2],:],VV[T[:,1],:], normals[ :,T[:,2]].T ),
               ffunc(VV[T[:,2],:],VV[T[:,0],:], normals[ :,T[:,0]].T )]
    
    Sn = np.r_[ SS,SS ].T.flatten()
    
    WW2 =WW1.dot(sparse.csc_matrix((Sn,(In,Jn)),shape=(nv,nv)))# WW * sparse( In, Jn, Sn, nv, nv );
    #print(WW2.toarray())
    
    #data-driven term: delta_mu=================================================================
    delta_mu = params_eta * WW2.dot(mu)
    delta_mu = delta_mu.reshape(-1,1)* normals.T 
    #=================================================================================
    
    VV_t = VV + params_step_size * ( params_w_mu * delta_mu + params_w_s * delta_s ) / ( params_w_mu + params_w_s )

    return VV_t
    
if __name__ == "__main__":
    
    params_dt=0.8
    params_eta = .17
    params_step_size = .1;
    w_mu = .1;
    
    params_w_mu = w_mu;
    params_w_s = 1 - w_mu;

    X=np.array([1,2,1,2,3,4,
                2,3,4,5,6,7,
                1,3,5,3,1,7]).reshape(3,-1).T
    T=np.array([1,2,3,4,5,6,
                2,3,4,5,6,1,
                4,5,6,1,3,2]).reshape(3,-1).T-1
    I=np.array([1,2,1,2,3,4,
                2,3,4,5,6,7,
                1,3,5,3,1,7]).reshape(3,-1).T/7
    
    nv=X.shape[0]
    W=build_laplacian( X, T )    
    
    VV=inv(sparse.eye(X.shape[0],format="csc") + 10 * W).dot(X)
    #np.linalg.inv((sparse.eye(X.shape[0]) + 10 * W).toarray()).dot(X)
    
    normals = get_normal(VV, T).T
    
    I = ( np.mean(I, 1) - .5 ) * 2
    mu = inv(sparse.eye(X.shape[0],format="csc") + params_dt * W).dot(I)#( speye( size( W, 1 ) ) + params.dt * W )\I;
    mu = I + mu
    
    delta_s = W.dot(VV)
    
    II =  np.r_[ T[ :, 0 ],T[ :,1],T[ :, 2]]
    JJ =  np.r_[ T[ :, 1 ],T[ :,2],T[ :, 0]]
    #nrm2 =lambda x: np.sqrt(np.sum(x*x,1))# @( x )( sqrt( sum( x .^ 2, 2 ) ) );
    dst2 =lambda x,y:np.sqrt(np.sum((x-y)**2,1))# @( x, y )( nrm2( x - y ) );
    expdst2 = lambda x,y: np.exp(-np.sqrt(np.sum((x-y)**2,1)))#@( x, y )( exp(  - dst2( x, y ) ) );
    SS = np.c_[ expdst2( VV[T[ :, 1 ], : ], VV[T[ :, 0 ], : ] ),
                expdst2( VV[T[ :, 2 ], : ], VV[T[ :, 1 ], : ] ),
                expdst2( VV[T[ :, 2 ], : ], VV[T[ :, 0 ], : ] )]
    
    In = np.r_[ II,JJ ]
    Jn = np.r_[ JJ,II ]
    Sn = np.r_[ SS,SS ].T.flatten()
    WW = sparse.coo_matrix((Sn,(In,Jn)),shape=(nv,nv))#WW = sparse( In, Jn, Sn, nv, nv );
    
    
    WW1 = sparse.diags( 1 / ( np.sum( WW.toarray(), 1) )).dot (WW)
    #print(WW1.toarray())
    
    ffunc = lambda x, y, n: ( 1 - np.abs( np.multiply(x-y,n).sum(axis=1) ) / dst2( x, y ))
    #@( x, y, n )( 1 - abs( dot( x - y, n, 2 ) ) ./ dst2( x, y ) );
    SS = np.c_[ffunc(VV[T[:,1],:],VV[T[:,0],:], normals[ :,T[:,1]].T ),
               ffunc(VV[T[:,2],:],VV[T[:,1],:], normals[ :,T[:,2]].T ),
               ffunc(VV[T[:,2],:],VV[T[:,0],:], normals[ :,T[:,0]].T )]
    
    Sn = np.r_[ SS,SS ].T.flatten()
    WW2 =WW1.dot(sparse.coo_matrix((Sn,(In,Jn)),shape=(nv,nv)))# WW * sparse( In, Jn, Sn, nv, nv );
    #print(WW2.toarray())
    
    delta_mu = params_eta * WW2.dot(mu)
    delta_mu = delta_mu.reshape(-1,1)* normals.T
    VV_t = VV + params_step_size * ( params_w_mu * delta_mu + params_w_s * delta_s ) / ( params_w_mu + params_w_s )

    
