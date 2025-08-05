'''
functions about rendering mesh(from 3d obj to 2d image).
only use rasterization render here.
Note that:
1. Generally, render func includes camera, light, raterize. Here no camera and light(I write these in other files)
2. Generally, the input vertices are normalized to [-1,1] and cetered on [0, 0]. (in world space)
   Here, the vertices are using image coords, which centers on [w/2, h/2] with the y-axis pointing to oppisite direction.
Means: render here only conducts interpolation.(I just want to make the input flexible)

Preparation knowledge:
z-buffer: https://cs184.eecs.berkeley.edu/lecture/pipeline

Author: Yao Feng 
Mail: yaofeng1995@gmail.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
#from time import time
import math
import matplotlib.pyplot as plt

from .visualize import plot_mlabvertex, plot_matuvmap

def pointTriangleDistance(TRI, P):
    '''
     TRI = np.array([[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0]])
    P = np.array([0.5,-0.3,0.5])
    dist, pp0 = pointTriangleDistance(TRI,P)

    Parameters
    ----------
    TRI : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.

    Returns
    -------
    dist : TYPE
        DESCRIPTION.
    PP0 : TYPE
        DESCRIPTION.

    '''
    # function [dist,PP0] = pointTriangleDistance(TRI,P)
    # calculate distance between a point and a triangle in 3D
    # SYNTAX
    #   dist = pointTriangleDistance(TRI,P)
    #   [dist,PP0] = pointTriangleDistance(TRI,P)
    #
    # DESCRIPTION
    #   Calculate the distance of a given point P from a triangle TRI.
    #   Point P is a row vector of the form 1x3. The triangle is a matrix
    #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
    #   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
    #   to the triangle TRI.
    #   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
    #   closest point PP0 to P on the triangle TRI.
    #
    # Author: Gwolyn Fischer
    # Release: 1.0
    # Release date: 2002-09-02
    # Release: 1.1 Fixed Bug because of normalization
    # Release: 1.2 Fixed Bug because of typo in region 5 20101013
    # Release: 1.3 Fixed Bug because of typo in region 2 20101014

    # Possible extention could be a version tailored not to return the distance
    # and additionally the closest point, but instead return only the closest
    # point. Could lead to a small speed gain.

    # Example:
    # %% The Problem
    # P0 = [0.5 -0.3 0.5]
    #
    # P1 = [0 -1 0]
    # P2 = [1  0 0]
    # P3 = [0  0 0]
    #
    # vertices = [P1; P2; P3]
    # faces = [1 2 3]
    #
    # %% The Engine
    # [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0)
    #
    # %% Visualization
    # [x,y,z] = sphere(20)
    # x = dist*x+P0(1)
    # y = dist*y+P0(2)
    # z = dist*z+P0(3)
    #
    # figure
    # hold all
    # patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8)
    # plot3(P0(1),P0(2),P0(3),'b*')
    # plot3(PP0(1),PP0(2),PP0(3),'*g')
    # surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
    # view(3)

    # The algorithm is based on
    # "David Eberly, 'Distance Between Point and Triangle in 3D',
    # Geometric Tools, LLC, (1999)"
    # http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    #
    #        ^t
    #  \     |
    #   \reg2|
    #    \   |
    #     \  |
    #      \ |
    #       \|
    #        *P2
    #        |\
    #        | \
    #  reg3  |  \ reg1
    #        |   \
    #        |reg0\
    #        |     \
    #        |      \ P1
    # -------*-------*------->s
    #        |P0      \
    #  reg4  | reg5    \ reg6
    # rewrite triangle in normal form
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    #print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    dist = np.sqrt(sqrdistance)

    PP0 = B + s * E0 + t * E1
    return dist, PP0

def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        bool: true for in triangle
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

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

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

def rasterize_triangles(vertices, triangles, h, w):
    ''' 
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        h: height
        w: width
    Returns:
        depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle). 
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # h, w is the size of rendering
    '''
    # initial 
    depth_buffer = np.zeros([h, w]) + 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
    triangle_buffer = np.zeros([h, w], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = np.zeros([h, w, 3], dtype = np.float32)  # 
    
    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2]) # barycentric weight
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                if point_depth < depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    triangle_buffer[v, u] = i
                    barycentric_weight[v, u, :] = np.array([w0, w1, w2])

    return depth_buffer, triangle_buffer, barycentric_weight


def render_colors_ras(vertices, triangles, colors, h, w, c = 3):
    ''' render mesh with colors(rasterize triangle first)
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
        c: channel
    Returns:
        image: [h, w, c]. rendering.
    '''
    assert vertices.shape[0] == colors.shape[0]

    depth_buffer, triangle_buffer, barycentric_weight = rasterize_triangles(vertices, triangles, h, w)

    triangle_buffer_flat = np.reshape(triangle_buffer, [-1]) # [h*w]
    barycentric_weight_flat = np.reshape(barycentric_weight, [-1, c]) #[h*w, c]
    weight = barycentric_weight_flat[:, :, np.newaxis] # [h*w, 3(ver in tri), 1]

    colors_flat = colors[triangles[triangle_buffer_flat, :], :] # [h*w(tri id in pixel), 3(ver in tri), c(color in ver)]
    colors_flat = weight*colors_flat # [h*w, 3, 3]
    colors_flat = np.sum(colors_flat, 1) #[h*w, 3]. add tri.

    image = np.reshape(colors_flat, [h, w, c])
    # mask = (triangle_buffer[:,:] > -1).astype(np.float32)
    # image = image*mask[:,:,np.newaxis]
    pixelCoord=np.array(np.where(triangle_buffer>-1)).T
    pixelFaces=triangle_buffer[triangle_buffer>-1]
    pixelBarycentricCoords=barycentric_weight.reshape(-1,3)[triangle_buffer.flatten()>-1,:]

    return image,pixelCoord,pixelFaces,pixelBarycentricCoords



def render_colors(vertices, triangles, colors, h, w, c = 3,flag_depth=False):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
    Returns:
        image: [h, w, c]. 
    '''
    assert vertices.shape[0] == colors.shape[0]
    
    triangles=triangles[np.argsort(-(vertices[triangles[:,0],2]+vertices[triangles[:,1],2]+vertices[triangles[:,2],2])),:]
    # initial 
    image = np.ones((h, w, c))
    depth_buffer = np.zeros([h, w]) - 999999.
    
    depth_vertices=np.zeros(vertices.shape[0])
    
    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]

                if point_depth > depth_buffer[v, u]:
                    depth_vertices[tri]=1
                    depth_buffer[v, u] = point_depth
                    image[v, u, :] = w0*colors[tri[0], :] + w1*colors[tri[1], :] + w2*colors[tri[2], :]
    if flag_depth:
        return image,depth_vertices
    else:
        return image

def barycentricReconstruction(vertices, pixelFaces, pixelBarycentricCoords, indexData):
    """Reconstructs per-pixel attributes from a barycentric combination of the vertices in the triangular face underlying the pixel.
    
    Args:
        vertices (ndarray): An array of a certain per-vertex attribute, e.g., vertex coordinates, vertex colors, spherical harmonic bases, etc., (n, numVertices)
        pixelFaces (ndarray): The triangular face IDs for each pixel where the 3DMM is drawn, (numPixels,)
        pixelBarycentricCoords (ndarray): The barycentric coordinates of the vertices on the triangular face underlying each pixel where the 3DMM is drawn, (numPixels, 3)
        indexData (ndarray): An array containing the vertex indices for each face, (numFaces, 3)
    
    Returns:
        ndarray: The per-pixel barycentric reconstruction of the desired per-vertex attribute
    """
    pixelVertices = indexData[pixelFaces, :]
    
    if len(vertices.shape) is 1:
        vertices = vertices[np.newaxis, :]
    
    numChannels = vertices.shape[0]
        
    colorMat = vertices[:, pixelVertices.flat].reshape((numChannels, 3, pixelFaces.size), order = 'F')
    
    return np.einsum('ij,kji->ik', pixelBarycentricCoords, colorMat)



def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, c = 3, mapping_type = 'nearest'):
    ''' render mesh with texture map
    Args:
        vertices: [nver], 3
        triangles: [ntri, 3]
        texture: [tex_h, tex_w, 3]
        tex_coords: [ntexcoords, 3]
        tex_triangles: [ntri, 3]
        h: height of rendering
        w: width of rendering
        c: channel
        mapping_type: 'bilinear' or 'nearest'
    '''
    assert triangles.shape[0] == tex_triangles.shape[0]
    tex_h, tex_w, _ = texture.shape

    # initial 
    image = np.zeros((h, w, c))
    depth_buffer = np.zeros([h, w]) - 999999.

    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices
        tex_tri = tex_triangles[i, :] # 3 tex indice

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                if point_depth > depth_buffer[v, u]:
                    # update depth
                    depth_buffer[v, u] = point_depth    
                    
                    # tex coord
                    tex_xy = w0*tex_coords[tex_tri[0], :] + w1*tex_coords[tex_tri[1], :] + w2*tex_coords[tex_tri[2], :]
                    tex_xy[0] = max(min(tex_xy[0], float(tex_w - 1)), 0.0); 
                    tex_xy[1] = max(min(tex_xy[1], float(tex_h - 1)), 0.0); 

                    # nearest
                    if mapping_type == 'nearest':
                        tex_xy = np.round(tex_xy).astype(np.int32)
                        tex_value = texture[tex_xy[1], tex_xy[0], :] 

                    # bilinear
                    elif mapping_type == 'bilinear':
                        # next 4 pixels
                        ul = texture[int(np.floor(tex_xy[1])), int(np.floor(tex_xy[0])), :]
                        ur = texture[int(np.floor(tex_xy[1])), int(np.ceil(tex_xy[0])), :]
                        dl = texture[int(np.ceil(tex_xy[1])), int(np.floor(tex_xy[0])), :]
                        dr = texture[int(np.ceil(tex_xy[1])), int(np.ceil(tex_xy[0])), :]

                        yd = tex_xy[1] - np.floor(tex_xy[1])
                        xd = tex_xy[0] - np.floor(tex_xy[0])
                        tex_value = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd

                    image[v, u, :] = tex_value
    return image



def img2mesh_color(image_rgb, 
                   P_vertices_trans,S_colors0,S_triangles0,
                   point_isvisible, triangle_buffer0,
                   X_ind0=None,model_type='full',flag_show=True):
    import skimage
    from skimage.morphology import disk
    triangle_buffer=triangle_buffer0.copy()
    triangle_buffer[triangle_buffer>=0] =1.0
    triangle_buffer[triangle_buffer==-1]=0.0
    triangle_buffer =skimage.morphology.erosion(triangle_buffer, disk(7))
    #plt.imshow(np.abs(triangle_buffer1-triangle_buffer))
    
    h=image_rgb.shape[0]
    w=image_rgb.shape[1]
#    rendering,pixelCoord,pixelFaces,pixelBarycentricCoords=render_colors_ras(P_vertices_trans,
#                                                                             S_triangles0,
#                                                                             S_colors0,h,w)
    x,y=np.where(triangle_buffer>0)
    #imgReconstruction = barycentricReconstruction(S_colors0.T, pixelFaces, pixelBarycentricCoords, S_triangles0)
    image = np.zeros(image_rgb.shape)
    image[x, y, :] = image_rgb[x, y, :]
    
    if flag_show:
        import matplotlib.pyplot as plt
        x,y=np.where(triangle_buffer0>=0)
        image_dis = np.zeros(image_rgb.shape)
        image_dis[x, y, :] = image_rgb[x, y, :]
    
        plt.figure(figsize=(7,7))
        plt.imshow(image_dis)
        
        if (X_ind0 is not None)  :
            if  (len(X_ind0)==68):
                LEFT_EYE_POINTS = list(range(42, 48))
                RIGHT_EYE_POINTS = list(range(36, 42))
                LEFT_BROW_POINTS = list(range(22, 27))
                RIGHT_BROW_POINTS = list(range(17, 22))
                NOSE_POINTS = list(range(27, 36))
                MOUTH_POINTS = list(range(48, 67))
                JAW_PROFILE=list(range(0,17))
                face_groups=[JAW_PROFILE,
                            LEFT_EYE_POINTS,RIGHT_EYE_POINTS,
                            LEFT_BROW_POINTS,RIGHT_BROW_POINTS,
                            NOSE_POINTS,
                            MOUTH_POINTS]
                for group in face_groups:
                    plt.plot(P_vertices_trans[X_ind0[group],0],
                                P_vertices_trans[X_ind0[group],1],
                                linewidth=0.75,zorder=1,c='w')
        plt.scatter(P_vertices_trans[X_ind0,0],
                    P_vertices_trans[X_ind0,1],
                    c='r',s=15, linewidths=0.5, edgecolors="k",marker="o",zorder=2) 
        plt.show()
    #w=image_rgb.shape[1]
    
    
    S_colors1=S_colors0.copy()
    
    face_centeridx1=np.where(point_isvisible==1)[0]#range(len(P_vertices_trans))
        
    face_center=P_vertices_trans[face_centeridx1,:]          
    real_colors1=np.zeros((len(face_center),3))
    
    pixels_ii,pixels_jj=np.where(image[:,:,0]!=0)
    
    S_colors1=S_colors0.copy()
    S_colors1[face_centeridx1,:]=bilinear_interpolate(image,face_center[:,0],face_center[:,1])
    
    face_centeredge_idx=[]
    real_colors1=np.zeros((len(face_center),3))
    for i, x in enumerate(face_center):
        img_i= min(h-3,max(int(np.round(x[1])),0))
        img_j= min(w-3,max(int(np.round(x[0])),0))
        if np.sum(image[img_i:img_i+2,img_j:img_j+2,0]==0)>=1:#  & img_i<h-3  & img_j<w-3 & img_i>1  & img_j>1:
            #print(img_i,img_j)
            #print(img_i,img_j)
            # img_colors=image[img_i:img_i+2,img_j:img_j+2,:].reshape(4,3)
            # ii=np.array([img_i,img_i+1,img_i,img_i+1])
            # jj=np.array([img_j,img_j,img_j+1,img_j+1])
            # near_point=np.vstack((jj,ii)).T
            # dist=np.sqrt(np.sum((near_point-x[0:2].reshape(-1,2).repeat(4,axis=0))**2,1)).reshape(4,-1)
            # real_colors1[i,:]=np.sum(img_colors*(dist/dist.sum()).repeat(3,axis=1),0)
        # else:
            dist_img=np.sqrt((pixels_ii-img_i)**2+(pixels_jj-img_j)**2)
            dist_minidx=np.argsort(dist_img)[0:5].reshape(5,-1)
            dist=np.exp(-dist_img[dist_minidx])
            img_colors=np.array([image[x[0],x[1]] for x in np.hstack((pixels_ii[dist_minidx],pixels_jj[dist_minidx]))]).reshape(-1,3)
            real_colors1[i,:]=np.sum(img_colors*(dist/dist.sum()).repeat(3,axis=1),0)
            face_centeredge_idx.append(i)
    S_colors1[face_centeridx1,0:3]=real_colors1
    
    if flag_show:
        if X_ind0 is not None:
            plot_mlabvertex(P_vertices_trans,S_colors1,S_triangles0,P_vertices_trans[X_ind0,:])
        else:
            plot_mlabvertex(P_vertices_trans,S_colors1,S_triangles0) 
                
    #=========================================Rest Region=================================================   
    if model_type=='full':
        face_centeridx2=np.where(point_isvisible==0)[0]
        #face_centeridx2=np.where((fitted_vertices[:,0]<x_min) | (fitted_vertices[:,0]>x_max))[0]
        face_center=P_vertices_trans[face_centeridx2,:]
        #real_colors=np.array([ new_image[int(x[1]),int(x[0]),:] for x in face_center]).reshape(-1,3)
        real_colors2=np.zeros((len(face_center),3))
        
        pixels_ii,pixels_jj=np.where(image[:,:,0]!=0)
        #pixels_ij=np.vstack((pixels_ii,pixels_jj)).T
        for i, x in enumerate(face_center):
            img_i= np.min([int(np.round(x[1])),h-2])+np.random.randint(-3,3)
            #img_j= math.floor(x[0]*0.99) if nose_x<x[0] else  math.floor(x[0]*1.03)
            right_left= 1 if x[0]<P_vertices_trans[:,0].mean() else -1 
            img_j= int(np.round(x[0]))+right_left*np.random.randint(3,10)
            
            if np.sum(image[img_i:img_i+2,img_j:img_j+2,0]==0)<1:
                img_colors=image[img_i:img_i+2,img_j:img_j+2,:].reshape(4,3)
                ii=np.array([img_i,img_i+1,img_i,img_i+1])
                jj=np.array([img_j,img_j,img_j+1,img_j+1])
                near_point=np.vstack((jj,ii)).T
                dist=np.sqrt(np.sum((near_point-x[0:2].reshape(-1,2).repeat(4,axis=0))**2,1)).reshape(4,-1)
                real_colors2[i,:]=np.sum(img_colors*(dist/dist.sum()).repeat(3,axis=1),0)
            else:
                dist_img=np.sqrt((pixels_ii-img_i)**2+(pixels_jj-img_j)**2)
                dist_minidx=np.argsort(dist_img)[0:5].reshape(5,-1)
                dist=dist_img[dist_minidx]
                img_colors=np.array([image[x[0],x[1]] for x in np.hstack((pixels_ii[dist_minidx],pixels_jj[dist_minidx]))]).reshape(-1,3)
                real_colors2[i,:]=np.sum(img_colors*(dist/dist.sum()).repeat(3,axis=1),0)
                #print(dist)
        S_colors1[face_centeridx2,0:3]=real_colors2           
        
        if flag_show:
            if X_ind0 is not None:
                plot_mlabvertex(P_vertices_trans,S_colors1,S_triangles0,P_vertices_trans[X_ind0,:])
            else:
                plot_mlabvertex(P_vertices_trans,S_colors1,S_triangles0)
                
#    if model_type=='BFM2009':
#        faceidx=np.loadtxt('./model_data/face_region_idx.txt').astype(int)
#        
#        #==================================Eye-Nose-Month-Region================================================
#        face_centeridx1=np.where(faceidx[:,4]==0)[0]
#        face_center=P_vertices_trans[face_centeridx1,:]
#                    
#        #real_colors=np.array([ new_image[int(x[1]),int(x[0]),:] for x in face_center]).reshape(-1,3)
#        real_colors1=np.zeros((len(face_center),3))
#        for i, x in enumerate(face_center):
#            img_i= int(np.round(x[1]))
#            img_j= int(np.round(x[0]))
#            img_colors=image[img_i:img_i+2,img_j:img_j+2,:].reshape(4,3)
#            ii=np.array([img_i,img_i+1,img_i,img_i+1])
#            jj=np.array([img_j,img_j,img_j+1,img_j+1])
#            near_point=np.vstack((jj,ii)).T
#            dist=np.sqrt(np.sum((near_point-x[0:2].reshape(-1,2).repeat(4,axis=0))**2,1)).reshape(4,-1)
#            real_colors1[i,:]=np.sum(img_colors*(dist/dist.sum()).repeat(3,axis=1),0)
#            
#        S_colors1[face_centeridx1,0:3]=real_colors1
#        
#        
    return S_colors1


def calculate_point_isvisible(vertices0, triangles,h=512,w=512):
    ''' 
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        h: height
        w: width
    Returns:
        depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle). 
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # h, w is the size of rendering
    '''
    # initial 
    #scale=526    
    vertices=vertices0.copy()
    
   
    h_min=vertices[:,1].min()
    h_max=vertices[:,1].max()
    w_min=vertices[:,0].min()
    w_max=vertices[:,0].max()
    
    vertices[:,1]=(vertices[:,1]-h_min)/(h_max-h_min)*h
    vertices[:,0]=(vertices[:,0]-w_min)/(w_max-w_min)*w
   
    #h=scale#vertices[:,0:2].max()
    #w=scale#vertices[:,0:2].max()
    num=vertices.shape[0]
    point_isvisible = np.zeros(num)
    depth_buffer = np.zeros([h, w]) + 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
    triangle_buffer0 = np.zeros([h, w], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    triangle_buffer1 = np.zeros([h, w], dtype = np.int32) - 1    
    #barycentric_weight = np.zeros([h, w, 3], dtype = np.float32)  # 
    
    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2]) # barycentric weight
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                if point_depth < depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    triangle_buffer0[v, u] = i
                    #point_isvisible[tri]   = 1
                    #triangle_buffer1[int(v/h*(h_max-h_min)+h_min),int(u/w*(w_max-w_min)+w_min)]=i
                    #barycentric_weight[v, u, :] = np.array([w0, w1, w2])
    triangle_idx=triangle_buffer0[triangle_buffer0>=0]
    point_isvisible[triangles[triangle_idx,:].flatten()]=1 
          
    return point_isvisible,triangle_buffer0


def Compose_Textures(textures,masks,brightnessRemoval=0.7):
    # Normalize masks. mask_i = mask_i / sum_masks
	# adding small amount avoids divisons by zero
    assert(len(textures) == len(masks))
    
    Num_textures=len(masks)
    Num_point=len(textures[0])
    
    sum_masks = sum(masks) + 0.0000001
    for i in range(Num_textures):
        masks[i] = masks[i]/ sum_masks
        
    # Calculate mean texture taking into consideretion the normalized masks.
		# mean_text =  mask_1 * text_1 +...+ mask_i * text_i 
    mean_texture = np.zeros_like(textures[0])
    
    for i in range(Num_textures):
        mask = masks[i]
        # Repeat mask to give 3 channels
        # mask = np.repeat(mask[:,:, np.newaxis], 3, axis=2) # less complicated method below
        mask = np.c_[mask,mask,mask]
        mean_texture += mask * textures[i]
    
    
    # We want to reduce shadows and specular highlights. To do this we'll iterate through every texture pixel and compute its brightness and compare to the brightness of the average texture at that pixel. The difference D = |(current texture brightness) - (average texture brightness)| will be in the range [0, 254]. 
	# D/254 will normalize it to [0,1], being 1 total difference. Given that the mask value of that texture at that pixel is W, we'll have: 
	# new W = (W -W*D*brightnessRemoval) = W*(1 - D*brightnessRemoval)
	# brightnessRemoval might be greater than 1, so the new W might become negative. In this case we set its value to 0
	# We'll skip the pixel if W is already 0.
    
    for k in range(Num_point):
        R, G, B = mean_texture[k, :]
        avg_brightness = 0.5 * max(R, G, B) + 0.5*min(R, G, B)
        for i in range(Num_textures):
            W = masks[i][k]
            if W == 0:
                continue
            R, G, B = textures[i][k, :]
            tex_brightness = 0.5 * max(R, G, B) + 0.5*min(R, G, B)
            D = abs(avg_brightness- tex_brightness)/254
            newW = W*(1-D*brightnessRemoval)
            masks[i][k] = max(0, newW)
    
    # Normalize masks again
    sum_masks = sum(masks) + 0.0000001
    for i in range(Num_textures):
        masks[i] = masks[i]/ sum_masks
			
	# Compose final texture:
    final_texture = np.zeros_like(textures[0])
    for i in range(Num_textures):
        mask = masks[i]
        text = textures[i]
        weighted_text = np.c_[mask,mask,mask] * text
        final_texture += weighted_text
    #plot_mlabvertex(fitted_vertices2,final_texture,M_triangles,fitted_vertices1[X_ind,:])
    return final_texture



def get_colors(img, ver,method='nearest'):
    if method=='nearest':
        # nearest-neighbor sampling
        [h, w, _] = img.shape
        ver[0, :] = np.minimum(np.maximum(ver[0, :], 0), w - 1)  # x
        ver[1, :] = np.minimum(np.maximum(ver[1, :], 0), h - 1)  # y
        ind = np.round(ver).astype(np.int32)
        colors = img[ind[1, :], ind[0, :], :]  # n x 3
    elif  method=='interpolate':
        colors=bilinear_interpolate(img, ver[0, :], ver[1, :])
    return colors


def bilinear_interpolate(img, x, y):
    """
    https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    """
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[..., np.newaxis] * i_a + wb[..., np.newaxis] * i_b + wc[..., np.newaxis] * i_c + wd[..., np.newaxis] * i_d


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1), dtype=np.float32)))  # add z
    return uv_coords

