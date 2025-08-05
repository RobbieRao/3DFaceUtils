'''
Functions about lighting mesh(changing colors/texture of mesh).
1. add light to colors/texture (shade each vertex)
2. fit light according to colors/texture & image.

Preparation knowledge:
lighting: https://cs184.eecs.berkeley.edu/lecture/pipeline
spherical harmonics in human face: '3D Face Reconstruction from a Single Image Using a Single Reference Face Shape'
'''

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import numpy as np

_norm = lambda arr: arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]

def get_normal(vertices, triangles):
    ''' calculate normal direction in each vertex
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3]
    '''
    try:
        from pyvisita import TriMesh
        mesh = TriMesh(vertices.copy(), triangles.copy())
        normal=mesh.vertex_normals()
        
    except:
        pt0 = vertices[triangles[:, 0], :] # [ntri, 3]
        pt1 = vertices[triangles[:, 1], :] # [ntri, 3]
        pt2 = vertices[triangles[:, 2], :] # [ntri, 3]
        #tri_normal = np.cross(pt0 - pt1, pt0 - pt2) # [ntri, 3]. normal of each triangle
        tri_normal = np.cross(pt1 - pt0, pt2 - pt0) # [ntri, 3]. normal of each triangle
        d = np.sqrt( np.sum(tri_normal *tri_normal, 1))
        zero_ind = (d == 0)
        d[zero_ind] = 1
        tri_normal=tri_normal/d.reshape(-1,1)
        
        normal = np.zeros_like(vertices) # [nver, 3]
        for i in range(triangles.shape[0]):
            normal[triangles[i, 0], :] = normal[triangles[i, 0], :] + tri_normal[i, :]
            normal[triangles[i, 1], :] = normal[triangles[i, 1], :] + tri_normal[i, :]
            normal[triangles[i, 2], :] = normal[triangles[i, 2], :] + tri_normal[i, :]
        
        # normalize to unit length
        mag = np.sum(normal**2, 1) # [nver]
        zero_ind = (mag == 0)
        mag[zero_ind] = 1;
        normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))
    
        normal = normal/np.sqrt(mag[:,np.newaxis])

    return normal


def sh9(x, y, z):
    """
    First nine spherical harmonics as functions of Cartesian coordinates
    """
    h = np.empty((9, x.size))
    h[0, :] = 1/np.sqrt(4*np.pi) * np.ones(x.size)
    h[1, :] = np.sqrt(3/(4*np.pi)) * z
    h[2, :] = np.sqrt(3/(4*np.pi)) * x
    h[3, :] = np.sqrt(3/(4*np.pi)) * y
    h[4, :] = 1/2*np.sqrt(5/(4*np.pi)) * (3*np.square(z) - 1)
    h[5, :] = 3*np.sqrt(5/(12*np.pi)) * x * z
    h[6 ,:] = 3*np.sqrt(5/(12*np.pi)) * y * z
    h[7, :] = 3/2*np.sqrt(5/(12*np.pi)) * (np.square(x) - np.square(y))
    h[8, :] = 3*np.sqrt(5/(12*np.pi)) * x * y
    
    return h * np.r_[np.pi, np.repeat(2*np.pi/3, 3), np.repeat(np.pi/4, 5)][:, np.newaxis]


def add_light_sh(vertices, triangles, colors, sh_coeff):
    ''' 
    spherical harmonics lighting model
    In 3d face, usually assume:
    1. The surface of face is Lambertian(reflect only the low frequencies of lighting)
    2. Lighting can be an arbitrary combination of point sources
    --> can be expressed in terms of spherical harmonics(omit the lighting coefficients)
    I = albedo * (sh(n) x sh_coeff)

    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        colors: [nver, 3] albedo
        sh_coeff: [9, 3] spherical harmonics coefficients

    Returns:
        lit_colors: [nver, 3]
    '''
    assert vertices.shape[0] == colors.shape[0]
    
    vertexNorms = get_normal(vertices, triangles) # [nver, 3]
    
    sh = sh9(vertexNorms[:, 0], vertexNorms[:, 1], vertexNorms[:, 2])
    
    lit_colors = np.empty_like(colors)
    for c in range(3):
        lit_colors[:, c] = np.dot(sh_coeff[:, c], sh) * colors[:,c]
        
    return lit_colors

def convert_type(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj

def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices

# Reference: https://github.com/cleardusk/3DDFA/blob/master/utils/lighting.py
def add_light_BP(vertices, triangles, colors, **kwargs):
    #BlinnPhong光照模型
    intensity_ambient = convert_type(kwargs.get('intensity_ambient', 0.3))
    intensity_directional = convert_type(kwargs.get('intensity_directional', 0.6))
    intensity_specular = convert_type(kwargs.get('intensity_specular', 0.9))
    specular_exp = kwargs.get('specular_exp', 5)
    color_ambient = convert_type(kwargs.get('color_ambient', (1, 1, 1)))
    color_directional = convert_type(kwargs.get('color_directional', (1, 1, 1)))
    light_pos = convert_type(kwargs.get('light_pos', (0, 0, 1)))
    view_pos = convert_type(kwargs.get('view_pos', (0, 0, 1)))
    
    assert vertices.shape[0] == colors.shape[0]
    normal = get_normal(vertices, triangles) # [nver, 3]
    
    lit_colors=colors.copy()
    # ambient component 环境光
    #if intensity_ambient > 0:
    lit_colors += intensity_ambient * color_ambient
        
    vertices_n = norm_vertices(vertices.copy())  
    
    if intensity_directional > 0:
    # diffuse component 漫反射
        direction = _norm(light_pos - vertices_n)
        cos = np.sum(normal * direction, axis=1)[:, None]
        # cos = np.clip(cos, 0, 1)
        #  todo: check below
        lit_colors += intensity_directional * (color_directional * np.clip(cos, 0, 1))

    # specular component 镜面反射
    if intensity_specular > 0:
        v2v = _norm(view_pos - vertices_n)
        reflection = 2 * cos * normal - direction
        spe = np.sum((v2v * reflection) ** specular_exp, axis=1)[:, None]
        spe = np.where(cos != 0, np.clip(spe, 0, 1), np.zeros_like(spe))
        lit_colors += intensity_specular * color_directional * np.clip(spe, 0, 1)
        
        
    lit_colors = np.clip(lit_colors, 0, 1)
        
    return lit_colors

def calculate_diffuse_reflectance(vertices, triangles, light_positions = 0, light_intensities = 0):
    ''' Gouraud shading. add point lights.
    In 3d face, usually assume:
    1. The surface of face is Lambertian(reflect only the low frequencies of lighting)
    2. Lighting can be an arbitrary combination of point sources
    3. No specular (unless skin is oil, 23333)

    Ref: https://cs184.eecs.berkeley.edu/lecture/pipeline    
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        light_positions: [nlight, 3] 
        light_intensities: [nlight, 3]
    Returns:
        lit_colors: [nver, 3]
    '''
    
    normals = get_normal(vertices, triangles) # [nver, 3]

    # ambient
    # La = ka*Ia

    # diffuse
    # Ld = kd*(I/r^2)max(0, nxl)
    direction_to_lights = vertices[np.newaxis, :, :] - light_positions[:, np.newaxis, :] # [nlight, nver, 3]
    direction_to_lights_n = np.sqrt(np.sum(direction_to_lights**2, axis = 2)) # [nlight, nver]
    direction_to_lights = direction_to_lights/direction_to_lights_n[:, :, np.newaxis]
    normals_dot_lights = normals[np.newaxis, :, :]*direction_to_lights # [nlight, nver, 3]
    normals_dot_lights = np.sum(normals_dot_lights, axis = 2) # [nlight, nver]
    #diffuse_output = normals_dot_lights[:, :, np.newaxis]*light_intensities[:, np.newaxis, :]
    #diffuse_output = np.sum(diffuse_output, axis = 0) # [nver, 3]
    
    diffuse_reflectance= np.maximum(normals_dot_lights[:, :, np.newaxis],0)*light_intensities[:, np.newaxis, :]
    # specular
    # h = (v + l)/(|v + l|) bisector
    # Ls = ks*(I/r^2)max(0, nxh)^p
    # increasing p narrows the reflectionlob

    #lit_colors = diffuse_output # only diffuse part here.
    #lit_colors = np.minimum(np.maximum(lit_colors, 0), 1)
    return diffuse_reflectance

