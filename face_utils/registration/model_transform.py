# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:58:53 2025

@author: Robbie
"""

import numpy as np
import pickle
import open3d as o3d #http://www.open3d.org/docs/release/tutorial/Basic/mesh.html 

def BFM20092Models(vertices,colors=None,model_type=None):
    # if model_type=="Facescape":
    #     vert_weight=np.load('./model_data/FaceScape/BFM2FaceScape_vert_weight.npy')
    #     vert_distidx=np.load('./model_data/FaceScape/BFM2FaceScape_vert_distidx.npy')
        
    #     facescape_triangles=np.load("./model_data/FaceScape/BFM2FaceScape_face_triangles.npy")
    #     face_idx=np.load('./model_data/FaceScape/BFM2FaceScape_face_idx.npy')
    if model_type=="SymHead":
        vert_weight=np.load('./model_data/BFM_2009_Model/BFM20092SymHead_vert_weight.npy')
        vert_distidx=np.load('./model_data/BFM_2009_Model/BFM20092SymHead_vert_distidx.npy')
        
        facescape_triangles=np.load("./model_data/BFM_2009_Model//BFM20092SymHead_face_triangles.npy")
        face_idx=np.load('./model_data/SymHead_Chinese_Model/SymHead_front_faceidx.npy')
        
    colorMat = vertices.T[:, vert_distidx.flat].reshape((vertices.shape[1], 3, len(vert_distidx)), order = 'F')
    
    facescape_vertices=np.einsum('ij,kji->ik', vert_weight, colorMat)
    
    if colors is not None:
        colors=colors.astype(float)
        colorMat = colors.T[:, vert_distidx.flat].reshape((vertices.shape[1], 3, len(vert_distidx)), order = 'F')
    
        facecape_colors=np.einsum('ij,kji->ik', vert_weight, colorMat)
    else:
        facecape_colors=np.repeat(np.array([100.,100.,100.]),len(facescape_vertices),0).reshape(-1,3)
        
    return facescape_vertices,facecape_colors,facescape_triangles,face_idx
    
def Models2BFM2009(vertices,colors=None,model_type=None):
    if model_type=="LYHM_FLAME":
        vert_weight=np.load('./model_data/FLAME2020/LYHM_FLAME2BFM_vert_weight.npy')
        vert_distidx=np.load('./model_data/FLAME2020/LYHM_FLAME2BFM_vert_distidx.npy')
    elif model_type=="Facescape":
        vert_weight=np.load('./model_data/FaceScape/Facescape2BFM_vert_weight.npy')
        vert_distidx=np.load('./model_data/FaceScape/Facescape2BFM_vert_distidx.npy')
    elif model_type=="BFM2019":
        vert_weight=np.load('./model_data/BFM_2019_Model/BFM20192BFM_vert_weight.npy')
        vert_distidx=np.load('./model_data/BFM_2019_Model/BFM20192BFM_vert_distidx.npy')
    elif model_type=="FLAME2020":
        vert_weight=np.load('./model_data/FLAME2020/FLAME20202BFM_vert_weight.npy')
        vert_distidx=np.load('./model_data/FLAME2020/FLAME20202BFM_vert_distidx.npy')
    elif model_type=="LYHM":
        vert_weight=np.load('./model_data/LYHM_Model/LYHM2BFM_vert_weight.npy')
        vert_distidx=np.load('./model_data/LYHM_Model/LYHM2BFM_vert_distidx.npy')
    elif model_type=="UHM":
        vert_weight=np.load('./model_data/UHM_models/UHM2BFM_vert_weight.npy')
        vert_distidx=np.load('./model_data/UHM_models/UHM2BFM_vert_distidx.npy')
    elif model_type=="ICT":
        vert_weight=np.load('./model_data/ICT_Model/ICT2BFM_vert_weight.npy')
        vert_distidx=np.load('./model_data/ICT_Model/ICT2BFM_vert_distidx.npy')
    elif model_type=="SymHead":     
        vert_weight=np.load('./model_data/SymHead_Chinese_Model/SymHead2BFM2009_vert_weight.npy')
        vert_distidx=np.load('./model_data/SymHead_Chinese_Model/SymHead2BFM2009_vert_distidx.npy')
    elif model_type=="BFM2017":   
        vert_weight=np.load('./model_data/BFM_2017_Model/BFM20172BFM2009_vert_weight.npy')
        vert_distidx=np.load('./model_data/BFM_2017_Model/BFM20172BFM2009_vert_distidx.npy')

    else:
         raise ValueError('There is no such model type')
    
    vert_weight=np.abs(vert_weight)/np.abs(vert_weight).sum(1).reshape(-1,1)
    
    colorMat = vertices.T[:, vert_distidx.flat].reshape((vertices.shape[1], 3, len(vert_distidx)), order = 'F')
    
    BFM_vertices=np.einsum('ij,kji->ik', vert_weight, colorMat)
    
    nose_outlier_idx=np.array([7118, 7119, 7120, 7238, 7239, 7240, 7358, 7359, 7360, 7478, 7479,
                               7480, 7598, 7599, 7600, 8798, 8799, 8800, 8918, 8919, 8920, 9038,
                               9039, 9040, 9158, 9159, 9160, 9278, 9279, 9280, 9398, 9399, 9400],
                               dtype=np.int32)
    #BFM_vertices[nose_outlier_idx,2]=BFM_vertices[nose_outlier_idx,2]-3
    BFM_vertices[nose_outlier_idx,1]=BFM_vertices[nose_outlier_idx,1]-2
    bfm_triangles=np.load("./model_data/BFM_2009_Model/uv-data/BFM_triangles.npy")
    
    if colors is not None:
        colors=colors.astype(float)
        colorMat = colors.T[:, vert_distidx.flat].reshape((vertices.shape[1], 3, len(vert_distidx)), order = 'F')
    
        BFM_colors=np.einsum('ij,kji->ik', vert_weight, colorMat)
    else:
        BFM_colors=np.repeat(np.array([150.,150.,150.]),len(BFM_vertices),0).reshape(-1,3)
        
    return BFM_vertices,BFM_colors,bfm_triangles


def Models2SymHead(vertices,colors=None,model_type=None):
     if model_type=="BFM2019":
        vert_weight=np.load('./model_data/BFM_2019_Model/BFM20192SymHead_vert_weight.npy')
        vert_distidx=np.load('./model_data/BFM_2019_Model/BFM20192SymHead_vert_distidx.npy')
     elif model_type=="Facescape":
        vert_weight=np.load('./model_data/FaceScape/Facescape2SymHead_vert_weight.npy')
        vert_distidx=np.load('./model_data/FaceScape/Facescape2SymHead_vert_distidx.npy')
     elif model_type=="FLAME2020":
        vert_weight=np.load('./model_data/FLAME2020/FLAME20202SymHead_vert_weight.npy')
        vert_distidx=np.load('./model_data/FLAME2020/FLAME20202SymHead_vert_distidx.npy')
    
     vert_weight=np.abs(vert_weight)/np.abs(vert_weight).sum(1).reshape(-1,1)
    
     colorMat = vertices.T[:, vert_distidx.flat].reshape((vertices.shape[1], 3, len(vert_distidx)), order = 'F')
     BFM_vertices=np.einsum('ij,kji->ik', vert_weight, colorMat)
    
     bfm_triangles=np.load("./model_data/BFM_2019_Model/SymHead_triangles.npy")
     if colors is not None:
        colors=colors.astype(float)
        colorMat = colors.T[:, vert_distidx.flat].reshape((vertices.shape[1], 3, len(vert_distidx)), order = 'F')
    
        BFM_colors=np.einsum('ij,kji->ik', vert_weight, colorMat)
     else:
        BFM_colors=np.repeat(np.array([150.,150.,150.]),len(BFM_vertices),0).reshape(-1,3)
        
     if model_type=="FLAME2020":
         
        mesh=o3d.pybind.geometry.TriangleMesh()
        mesh.vertices =o3d.pybind.utility.Vector3dVector(BFM_vertices)
        mesh.triangles=o3d.pybind.utility.Vector3iVector(bfm_triangles)
        mesh=mesh.filter_smooth_taubin(20,0.75)#filter_smooth_laplacian(10, 0.5)
        BFM_vertices=np.asarray(mesh.vertices)
     return BFM_vertices,BFM_colors,bfm_triangles


def FaceScape_AddEye(vertices,triangles,colors=None):
    Eye_triangles=np.load('./model_data/FaceScape/Eye_triangles.npy')
    Eye_idx=np.load('./model_data/FaceScape/Eye_verticesIdx.npy')
    
    BFM_vertices,BFM_colors,bfm_triangles=Models2BFM2009(vertices,None,model_type="Facescape")
    
    vertices_witheye=np.r_[vertices,BFM_vertices[Eye_idx,:]]
    triangles_witheye=np.r_[triangles,Eye_triangles]
    colors_witheye=np.repeat(np.array([100.,100.,100.]),len(vertices_witheye),0).reshape(-1,3)
    if colors is not None:
        colors_witheye[0:len(colors),:]=colors
        
    return vertices_witheye,colors_witheye,triangles_witheye


class load_ModelFacescape():
    def __init__(self, model_dir):
        self.num_lm = 68

        self.lm_index = np.zeros(self.num_lm, dtype=int)
        with open(f'{model_dir}/index_68.txt', 'r') as f:
            lines = f.readlines()
            for i in range(self.num_lm):
                line = lines[i]
                values = line.split()
                self.lm_index[i] = int(values[0])

        with open(f'{model_dir}/faces.pkl', 'rb') as f:
            self.texcoords, self.faces = pickle.load(f)
        with open(f'{model_dir}/front_verts_indices.pkl', 'rb') as f:
            self.front_verts_indices = pickle.load(f)
        with open(f'{model_dir}/front_texcoords.pkl', 'rb') as f:
            self.front_texcoords = pickle.load(f)
        with open(f'{model_dir}/front_faces.pkl', 'rb') as f:
            self.front_faces = pickle.load(f)

        with open(f'{model_dir}/id_mean.pkl', 'rb') as f:
            self.id_mean = pickle.load(f)
        with open(f'{model_dir}/id_var.pkl', 'rb') as f:
            self.id_var = pickle.load(f)
        #with open(f'{model_dir}/exp_GMM.pkl', 'rb') as f:
        #    self.exp_gmm = pickle.load(f)
        with open(f'{model_dir}/front_faces.pkl', 'rb') as f:
            self.front_faces = pickle.load(f)

        with open(f'{model_dir}/contour_line_68.pkl', 'rb') as f:
            self.contour_line_right, self.contour_line_left = pickle.load(f)
        self.core_tensor = np.load(f'{model_dir}/core_847_300_52.npy')
        self.factors_id = np.load(f'{model_dir}/factors_id_847_300_52.npy')

        self.core_tensor = self.core_tensor.transpose((2, 1, 0))
        for i in range(51):
            self.core_tensor[:, i + 1, :] = self.core_tensor[:, i + 1, :] - self.core_tensor[:, 0, :]

        with open(f'{model_dir}/front_face_indices.pkl', 'rb') as f:
            self.front_face_indices = pickle.load(f)
        
        self.matrix_tex = np.load(f'{model_dir}/matrix_text_847_100.npy')
        self.mean_tex = np.load(f'{model_dir}/mean_text_847_100.npy')
        self.factors_tex = np.load(f'{model_dir}/factors_tex_847_100.npy')

        #self.detector = dlib.get_frontal_face_detector()
        #self.points = dlib.shape_predictor('./landmark_predictor/shape_predictor_68_face_landmarks.dat')

        # for render
        tris = []
        self.vert_texcoords = np.zeros((len(self.front_verts_indices), 2))
        for face in self.front_faces:
            vertices, normals, texture_coords, material = face
            tris.append([vertices[0] - 1, vertices[1] - 1, vertices[2] - 1])
            for i in range(len(vertices)):
                self.vert_texcoords[vertices[i] - 1] = self.front_texcoords[texture_coords[i] - 1]
        self.front_tris= np.array(tris)
        self.triangles=np.load(f'{model_dir}//facescape.npy')    