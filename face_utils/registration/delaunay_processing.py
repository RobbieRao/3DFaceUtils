# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:28:57 2025

@author: Robbie
"""
import numpy as np
from scipy.spatial import Delaunay

dist_tri=lambda x, y, z: np.max((np.sqrt(sum(np.power((x - y), 2))),
                                  np.sqrt(sum(np.power((x - z), 2))),
                                  np.sqrt(sum(np.power((y - z), 2)))))
    

def mapuv_Delaunay(uvmap,vertices=None,Theshold_dtriangles=None):
    uvmap_Delaunay= Delaunay(uvmap[:,0:2])
    triangles_uv=uvmap_Delaunay.simplices
    
    if (vertices is not None) & (Theshold_dtriangles is not None) :
        #Theshold_dtriangles=1
        dist_triangles=[dist_tri(vertices[x[0],:],vertices[x[1],:],vertices[x[2],:] )
                        for x in triangles_uv]
        triangles_uv=triangles_uv[np.array(dist_triangles)<=Theshold_dtriangles,:]
    return triangles_uv