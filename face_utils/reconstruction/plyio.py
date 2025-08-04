# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:49:29 2020

@author: Peter_Zhang
"""
#from plyfile import PlyData, PlyElement
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
#import json
import logging
#from pprint import pprint
import os


def write_obj(obj_name, vertices,colors, triangles):
    #Ref:https://github.com/YadiraF/PRNet/blob/fc12fe5e1f1462bdea52409b213d0cf1c8cf6c5b/utils/write.py
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
 
            if colors is not None:
                s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            else:
                 s = 'v {} {} {} \n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)
        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            #s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)

def read_obj(fname):
    
    assert os.path.exists(fname), 'file %s not found' % fname
    
    tri_indices = []
    vertices=[]
    tex_coords=[]
    colors=[]
    with open(fname, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines if line.strip()]

        for line in lines:
            parts = line.split(' ')
            words = [part.strip() for part in parts if part]

            # Parse command and data from each line
            words = line.split()
            command = words[0]
            data = words[1:]
    
            if command == 'mtllib': # Material library
                model_path = os.path.split(fname)[0]
                mtllib_path = os.path.join( model_path, data[0] )                                          
            elif command == 'v': # Vertex
                if len(data)==3:
                    x, y, z = data
                    vertex = [float(x), float(y), float(z)]
                    vertices.append(vertex)
                elif len(data)==6:
                    x,y,z,r,g,b=data
                    vertex = [float(x), float(y), float(z)]
                    vertices.append(vertex)
                    color=[float(r), float(g), float(b)]
                    colors.append(color)
            elif command == 'vt': # Texture coordinate
                s, t = data
                tex_coord = [float(s), float(t)]
                tex_coords.append(tex_coord)
            elif command == 'f':
                if len(data) !=  3:
                    return np.array(vertices)
                assert len(data) ==  3, "Sorry, only triangles are supported"
                vi, ti, ni = data
                try:
                    indices = [int(ni) - 1, int(ti) - 1, int(vi) - 1]
                except:
                    return np.array(vertices)    
                tri_indices.append(indices)
        
        vertices=np.array(vertices)
        tri_indices=np.array(tri_indices)
        if (len(colors)==0) &  (len(tex_coords)==0):
            return vertices,tri_indices
        elif (len(colors)!=0) &  (len(tex_coords)==0):
            colors=np.array(colors)
            return vertices, colors,tri_indices     
        elif (len(colors)==0) &  (len(tex_coords)!=0):
            tex_coords=np.array(tex_coords)
            return vertices, tex_coords,tri_indices  
        


sys_byteorder = (">", "<")[sys.byteorder == "little"]

ply_dtypes = dict(
    [
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "b1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)

valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}


def read_ply(filename):
    """Read a .ply (binary or ascii) file and store the elements in pandas DataFrame
    Parameters
    ----------
    filename: str
        Path to the filename
    Returns
    -------
    data: dict
        Elements as pandas DataFrames; comments and ob_info as list of string
    """

    with open(filename, "rb") as ply:

        if b"ply" not in ply.readline():
            raise ValueError("The file does not start whith the word ply")
        # get binary_little/big or ascii
        fmt = ply.readline().split()[1].decode()
        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        line = []
        dtypes = defaultdict(list)
        count = 2
        points_size = None
        mesh_size = None
        while b"end_header" not in line and line != b"":
            line = ply.readline()

            if b"element" in line:
                line = line.split()
                name = line[1].decode()
                size = int(line[2])
                if name == "vertex":
                    points_size = size
                elif name == "face":
                    mesh_size = size

            elif b"property" in line:
                line = line.split()
                # element mesh
                if b"list" in line:
                    mesh_names = ["n_points", "v1", "v2", "v3"]

                    if fmt == "ascii":
                        # the first number has different dtype than the list
                        dtypes[name].append((mesh_names[0], ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ply_dtypes[line[3]]
                    else:
                        # the first number has different dtype than the list
                        dtypes[name].append((mesh_names[0], ext + ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ext + ply_dtypes[line[3]]

                    for j in range(1, 4):
                        dtypes[name].append((mesh_names[j], dt))
                else:
                    if fmt == "ascii":
                        dtypes[name].append((line[2].decode(), ply_dtypes[line[1]]))
                    else:
                        dtypes[name].append(
                            (line[2].decode(), ext + ply_dtypes[line[1]])
                        )
            count += 1

        # for bin
        end_header = ply.tell()

    data = {}

    if fmt == "ascii":
        top = count
        bottom = 0 if mesh_size is None else mesh_size

        names = [x[0] for x in dtypes["vertex"]]
        #print(names)
        data["points"] = pd.read_csv(
            filename,
            sep=" ",
            header=None,
            engine="python",
            skiprows=top,
            skipfooter=bottom,
            usecols=names,
            names=names,
        )

        for n, col in enumerate(data["points"].columns):
            data["points"][col] = data["points"][col].astype(dtypes["vertex"][n][1])
            
            #print(data["points"])
        #print(mesh_size)
        if mesh_size is not None:
            top = count + points_size

            names = [x[0] for x in dtypes["face"]]
            #usecols = [1, 2, 3]

            data["mesh"] = pd.read_csv(
                filename,
                sep=" ",
                header=None,
                engine="python",
                skiprows=top,
                usecols=names,
                names=names,
            )
            #print(name,data["mesh"] )
            for n, col in enumerate(data["mesh"].columns):
                data["mesh"][col] = data["mesh"][col].astype(dtypes["face"][n][1])

    else:
        with open(filename, "rb") as ply:
            ply.seek(end_header)
            points_np = np.fromfile(ply, dtype=dtypes["vertex"], count=points_size)
            if ext != sys_byteorder:
                points_np = points_np.byteswap().newbyteorder()
            data["points"] = pd.DataFrame(points_np)
            if mesh_size is not None:
                mesh_np = np.fromfile(ply, dtype=dtypes["face"], count=mesh_size)
                if ext != sys_byteorder:
                    mesh_np = mesh_np.byteswap().newbyteorder()
                data["mesh"] = pd.DataFrame(mesh_np)
                data["mesh"].drop("n_points", axis=1, inplace=True)
                
    vertices=data["points"].values[:,0:3]
    #print(data["points"])
    if data["mesh"].shape[1]>3:
        triangles=data["mesh"].values[:,1:4]
    else:
        triangles=data["mesh"].values[:,0:3]
    
    if data["points"].values.shape[1]>=6:
        colors=data["points"].values[:,3:6]            
        if colors.max()>1.: colors=colors/255 
    else:
        colors=np.repeat(np.array([30,144,195.]).reshape(1,3),len(vertices),axis=0)/255 #blue
    return vertices,colors,triangles



def describe_element(name, df):
    """Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {"f": "float", "u": "uchar", "i": "int"}
    element = ["element " + name + " " + str(len(df))]

    if name == "face":
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append("property " + f + " " + str(df.columns.values[i]))

    return element


def write_ply(filename, points=None, colors=None, mesh=None, as_text=True):
    
    if colors.max()<=1.: colors=colors*255
    
    points = pd.DataFrame(points, columns=["x", "y", "z"])
    mesh = pd.DataFrame(mesh, columns=["v1", "v2", "v3"])
    if colors is not None:
        colors = pd.DataFrame(colors, columns=["red", "green", "blue"])
        points = pd.concat([points, colors], axis=1)
    """
    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
    Returns
    -------
    boolean
        True if no problems
    """
    if not filename.endswith("ply"):
        filename += ".ply"

    # open in text mode to write the header
    with open(filename, "w") as ply:
        header = ["ply"]

        if as_text:
            header.append("format ascii 1.0")
        else:
            header.append("format binary_" + sys.byteorder + "_endian 1.0")

        if points is not None:
            header.extend(describe_element("vertex", points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element("face", mesh))

        header.append("end_header")

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(
                filename, sep=" ", index=False, header=False, mode="a", encoding="ascii"
            )
        if mesh is not None:
            mesh.to_csv(
                filename, sep=" ", index=False, header=False, mode="a", encoding="ascii"
            )

    else:
        # open in binary/append to use tofile
        with open(filename, "ab") as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True     
   
# def write_ply(vertices,colors,triangles,filename):
    
#      colors=np.c_[colors.astype(np.uint8),255*np.ones((len(colors),1))]
#      vertex = np.array([tuple(x)  for x in np.hstack((vertices,colors[:,0:4]))], 
#                   dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
#                          ("red", "u1"), ("green", "u1"), ("blue", "u1"),("alpha", "u1")])
#      #vertex = np.array([tuple(x)  for x in vertices], 
#      #             dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
#      face_colors=colors[triangles[:,0],0:3]
#      face =np.array([(x, r, g, b) for x,r,g,b in zip(triangles,
#                                                      face_colors[:,0],
#                                                      face_colors[:,1],
#                                                      face_colors[:,2])],
#                      dtype=[("vertex_indices", "i4", (3,)), 
#                             ("red", "u1"), 
#                             ("green", "u1"), 
#                             ("blue", "u1")])
         
#      #face =np.array([tuple(x)  for x,r,g,b in triangles],
#      #                dtype=("vertex_indices", "i4", (3)))
          
#      el =  PlyElement.describe(vertex, "vertex")
#      el2 = PlyElement.describe(face, "face")
#      #el3 = PlyElement.describe(edge, "edge")
#      plydata = PlyData([el, el2],text=True)
#      plydata.write(filename)
     
# def read_ply(filename):
#     plydata = PlyData.read(filename)

#     tri_data = plydata['face'].data['vertex_indices']
#     triangles = np.vstack(tri_data)  # MX3
#     vertex_color = plydata['vertex'] 
#     #print(vertex_color[0])
#     if  len(vertex_color[0])==7:
#         (x, y, z,r,g,b) = (vertex_color[t] for t in ('x', 'y', 'z','red', 'green', 'blue'))
#         vertices=np.array([x,y,z]).T  # NX3
#         colors=np.array([r,g,b]).T
#     else:
#         (x, y, z) = (vertex_color[t] for t in ('x', 'y', 'z'))
#         vertices=np.array([x,y,z]).T  # NX3
#         colors=np.repeat(np.array([30,144,195.]).reshape(1,3),len(vertices),axis=0)/255
#     #if flag_show:
#     #    plot_mlabvertex(vertices,colors,triangles)
#     return vertices,colors,triangles

def skip_pass(marker, lines):
    """
    Skip until reach the line which contains the marker, then also skip
    the marker line
    """
    result = itertools.dropwhile(
        lambda line: marker not in line,  # Condition
        lines)                            # The lines
    next(result)                          # skip pass the marker
    
    return result

def take(marker, lines):
    """
    Take and return those lines which contains a marker
    """
    #result = itertools.takewhile(
    #    lambda line: marker in line,      # Condition
    #    lines)          
   # marker=']'                  # The lines
    result = itertools.takewhile(
        lambda line: marker not in line,  # Condition
        lines)                            # The lines
    return result

def parse_indexed_face_set(translate, lines):
    """
    Parse one block of 'geometry IndexedFaceSet'
    """
    # For the next line, use logging.WARN to turn off debug print, use
    # logging.DEBUG to turn on
    logging.basicConfig(level=os.getenv('LOGLEVEL', logging.WARN))
    logger = logging.getLogger(__name__)

    # lines = skip_pass('geometry IndexedFaceSet', lines)
     # Parse the "point" structure
    #lines = skip_pass('children', lines)
    #imagename = str(take('repeatS FALSE', lines))
    #print(imagename)
    #logger.debug('imagename: %r', imagename)
    # parse the coordIndex structure
    
    
    # Parse the "point" structure
    lines = skip_pass('[', lines)
    point_lines = take(']', lines)
    verts = [[float(token) for token in line.strip(',\n').split()] for line in point_lines]
    #print(point_lines[-1])
    logger.debug('point: %r', verts)
    #logger.debug('verts: %r', verts)
    # parse the coordIndex structure
    lines = skip_pass('[', lines)
    
    coor_lines = take(']', lines)
    colors = [[int(token) for token in line.strip(',\n').split()[:3]] for line in coor_lines]
    #[tuple(int(token) for token in line.strip(',\n').split(',')) for line in coor_lines]
    #print(facets)
    #logger.debug('coord_index: %r', coord_index)
    #facets = [[verts[i-1] for i in indices[:3]] for indices in coord_index]
    logger.debug('coordIndex: %r', colors)
    #logger.debug('facets: %r', colors)
    
    
    lines = skip_pass('[', lines)
    point_lines = take(']', lines)
    facets = [[float(token) for token in line.strip(',\n').split()[:3]] for line in point_lines]
    #print(point_lines[-1])
    logger.debug('texCoord TextureCoordinate: %r', facets)
    # parse the coordIndex structure
    #lines = skip_pass('texCoordIndex', lines)
    
    lines = skip_pass('[', lines)
    point_lines = take(']', lines)
    texCoordIndex = [[int(token) for token in line.strip(',\n').split()[:3]] for line in point_lines]
    #print(point_lines[-1])
    logger.debug('texCoordIndex: %r', facets)
    
    # coor_lines = take(']', lines)
    # texCoordIndex = [[int(token) for token in line.strip(',\n').split()[:3]] for line in coor_lines]
    # #[tuple(int(token) for token in line.strip(',\n').split(',')) for line in coor_lines]
    # #print(facets)
    # #logger.debug('coord_index: %r', coord_index)
    # #facets = [[verts[i-1] for i in indices[:3]] for indices in coord_index]
    # logger.debug('texCoordIndex: %r', texCoordIndex)
    
    return dict(vert=verts, colors=colors,facets=facets,texCoordIndex=texCoordIndex,
                #texCoord=texCoord, texCoordIndex=texCoordIndex,
                translate=[], normals=[])

def parse_translate(line):
    """
    Given a line such as: "translate 5 6 7", return [5.0, 6.0, 7.0]
    """
    translate = [float(x) for x in line.split()[1:4]]
    return translate

def read_VRML(root):
    indexed_face_sets = []
    translate = []
    with open(root + '.wrl') as infile:
        for line in infile:
            #if 'url' in line:
            #    imgname=str(line)[15:][:-2]
                
            if 'coord Coordinate' in line:
                indexed_face_sets = parse_indexed_face_set(translate=translate, lines=infile)
                #indexed_face_sets.append(a_set)
            #elif 'translation' in line and line.split()[0] == 'translation':
            #    translate = parse_translate(line)
    vertices=np.asarray(indexed_face_sets['vert'])
    triangles=np.asarray(indexed_face_sets['facets'])
    colors=np.asarray(indexed_face_sets['colors'])
    texCoordIndex=np.asarray(indexed_face_sets['texCoordIndex'])
    
    return vertices,colors,triangles,texCoordIndex


