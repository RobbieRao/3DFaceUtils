# -*- coding: utf-8 -*-
"""
Created on 2025-04-04 17:49:40

@author: Robbie
"""
#import tkinter
import matplotlib
matplotlib.use("Agg")

import cv2  #cv2.__version__
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import img_as_float
from . import transform, render
#import dlib #dlib.__version__  conda install -c conda-forge dlib  

LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 36))
MOUTH_POINTS = list(range(48, 68))
JAW_PROFILE=list(range(0,17))
FACE_PROFILE=list(range(0,17))+list(range(26,16,-1))
face_groups=[JAW_PROFILE,
             RIGHT_BROW_POINTS,LEFT_BROW_POINTS,
             NOSE_POINTS,
             RIGHT_EYE_POINTS,LEFT_EYE_POINTS,
             MOUTH_POINTS]

face_rightgroups=[list(range(0,9)),
            RIGHT_EYE_POINTS,
            RIGHT_BROW_POINTS,
            list(range(27, 34)),
            
            list(range(48,52))+list(range(57,63))+[66,67]]

face_leftgroups=[list(range(8,17)),
            LEFT_EYE_POINTS,
            LEFT_BROW_POINTS,
            list(range(27, 31))+list(range(33, 36)),
            list(range(51,58))+[62,63]+[64,65,66]]


def mouse_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_cv, (x, y), 1, (255, 255, 255), thickness = -1)
        pixels.append([x,y])
        #print('Pixels x,y:',x,y)
        cv2.imshow("image", img_cv)
        
def landmark2d_detect(img,method='dlib',landmark3d=None,flag_show=True,return_selectlm=False):
    if method=='dlib':
        import dlib #dlib.__version__  conda install -c conda-forge dlib  

        detector = dlib.get_frontal_face_detector()
        model_path = os.path.join(
            os.path.dirname(__file__),
            "landmark_predictor",
            "shape_predictor_68_face_landmarks.dat",
        )
        predictor = dlib.shape_predictor(model_path)
        #取灰度
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #人脸数rects
        rects = detector(img_gray)
        df_pos = np.array([[p.x, p.y] for p in predictor(img,rects[0]).parts()]).reshape(-1,2)
        
    elif method=='face_alignment':
        import face_alignment #https://github.com/1adrianb/face-alignment

        fa2d = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cpu')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        df_pos = np.array(fa2d.get_landmarks(img1)).reshape(-1,2)
    
    elif method=='baidu':
        from aip import AipFace 
        #pip install baidu-aip
        #pip uninstall chardet
        import base64#,json,os
        from PIL import Image
        from io import BytesIO
        """ 你的 APPID AK SK """
        APP_ID = '18426870'
        API_KEY = 'OESzO9MqG92VGhSU4B05SxwR'
        SECRET_KEY = '2SV73BdLcpzHz3G9BAcQNn1gEbAwA0nW'
        client = AipFace(APP_ID, API_KEY, SECRET_KEY)
        
        #Convert opencv image format to PIL image format
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img1)
        
        output_buffer = BytesIO()
        im_pil.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        base64_img = base64.b64encode(byte_data)
        
        base64_img = str(base64_img,'utf-8')
        # image = "取决于image_type参数，传入BASE64字符串或URL字符串或FACE_TOKEN字符串"
        imageType = "BASE64"

        """ 如果有可选参数 """
        options = {}
        options["face_field"] = "landmark150"#"landmark150"
        """ 带参数调用人脸检测 """
        content = client.detect(base64_img, imageType, options)
        landmark_label = content['result']['face_list'][0]["landmark150"]
        #print(landmark_label)
        landmark_label = landmark_label.values()
        df_pos=np.array([ [xy['x'],xy['y']]  for xy in landmark_label ]).reshape(-1,2)
        
    
    elif method=='manual':
        global pixels,img_cv
        pixels=[]
        img_cv=img.copy()
        
        width0  = img.shape[1]
        height0 = img.shape[0]
        width1 = 600
        height1 = 600
        dim = (width1, height1)
        # resize image
        img_cv = cv2.resize(img_cv, dim, interpolation = cv2.INTER_AREA)
        
        try:
            cv2.namedWindow("image")
            cv2.imshow("image", img_cv)
            cv2.resizeWindow("image", width1, height1)
            cv2.setMouseCallback("image", mouse_pixel)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as err:
            raise RuntimeError(
                "OpenCV GUI functions are not available. Install opencv-python with GUI support or use a display-capable environment"
            ) from err

        df_pos=np.array(pixels).reshape(-1,2)
        df_pos[:,0]=df_pos[:,0]/width1*width0
        df_pos[:,1]=df_pos[:,1]/height1*height0

        del pixels,img_cv
        
    plt_groups=face_groups
    
    if landmark3d is not None:
        P = transform.estimate_affine_matrix_3d22d(landmark3d, df_pos)
        s, R, t = transform.P2sRt(P)
        rx, ry, rz = transform.matrix2angle(R)
        if ry>15:
            plt_groups=face_leftgroups
        elif ry<-15:
            plt_groups=face_rightgroups
            
    select_lm=np.array([i for k in plt_groups for i in k])    
    if flag_show:
            fig = plt.figure(figsize=(6*2,6), dpi=300)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            image = img_as_float(img.copy())
            image = image[:, :, ::-1]
            ax1.imshow(image)
            ax2.imshow(image)
            
            if (method=='face_alignment') | (method=='dlib'):
                for group in plt_groups:
                    ax2.plot(df_pos[group,0],df_pos[group,1],linewidth=0.75,zorder=1,c='k')
                    ax2.scatter(df_pos[group,0],df_pos[group,1],
                            c='r',s=20, linewidths=0.5, edgecolors="k",alpha=1,marker="o",zorder=2)
            else:
                ax2.scatter(df_pos[:,0],df_pos[:,1],
                            c='y',s=20, linewidths=0.5, edgecolors="k",alpha=1,marker="o",zorder=2)
            plt.show()
    if return_selectlm: 
        return df_pos,select_lm
    else:
        return df_pos

def face_pose2d_estimation(x, X_ind, model,landmark3d=None):
    if landmark3d is None:
        X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
        X_ind_all[1, :] += 1
        X_ind_all[2, :] += 2
        valid_ind = X_ind_all.flatten('F')
        X = model['shapeMU'][valid_ind, :]
        X = np.reshape(X, [int(len(X)/3), 3])
    else:
        X=landmark3d.copy()
        
    #----- estimate pose
    P = transform.estimate_affine_matrix_3d22d(X, x)
    s, R, t = transform.P2sRt(P)
    rx, ry, rz = transform.matrix2angle(R)
    
    return rx, ry, rz
      
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    #print len(channels)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def landmark3d_detect(vertices,colors,triangles,
                     method='dlib',angle=[0, 0, 0], 
                     img_h=500,img_w=500,
                     flag_show=True,flag_imgadjust=False,flag_Cplus=True):
    
    
    threshold_area=0.1*1000  #related to pyhsical unit
    
    if colors.max()<1:
        attribute=colors[:,0:3].astype(float)
    else:
        attribute=colors[:,0:3].astype(float)/255
    
    R0 = transform.angle2matrix(angle) 
    vertices=vertices.dot(R0.T)
   
    t = [-(vertices[:,0].max()+vertices[:,0].min())/2, 0, 0]
    transformed_vertices = vertices+np.squeeze(np.array(t, dtype = np.float32))
    
    s =min(img_h/2/(np.max(np.abs(transformed_vertices[:,0]))),
           img_w/2/(np.max(np.abs(transformed_vertices[:,1]))))*0.95
           
    transformed_vertices = s * transformed_vertices# + t3d[np.newaxis, :]      

    image_vertices = transform.to_image(transformed_vertices, img_h, img_w)
    
    transform_image,depth_vertices = render.render_colors(image_vertices, triangles, attribute,
                                                          img_h, img_w,c=3,flag_depth=True)   
    
    #plt.imshow(transform_image)
    if flag_imgadjust:  
        alpha=2.5
        beta=25
        gamma=0.6#0.3#
        img_color = (transform_image[:, :, ::-1]*255).astype(np.uint8)
        img_color = cv2.convertScaleAbs(img_color, alpha=alpha, beta=beta)
        img_color = adjust_gamma(img_color, gamma=gamma)
    
        #img_color=hisEqulColor(img_color)
        transform_image=cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB).astype(float)/255
    
    transform_image1=(transform_image*255).astype(np.uint8)
    transform_image1 = cv2.cvtColor(transform_image1, cv2.COLOR_BGR2RGB)
    df_pos=landmark2d_detect(transform_image1,method=method,flag_show=flag_show)

    X_min_img=np.min(image_vertices[:,0])
    X_max_img=np.max(image_vertices[:,0])
    Y_min_img=np.min(image_vertices[:,1])
    Y_max_img=np.max(image_vertices[:,1])


    X_min_face=np.min(vertices[:,0])
    X_max_face=np.max(vertices[:,0])
    Y_min_face=np.min(vertices[:,1])
    Y_max_face=np.max(vertices[:,1])
    
    #Select_landmark=[i for k in face_groups for i in k]
    N_landmark=df_pos.shape[0]#len(Select_landmark)
    T_landmark=np.zeros((N_landmark,3))
    T_landmark_idx=np.zeros((N_landmark))
    for i in range(N_landmark):
        #x0=df_pos[Select_landmark[i],0]
        #y0=df_pos[Select_landmark[i],1]
        x0=df_pos[i,0]
        y0=df_pos[i,1]
        x=(x0-X_min_img)/(X_max_img-X_min_img)*(X_max_face-X_min_face)+X_min_face
        y=(Y_max_img-y0)/(Y_max_img-Y_min_img)*(Y_max_face-Y_min_face)+Y_min_face
        dist=(vertices[:,0]-x)**2+(vertices[:,1]-y)**2
        
        #dist=dist*depth_vertices
        #dist[depth_vertices<1]=np.Inf
        #if np.sum(angle)==0:
        #    dist[vertices[:,2]<vertices[:,2].mean()]=np.Inf
        #else:
        dist[depth_vertices<1]=np.Inf
        dist[vertices[:,2]<vertices[:,2].mean()]=np.Inf
        #dist[vertices[:,2]<-threshold_area]=np.Inf
        
        T_landmark_idx[i]=np.argmin(dist)
        T_landmark[i,:]=np.array([x,y,vertices[int(T_landmark_idx[i]),2]])
    
    R_inv=np.linalg.inv(R0.T)
    T_landmark= T_landmark.dot(R_inv)
    vertices= vertices.dot(R_inv)

    return T_landmark,T_landmark_idx.astype(int)  