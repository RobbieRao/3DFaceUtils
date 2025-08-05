# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:53:58 2025

@author: Robbie
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances


colors=['red','green','blue','yellow','orange','cyan','magenta',
        'red','green','blue','yellow','orange','cyan','magenta']

#=========Clustering Clustering by fast search and find of density peaks===============================
#https://nbviewer.jupyter.org/gist/tclarke/54ed4c12e8344e4b5ddb

def plot_histo(x, ignore=None):
    "Generate and plot a histogram for some data. The optional ignore is a histogram index to 0 out."
    h,b = np.histogram(x)
    if ignore is not None: h[ignore] = 0
    c = (b[:-1]+b[1:])/2
    w = 0.7*(b[1]-b[0])
    fig = plt.figure(figsize=(6,5), dpi=80)
    plt.rc('axes',axisbelow=True)  
    plt.grid(axis="y",c=(217/256,217/256,217/256))         #设置网格线   
                           #将网格线置于底层
    ax = plt.gca()#获取边框
    ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
    ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
    ax.spines['left'].set_color('none')  # 设置上‘脊梁’为无色
    return plt.bar(c,h,align='center',width=w)


def density_showing(X,percent = 10.,dc=None,flag=False):
    from scipy.spatial import distance_matrix
    
    dist=distance_matrix(X,X)
    sA=sparse.csr_matrix(dist)
    
    rows, cols = sA.nonzero()
    xx=np.c_[rows,cols,sA.data]
    
    ND=dist.shape[0]
    N = xx.shape[0]
    # ND,NL = int(xx[:,1].max()), int(xx[:,0].max())
    # if NL > ND: ND = NL
    # dist = np.zeros((ND+1,ND+1),'float')
    # for d in xx:
    #     dist[int(d[0]),int(d[1])] = d[2]
    #     dist[int(d[1]),int(d[0])] = d[2]
    print('N: %r\tND: %r' % (N,ND))
     
    if dc is None:
        position = np.round(N*percent/100.) - 1
        sda = np.sort(xx[:,2])
        dc = sda[int(position)]
        print('Avg percentage of neigbors: %5.6f [%r]' % (percent,position))
        
    rho = np.zeros((ND,), 'float')
    
   
    
    print('Compute Rho with gaussian kernel of radius: %12.6f' % dc)
    tt=plot_histo(dist)
    plt.title("Distance histogram",loc= 'left')
    
    # Gaussian kernel
    for i in np.arange(ND-1):
        for j in np.arange(i+1, ND):
            tmp = np.exp(- (dist[i,j]/dc)**2)
            rho[i] += tmp
            rho[j] += tmp
    tt=plot_histo(rho)
    plt.title(r"$\rho$ histogram",loc= 'left')
    
    '''========================================================================================
    1. Determine maximum distance
    2. Sort \rho to find points of highest density
    3. Find nearest neighbors
    4. Create a decision graph of \rho vs. \delta to locate the cluster centers
    '''
    maxd = dist.max().max()
    print ('maxd: %r' % maxd)
    
    rho_sorted,ordrho = np.sort(rho)[::-1],np.argsort(rho)[::-1]
    delta,nneigh = np.zeros((ND,), 'float'),np.zeros((ND,), 'float')
    delta[ordrho[0]] = -1
    for ii in np.arange(1, ND):
        delta[ordrho[ii]] = maxd
        for jj in np.arange(ii):
            if dist[ordrho[ii],ordrho[jj]] < delta[ordrho[ii]]:
                delta[ordrho[ii]] = dist[ordrho[ii],ordrho[jj]]
                nneigh[ordrho[ii]] = ordrho[jj]
    delta[ordrho[0]] = delta.ravel().max()
    print ("Decision graph: Density,Delta")
    ##np.savetxt(r'decision_graph.txt', np.dstack((rho,delta)).squeeze(), '%6.2f', '\t')
    tt=plot_histo(delta)
    plt.title(r"$\delta$ histogram",loc= 'left')
    
    
    ind,gamma = np.zeros((ND,),'float'),np.zeros((ND,),'float')
    for i in np.arange(ND):
        ind[i] = i
        gamma[i] = rho[i]*delta[i]
    
    fig = plt.figure(figsize=(7,5), dpi=80)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\delta$')
    plt.title('Decision Graph',loc= 'left')
    plt.plot(rho,delta, 'o', color='black')
    
    if flag:
        return dist,rho, delta,nneigh,ordrho,dc
    else:
        return rho, delta

def density_finding(X,percent = 10.,rhomin=20,deltamin=0.1,dc=None):
    
    dist,rho, delta,nneigh,ordrho,dc=density_showing(X,percent = 10.,dc=dc,flag=True)
    
    
    ND=len(rho)
    
    print ("rhomin: %r\tdeltamin: %r" % (rhomin,deltamin))
    NCLUST=0
    cl = np.ndarray((ND,),"int")
    cl[:] = -1
    icl = []
    for i in np.arange(ND):
        if rho[i] > rhomin and delta[i] > deltamin:
            NCLUST += 1
            cl[i] = NCLUST-1
            icl.append(i)
    print ("NUMBER OF CLUSTERS: %r" % NCLUST)
    print ("Performing assignation")
    
    
    for i in np.arange(ND):
        if cl[ordrho[i]] == -1:
            cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

    # halo
    halo = cl.copy()
    if NCLUST > 1:
        bord_rho = np.zeros((NCLUST,),"float")
        for i in np.arange(ND-1):
            for j in np.arange(i+1,ND):
                if cl[i] != cl[j] and dist[i,j] <= dc:
                    rho_aver = (rho[i]+rho[j]) / 2.
                    if rho_aver > bord_rho[cl[i]]:
                        bord_rho[cl[i]] = rho_aver
                    if rho_aver > bord_rho[cl[j]]:
                        bord_rho[cl[j]] = rho_aver
        for i in np.arange(ND):
            if rho[i] < bord_rho[cl[i]]:
                halo[i] = -1
    for i in np.arange(NCLUST):
        nc,nh=0,0
        for j in np.arange(ND):
            if cl[j] == i:
                nc+=1
            if halo[j] == i:
                nh+=1
        print ("CLUSTER: %d CENTER: %d (%.4f,%4f) ELEMENTS: %d CORE: %d HALO: %d" % (i+1,icl[i],rho[icl[i]],delta[icl[i]],nc,nh,nc-nh))
    
    fig = plt.figure(figsize=(7,5), dpi=80)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\delta$')
    plt.title('Decision Graph with Cluster Centers')
    plt.plot(rho,delta, 'o', color='black')
    for i in np.arange(NCLUST):
        plt.plot(rho[icl[i]], delta[icl[i]], 'o', color=colors[i])
    plt.show()
        
    return halo


def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)

def compute_gap(clustering, data, k_max=5, n_references=5,flag_show=True):
    #https://glowingpython.blogspot.com/2019/01/a-visual-introduction-to-gap-statistics.html
    #: Tibshirani R, Walther G, Hastie T. 
    #Estimating the number of clusters in a dataset via the gap statistic. 
    #Journal of the Royal Statistics Society 2001.
    #Gap Statistics
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max+1):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))
    
    ondata_inertia = []
    for k in range(1, k_max+1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))
        
    gap = np.log(reference_inertia)-np.log(ondata_inertia)
    
    if flag_show:
        fig = plt.figure(figsize=(5,5), dpi=300)
        plt.plot(range(1, k_max+1), gap, '-o',color ='k',
                 markerfacecolor ='r',markeredgecolor='k',markersize =8)
        plt.ylabel('gap')
        plt.xlabel('k')
        plt.show()
    return gap, np.log(reference_inertia), np.log(ondata_inertia)

def clustering_evaluate(model, X,X0=None, k_max=5, flag_show=True):
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import calinski_harabasz_score
    import skfuzzy as fuzz
    #from fcmeans import FCM
    
    if X0 is None:
        X0=X
        
    score_SI=[]
    score_CA=[]
    #distortions = []
    #Inertia=[]
    #Max_num=16
    for k in range(2,k_max+1):
        
        #model = AgglomerativeClustering(n_clusters=num_cluster,linkage='ward').fit(X)
        if model=="FCM":
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    X.T, k, 2, error=0.00000001, maxiter=1000, init=None)
            
            labels = np.argmax(u, axis=0)
        else:
            model.n_clusters = k
            model.fit(X)
            labels=model.labels_
        
        score_SI0=silhouette_score(X0,labels)
        score_SI.append(score_SI0)
        #print('聚类%d簇的silhouette_score%f'%(num_cluster,score_SI0))
    
        score_CA0=calinski_harabasz_score(X0,labels)
        score_CA.append(score_CA0)
        print('聚类%d簇的calinski_harabaz: %f,silhouette_score: %f'%(k,score_CA0,score_SI0))
        
        #https://medium.com/@masarudheena/4-best-ways-to-find-optimal-number-of-clusters-for-clustering-with-python-code-706199fa957c
        #distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        #Inertia.append(model.inertia_)
        
    fig = plt.figure(figsize=(7,5), dpi=300)
    plt.plot(range(2,k_max+1),score_SI,zorder=1,c='k')
    plt.scatter(range(2,k_max+1),score_SI,c='r',edgecolors='k',s=20,zorder=2)
    plt.show()
    
    fig = plt.figure(figsize=(7,5), dpi=300)
    plt.plot(range(2,k_max+1),score_CA,zorder=1,c='k')
    plt.scatter(range(2,k_max+1),score_CA,c='r',edgecolors='k',s=20,zorder=2)
    plt.show()
    
    return range(2,k_max+1),score_SI,score_CA


