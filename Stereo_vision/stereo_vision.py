#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

# def fundamental_matrix(points1,points2): 
    # A = []
    # for p1,p2 in zip(points1 , points2):
    #     x1 , y1 = p1[0] , p1[1]
    #     x2 , y2 = p2[0] , p1[1]
    #     a = [x1*x2 , x1*y2 , x1 , y1*x2 , y1*y2 , y1 , x2 , y2 , 1]
    #     # a = [x1*x2 , x2*y1 , x2 , y2*x1 , y2*y1 , y2 , x1 , y1 , 1]
    #     A.append(a)

    # U , sigma , vt = np.linalg.svd(A,full_matrices=True)
    # F = vt.T[:,-1]
    # F  = F.reshape(3,3)

    # U ,sigma , vt = np.linalg.svd(F)
    # sigma = np.diag(sigma)
    # sigma[2,2] = 0
    # F_ = np.dot(U , np.dot(sigma , vt))
    # return F_
    #compute the centroids


def fundamental_matrix(feat_1,feat_2):
    feat_1_mean_x = np.mean(feat_1[:, 0])
    feat_1_mean_y = np.mean(feat_1[:, 1])
    feat_2_mean_x = np.mean(feat_2[:, 0])
    feat_2_mean_y = np.mean(feat_2[:, 1])
    
    #Recenter the coordinates by subtracting mean
    feat_1[:,0] = feat_1[:,0] - feat_1_mean_x
    feat_1[:,1] = feat_1[:,1] - feat_1_mean_y
    feat_2[:,0] = feat_2[:,0] - feat_2_mean_x
    feat_2[:,1] = feat_2[:,1] - feat_2_mean_y
    
        
    
    #Compute the scaling terms which are the average distances from origin
    s_1 = np.sqrt(2.)/np.mean(np.sum((feat_1)**2,axis=1)**(1/2)) 
    s_2 = np.sqrt(2.)/np.mean(np.sum((feat_2)**2,axis=1)**(1/2))
    
     
    #Calculate the transformation matrices
    T_a_1 = np.array([[s_1,0,0],[0,s_1,0],[0,0,1]])
    T_a_2 = np.array([[1,0,-feat_1_mean_x],[0,1,-feat_1_mean_y],[0,0,1]])
    T_a = T_a_1 @ T_a_2
    
    
    T_b_1 = np.array([[s_2,0,0],[0,s_2,0],[0,0,1]])
    T_b_2 = np.array([[1,0,-feat_2_mean_x],[0,1,-feat_2_mean_y],[0,0,1]])
    T_b = T_b_1 @ T_b_2
    

    #Compute the normalized point correspondences
    x1 = ( feat_1[:, 0].reshape((-1,1)))*s_1
    y1 = ( feat_1[:, 1].reshape((-1,1)))*s_1
    x2 = (feat_2[:, 0].reshape((-1,1)))*s_2
    y2 = (feat_2[:, 1].reshape((-1,1)))*s_2
    
    #-point Hartley
    #A is (8x9) matrix
    A = np.hstack((x2*x1, x2*y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones((len(x1),1))))    
        
        
    #Solve for A using SVD    
    A = np.array(A)
    U,S,V = np.linalg.svd(A)
    V = V.T
    
    #last col - soln
    sol = V[:,-1]
    F = sol.reshape((3,3))
    U_F, S_F, V_F = np.linalg.svd(F)
    
    #Rank-2 constraint
    S_F[2] = 0
    S_new = np.diag(S_F)
    
    #Recompute normalized F
    F_new = U_F @ S_new @ V_F
    F_norm = T_b.T @ F_new @ T_a
    F_norm = F_norm/F_norm[-1,-1]
    return F_norm

def ransac_best_fundamental_matrix(feat_1,feat_2):
    #RANSAC parameters
    threshold =0.05
    inliers_present= 0
    F_best = []
    #probability of selecting only inliers
    p = 0.99
    N = 100
    count = 0
    for count in range(N):
        # print(count)
        inlier_count= 0
        random_8_feat_1 = []
        random_8_feat_2 = []
        #generate a set of random 8 points
        random_list = np.random.randint(len(feat_1), size = 8)
        for k in random_list:
            random_8_feat_1.append(feat_1[k])
            random_8_feat_2.append(feat_2[k])
        F = fundamental_matrix(np.array(random_8_feat_1), np.array(random_8_feat_2))
        ones = np.ones((len(feat_1),1))
        x1 = np.hstack((feat_1,ones))
        x2 = np.hstack((feat_2,ones))
        e1, e2 = x1 @ F.T, x2 @ F
        error = np.sum(e2* x1, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e1[:, :-1],e2[:,:-1]))**2, axis = 1, keepdims=True)
        inliers = error<=threshold
        inlier_count = np.sum(inliers)
        #Record the best F
        if inliers_present <  inlier_count:
            inliers_present = inlier_count
            F_best = F 
        #Iterations to run the RANSAC for every frame
        inlier_ratio = inlier_count/len(feat_1)
        if np.log(1-(inlier_ratio**8)) == 0: 
            continue
        N = np.log(1-p)/np.log(1-(inlier_ratio**8))
        count += 1
    return F_best

def SIFT_feature_detection(image):
    sift = cv2.xfeatures2d.SIFT_create()    
    (kpoints, features) = sift.detectAndCompute(image, None)
    return (kpoints, features)

def matchPoints(featuresA, featuresB):
    bforce = cv2.BFMatcher()
    matches = bforce.knnMatch(featuresA,featuresB, k=2)
    best_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            best_matches.append(m)
    return best_matches

def get_essential(K,F):
    E = K.T @ F @ K
    u,s,v = np.linalg.svd(E)
    s = [1,1,0]
    E_fin = np.dot(u, np.dot(np.diag(s),v))
    return E_fin

def decomposeE(E):
    u,s,vt = np.linalg.svd(E)
    W = np.array([[0,-1,0] , [1,0,0] , [0,0,1]])
    c1 = u[:,2]
    r1 = np.dot(u , np.dot(W , vt))
    c2 = -u[:,2]
    r2 = np.dot(u , np.dot(W , vt))
    c3 = u[:,2]
    r3 = np.dot(u , np.dot(W.T , vt))
    c4 = -u[:,2]
    r4 = np.dot(u , np.dot(W.T , vt))

    R = np.array([r1,r2,r3,r4] , dtype =np.float32)
    C = np.array([c1,c2,c3,c4] , dtype = np.float32)
    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]
    return R , C



ImgA = cv2.imread("./Project_3/data/curule/im0.png")
ImgB = cv2.imread("./Project_3/data/curule/im1.png")
ImgA = cv2.cvtColor(ImgA, cv2.COLOR_BGR2GRAY)
ImgB = cv2.cvtColor(ImgB, cv2.COLOR_BGR2GRAY)
ImgA = cv2.GaussianBlur(ImgA,(7,7),0)
ImgB = cv2.GaussianBlur(ImgB,(7,7),0)
# cv2.imshow("img1",ImgA_grey)
# cv2.imshow("img2",ImgB_grey)
# cv2.waitKey(0)
kpsA, featuresA = SIFT_feature_detection(ImgA)
kpsB, featuresB = SIFT_feature_detection(ImgB)
print("Features detected")
# print(len(kpsB))

matches = matchPoints(featuresA,featuresB)

# matchMask = [[0,0] for i in range(len(matches))]

# ImgC = cv2.drawMatches(ImgA,kpsA,ImgB,kpsB,matches,ImgB, flags=2)


kp1_lst = []
kp2_lst = []
for match in matches:
    ImgA_idx = match.queryIdx
    ImgB_idx = match.trainIdx
    (x1,y1) = kpsA[ImgA_idx].pt
    (x2,y2) = kpsA[ImgB_idx].pt
    kp1_lst.append((x1,y1))
    kp2_lst.append((x2,y2)) 

A = []

for i in range(len(kp1_lst)):
    x1,y1 = kp1_lst[i]
    x2,y2 = kp2_lst[i]
    A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

A = np.array(A)
# print(A)

[u,s,v] = np.linalg.svd(A)
# print(v)
print("calculating fm")
F = ransac_best_fundamental_matrix(kp1_lst,kp2_lst)
f = v[-1,:]
F = f.reshape((3,3))
# print(f)
print("calculated fm")
K = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0, 0, 1]])
# print(K)

print("calculating em")
E = get_essential(K,F)
# print(E)
print("calculated em")
print("calculating R&C")
R,C = decomposeE(E)
print("calculated R&C")
print(R)
print("------")
print(C)
# cv2.imshow("ImgC",ImgC)
# cv2.waitKey(0)