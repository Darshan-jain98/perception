from email.mime import base
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def fundamental_matrix(points1,points2): 
    A = []
    for p1,p2 in zip(points1 , points2):
        x1 , y1 = p1[0] , p1[1]
        x2 , y2 = p2[0] , p1[1]
        a = [x1*x2 , x1*y2 , x1 , y1*x2 , y1*y2 , y1 , x2 , y2 , 1]
        # a = [x1*x2 , x2*y1 , x2 , y2*x1 , y2*y1 , y2 , x1 , y1 , 1]
        A.append(a)

    U , sigma , vt = np.linalg.svd(A,full_matrices=True)
    F = vt.T[:,-1]
    F  = F.reshape(3,3)

    U ,sigma , vt = np.linalg.svd(F)
    sigma = np.diag(sigma)
    sigma[2,2] = 0
    F_ = np.dot(U , np.dot(sigma , vt))
    return F_

def epipolarError(pt1 , pt2 , F):
    pt1 = np.array([pt1[0] , pt1[1] , 1])
    pt2 = np.array([pt2[0] , pt2[1] , 1]).T
    error = np.dot(pt2 , np.dot(F,pt1))

    return abs(error)

def ransac_best_fundamental_matrix(feat_1,feat_2):
    threshold =0.02
    max_inliers = 0
    #probability of selecting only inliers
    p = 0.99
    N = 100
    n_rows = feat_1.shape[0]
    for i in range(N):
        indices = []
        random_indices = np.random.choice(n_rows , size = 8)
        points1_8 = feat_1[random_indices]
        points2_8 = feat_2[random_indices]
        F = fundamental_matrix(points1_8 , points2_8)
        # print(F)
        for j in range(n_rows):
            error = epipolarError(feat_1[j] , feat_2[j] , F)
            if error < threshold:
                indices.append(j)
        if len(indices) > max_inliers:
            max_inliers = len(indices)
            inliers = indices
            F_final = F
    pts1_inliers , pts2_inliers = feat_1[inliers] , feat_2[inliers]
    return F_final , pts1_inliers , pts2_inliers

def cv2F(points1 , points2):
    F , mask = cv2.findFundamentalMat(points1 , points2 ,cv2.RANSAC, 1, 0.90)
    u , s , vt = np.linalg.svd(F)
    s[2] = 0.0
    F = u.dot(np.diag(s).dot(vt))
    return F , mask

def SIFT_feature_detection(image):
    sift = cv2.xfeatures2d.SIFT_create()    
    (kpoints, features) = sift.detectAndCompute(image, None)
    return (kpoints, features)

def matchPoints(kpsA,featuresA, kpsB, featuresB):
    bforce = cv2.BFMatcher()
    matches = bforce.knnMatch(featuresA,featuresB, k=2)
    best_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            best_matches.append(m)
    kp1_lst = []
    kp2_lst = []
    for match in best_matches:
        ImgA_idx = match.queryIdx
        ImgB_idx = match.trainIdx
        (x1,y1) = kpsA[ImgA_idx].pt
        (x2,y2) = kpsB[ImgB_idx].pt
        kp1_lst.append((x1,y1))
        kp2_lst.append((x2,y2)) 
    return np.array(kp1_lst), np.array(kp2_lst)

def get_essential(K,F):
    E = K.T @ F @ K
    u,s,v = np.linalg.svd(E)
    s = [1,1,0]
    E_fin = np.dot(u, np.dot(np.diag(s),v))
    return E_fin

def decomposeE(E):
    u,s,vt = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    c1 = u[:,2]
    r1 = np.dot(u , np.dot(W , vt))
    c2 = -u[:,2]
    r2 = np.dot(u , np.dot(W , vt))
    c3 = u[:,2]
    r3 = np.dot(u , np.dot(W.T , vt))
    c4 = -u[:,2]
    r4 = np.dot(u , np.dot(W.T , vt))
    if (np.linalg.det(r1) < 0):
        r1 = -r1
        c1 = -c1
    if (np.linalg.det(r2) < 0):
        r2 = -r2
        c2 = -c2
    if (np.linalg.det(r3) < 0):
        r3 = -r3
        c3 = -c3
    if (np.linalg.det(r4) < 0):
        r4 = -r4
        c4 = -c4
    c1 = c1.reshape((3,1))
    c2 = c2.reshape((3,1))
    c3 = c3.reshape((3,1))
    c4 = c4.reshape((3,1))
    
    return [r1,r2,r3,r4] , [c1,c2,c3,c4]

def point_3d(pt,pt_,R2,C2,K):
    #Find the projection matrices for respective frames
    C1 = [[0],[0],[0]]
    R1 = np.identity(3)
    R1C1 = -R1@C1
    R2C2 = -R2@C2
    #Current frame has no Rotation and Translation
    P1 = K @ np.hstack((R1, R1C1))
    
    #Estimate the projection matrix for second frame using returned R and T values
    P2 = K @ np.hstack((R2, R2C2))
    #P1_T = P1.T
    #P2_T = P2.T	
    X = []
    
    #Solve linear system of equations using cross-product technique, estimate X using least squares technique
    for i in range(len(pt)):
        x1 = pt[i]
        x2 = pt_[i]
        A1 = x1[0]*P1[2,:]-P1[0,:]
        A2 = x1[1]*P1[2,:]-P1[1,:]
        A3 = x2[0]*P2[2,:]-P2[0,:]
        A4 = x2[1]*P2[2,:]-P2[1,:]		
        A = [A1, A2, A3, A4]
        U,S,V = np.linalg.svd(A)
        V = V[3]
        V = V/V[-1]
        X.append(V)
    return X

def linear_triangulation(pt,pt_, R,C,K):
    #Check if the reconstructed points are in front of the cameras using cheilarity equations
    X1 = point_3d(pt,pt_,R,C,K)
    X1 = np.array(X1)	
    count = 0
    #r3(X-C)>0
    for i in range(X1.shape[0]):
        x = X1[i,:].reshape(-1,1)
        if R[2]@np.subtract(x[0:3],C) > 0 and x[2] > 0: 
            count += 1
    return count

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def SSD(win1 , win2):
    if win1.shape != win2.shape:
        return -1
    ssd = np.sum(np.square(win1 - win2))
    return ssd

def disparityMap(gray1, gray2, window_size, search_range, vmin, vmax):
    height , width = gray1.shape
    dis_map = np.zeros_like(gray1)
    gray1 , gray2 = gray1.astype(np.float64) , gray2.astype(np.float64)
    for y in tqdm(range(window_size, height-window_size)):
        for x in range(window_size, width - window_size):
            window = gray1[y-(window_size//2):y+(window_size//2) , x-(window_size//2):x+(window_size//2)]
            x1 = blockMatching(y,x,window,gray2 , window_size , search_range)
            NewValue = ((((np.abs((x1 - x))-vmin)*200)/(vmax-vmin))+0)
            dis_map[y,x] = NewValue
    return dis_map

def blockMatching(y,x,window,gray2 , window_size , searchRange):
    height1 , width1 = gray2.shape
    x_start = max(0, x - searchRange)
    x_end = min(width1 , x + searchRange)
    min_x = np.inf
    min_ssd  = np.inf
    for x in range(x_start , x_end,window_size):
        window2 = gray2[y-(window_size//2):y+(window_size//2), x-(window_size//2):x+(window_size//2)]
        ssd = SSD(window, window2)
        if ssd < min_ssd:
            min_ssd = ssd 
            min_x = x
    return  min_x

def computeDepth(disparity_map , baseline , f):
    depth_map = (baseline*f)/(disparity_map + 1e-10)
    depth_map = np.uint8(depth_map *255 / np.max(depth_map))
    return depth_map

# # Data for curule dataset
# baseline = 88.39
# f = 1758.23
# K = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0, 0, 1]])
# vmin = 55
# vmax = 195
# ImgA_c = cv2.imread("./Project_3/data/curule/im0.png")
# ImgB_c = cv2.imread("./Project_3/data/curule/im1.png")

# # Data for octagon dataset
baseline = 221.76
f = 1742.11
K = np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0, 0, 1]])
vmin = 29
vmax = 61
ImgA_c = cv2.imread("./Project_3/data/octagon/im0.png")
ImgB_c = cv2.imread("./Project_3/data/octagon/im1.png")

# # Data for pendulum dataset
# baseline = 537.75
# f = 1729.05
# K = np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0, 0, 1]])
# vmin = 25
# vmax = 150
# ImgA_c = cv2.imread("./Project_3/data/pendulum/im0.png")
# ImgB_c = cv2.imread("./Project_3/data/pendulum/im1.png")


ImgA = cv2.cvtColor(ImgA_c, cv2.COLOR_BGR2GRAY)
ImgB = cv2.cvtColor(ImgB_c, cv2.COLOR_BGR2GRAY)
ha,wa = ImgA.shape[:2]
hb,wb = ImgB.shape[:2]
kpsA, featuresA = SIFT_feature_detection(ImgA)
kpsB, featuresB = SIFT_feature_detection(ImgB)
print("Features detected")

kp1_lst, kp2_lst = matchPoints(kpsA,featuresA,kpsB,featuresB)
kp1_lst, kp2_lst = np.array(kp1_lst), np.array(kp2_lst)


F , pts1_inliers , pts2_inliers = ransac_best_fundamental_matrix(np.int32(kp1_lst),np.int32(kp2_lst))
print("Fundamental Matrix is: ")
print(F)
# pts1 = np.int32(kp1_lst)
# pts2 = np.int32(kp2_lst)
# fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
# We select only inlier points
# pts1 = pts1[inliers.ravel() == 1]
# pts2 = pts2[inliers.ravel() == 1]
# print("Value of F using inbuilt algorithm method is:")
# print(fundamental_matrix)
print("Essential Matrix is: ")
E = get_essential(K,F)
print(E)

R,C = decomposeE(E)
dis = 0
for i in range(4):
    Z = linear_triangulation(kp1_lst,kp2_lst,R[i],C[i],K)
    if dis<Z:
        dis,ind = Z,i

R = R[ind]
t = C[ind]
print("Rotation from Essential Matrix is: ")
print(R)
print("Translational from Essential Matrix is: ")
print(t)

lines1 = cv2.computeCorrespondEpilines(pts2_inliers.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(ImgA, ImgB, lines1, pts1_inliers, pts2_inliers)

lines2 = cv2.computeCorrespondEpilines(pts1_inliers.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(ImgB, ImgA, lines2, pts2_inliers, pts1_inliers)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.suptitle("Epilines in both images")
# plt.savefig("./Project_3/output/Epilines_with_features_5.png")
plt.show()

_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1_inliers), np.float32(pts2_inliers), F, imgSize=(wa, ha))

print("Homography matrix for left image is :")
print(H1)
print("Homography matrix for right image is :")
print(H2)

img1_rectified = cv2.warpPerspective(ImgA, H1, (wa, ha))
img2_rectified = cv2.warpPerspective(ImgB, H2, (wb, hb))

# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes[0].imshow(img1_rectified, cmap="gray")
# axes[1].imshow(img2_rectified, cmap="gray")
# axes[0].axhline(250)
# axes[1].axhline(250)
# axes[0].axhline(450)
# axes[1].axhline(450)
# plt.suptitle("Rectified images")
# # plt.savefig("./Project_3/output/rectified_images_5.png")
# plt.show()

# disparity = disparityMap(img1_rectified, img2_rectified, 10, 75, vmin, vmax)

# plt.imshow(disparity , cmap = 'gray')
# # plt.savefig("./Project_3/output/dis_map_g5.png")
# plt.show()
# plt.imshow(disparity , cmap = plt.cm.RdBu ,interpolation= 'bilinear')
# # plt.savefig("./Project_3/output/dis_map_c5.png")
# plt.show()

# depth_map = computeDepth(disparity,baseline,f)
# plt.imshow(depth_map, cmap = 'gray')
# # plt.savefig("./Project_3/output/depth_map_g5.png")
# plt.show()
# plt.imshow(depth_map, cmap = plt.cm.RdBu ,interpolation= 'bilinear')
# # plt.savefig("./Project_3/output/depth_map_c5.png")
# plt.show()
