import cv2
import numpy as np
import matplotlib.pyplot as plt

def orb_feature_detection(image):
    descriptor = cv2.ORB_create()    
    (kpoints, features) = descriptor.detectAndCompute(image, None)
    return (kpoints, features)

def Matcher(crossCheck):
    bforce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bforce

def matchPoints(featuresA, featuresB):
    bforce = Matcher(crossCheck=True)
    best_matches = bforce.match(featuresA,featuresB)
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    return rawMatches

def get_Homo(kpsA, kpsB,matches, Thresh):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,Thresh)
        return (H, status)

ImgB = cv2.imread("./Image_stitching/Input_images/Q2imageB.png")
ImgB_grey = cv2.cvtColor(ImgB, cv2.COLOR_BGR2GRAY)
ImgA = cv2.imread("./Image_stitching//Input_images/Q2imageA.png")
ImgA_grey = cv2.cvtColor(ImgA, cv2.COLOR_BGR2GRAY)

kpsA, featuresA = orb_feature_detection(ImgB_grey)
kpsB, featuresB = orb_feature_detection(ImgA_grey)

matches = matchPoints(featuresA,featuresB)

M = get_Homo(kpsA,kpsB,matches,Thresh=4)
(H, status) = M

width = ImgB.shape[1] + ImgA.shape[1]
height = ImgB.shape[0] + ImgA.shape[0]
result = cv2.warpPerspective(ImgB, H, (width, height))
result[0:ImgA.shape[0], 0:ImgA.shape[1]] = ImgA

result = result[0:325,0:630]
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.show()
