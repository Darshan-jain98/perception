import math
import numpy as np

x1,x2,x3,x4 =  5,150,150,5
y1,y2,y3,y4 = 5,5,150,150   
xp1,xp2,xp3,xp4 = 100,200,220,100
yp1,yp2,yp3,yp4 = 100,80,80,200

A = np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],
            [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],
            [-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],
            [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],
            [-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],
            [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],
            [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],
            [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]])

#Computing the Transpose of a matrix  
AT = A.T
X = A @ AT
Y = AT @ A

#calculatig the eigen values and eigen vectors of A.AT and AT.A
eigvalU,U = np.linalg.eig(X)
eigvalV,V = np.linalg.eig(Y)
S = np.diag(eigvalV)

#sorting the eigenvectors according to eigen values in decreasing order
sort_eigvalV = eigvalV.argsort()[::-1]
new_eigvalV = eigvalV[sort_eigvalV]
V = V[:,sort_eigvalV] 
V_T = V.T
sort_eigvalU = eigvalU.argsort()[::-1]
new_eigvalU = eigvalU[sort_eigvalU]
U = U[:,sort_eigvalU]

#calculating the sigma matrix
temp = np.diag((np.sqrt(new_eigvalU)))  
sigma = np.zeros_like(A).astype(np.float64)
sigma[:temp.shape[0],:temp.shape[1]]=temp

#Finding the homography matrix
H = V[:,8]
H = np.reshape(H,(3,3))

#Printing all the values
print("U = ",U)
print("V_T = ",V_T)
print("sigma = ",sigma)
print("Homography Matrix = ",H)
