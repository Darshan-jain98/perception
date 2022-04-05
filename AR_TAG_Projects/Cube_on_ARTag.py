import cv2
import numpy as np
import imutils
from random import randint


def findhomography(input_pts, output_pts):
    x1 = input_pts[0][0]
    x2 = input_pts[1][0]
    x3 = input_pts[2][0]
    x4 = input_pts[3][0]
    y1 = input_pts[0][1]
    y2 = input_pts[1][1]
    y3 = input_pts[2][1]
    y4 = input_pts[3][1]
    xp1 = output_pts[0][0]
    xp2 = output_pts[1][0]
    xp3 = output_pts[2][0]
    xp4 = output_pts[3][0]
    yp1 = output_pts[0][1]
    yp2 = output_pts[1][1]
    yp3 = output_pts[2][1]
    yp4 = output_pts[3][1]
    
    A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1], 
                   [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1], 
                   [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2], 
                   [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
                   [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3], 
                   [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
                   [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4], 
                   [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]])   
    U, Sigma, V = np.linalg.svd(A)   
    H = np.reshape(V[-1, :], (3, 3))
    Lambda = H[-1,-1]
    H = H/Lambda   
    return H

def projectionMatrix(h, K): 
    h = np.linalg.inv(h) 
    h1 = h[:,0]          
    h2 = h[:,1]
    h3 = h[:,2]

    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)
    
    det = np.linalg.det(b_t)
    if det > 0:
        b = b_t
    else:                  
        b = -1 * b_t  
    r1 = b[:, 0]
    r2 = b[:, 1]                      
    r3 = np.cross(r1.T, r2.T)
    r3 = r3.T
    t = b[:, 2]
    Rt = np.column_stack((r1, r2, r3, t))
    P = np.matmul(K,Rt)  
    return P

def warp(H,img,max_height,max_width):
    H_inv=np.linalg.inv(H)
    warp=np.zeros((max_height,max_width,3),np.uint8)
    for a in range(max_height):
        for b in range(max_width):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            warp[a][b] = img[int(y/z)][int(x/z)]
    warp = imutils.rotate(warp,90)
    return warp

def draw_cube(image , coord):
    img = image.copy()
    for i in coord:
        x , y  = i.ravel()
        x = int(x)
        y = int(y) 
        img = cv2.circle(img , (int(x),int(y)) ,2,(0,0,255) , -1)
        
        cube = cv2.line(img , tuple(coord[0].astype(int)) , tuple(coord[1].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        cube = cv2.line(img , tuple(coord[0].astype(int)) , tuple(coord[2].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        cube = cv2.line(img , tuple(coord[0].astype(int)) , tuple(coord[4].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        
        cube = cv2.line(img , tuple(coord[1].astype(int)) , tuple(coord[3].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        cube = cv2.line(img , tuple(coord[1].astype(int)) , tuple(coord[5].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        
        cube = cv2.line(img , tuple(coord[2].astype(int)) , tuple(coord[6].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        cube = cv2.line(img , tuple(coord[2].astype(int)) , tuple(coord[3].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)

        cube = cv2.line(img , tuple(coord[3].astype(int)) , tuple(coord[7].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)

        cube = cv2.line(img , tuple(coord[4].astype(int)) , tuple(coord[5].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        cube = cv2.line(img , tuple(coord[4].astype(int)) , tuple(coord[6].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        
        cube = cv2.line(img , tuple(coord[5].astype(int)) , tuple(coord[7].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)
        
        cube = cv2.line(img , tuple(coord[6].astype(int)) , tuple(coord[7].astype(int)) , (randint(50,100),randint(50,100),randint(50,100)) , 2)

    return cube

def cube_cor(P):
    x1, y1,z1 = P.dot([[0],[0],[0],[1]])
    x2 , y2 , z2 = P.dot([[0],[size] , [0],[1]])
    x3 , y3 , z3 = P.dot([[size] , [0] , [0] ,[1]])
    x4 , y4 , z4 = P.dot([[size],[size] , [0] ,[1]])

    x5 , y5 , z5 = P.dot([[0] , [0] , [-size] , [1]])
    x6 , y6 , z6 = P.dot([[0],[size] , [-size] , [1]])
    x7 , y7 , z7 = P.dot([[size] , [0],[-size],[1]])
    x8 , y8 , z8 = P.dot([[size] , [size] , [-size],[1]])
    x1 = int(x1)
    x2 = int(x2)
    x3 = int(x3)
    x4 = int(x4)
    x5 = int(x5)
    x6 = int(x6)
    x7 = int(x7)
    x8 = int(x8)
    y1 = int(y1)
    y2 = int(y2)
    y3 = int(y3)
    y4 = int(y4)
    y5 = int(y5)
    y6 = int(y6)
    y7 = int(y7)
    y8 = int(y8)
    z1 = int(z1)
    z2 = int(z2)
    z3 = int(z3)
    z4 = int(z4)
    z5 = int(z5)
    z6 = int(z6)
    z7 = int(z7)
    z8 = int(z8)

    X = [x1/z1 ,x2/z2 ,x3/z3 ,x4/z4 ,x5/z5 ,x6/z6 ,x7/z7 ,x8/z8]
    Y = [y1/z1 ,y2/z2 ,y3/z3 ,y4/z4 ,y5/z5 ,y6/z6 ,y7/z7 ,y8/z8]
    
    coordinates = np.dstack((X,Y))
    coordinates = coordinates.reshape((-1,2))
    return coordinates

def distanceCalc(a,b):
    dist = np.sqrt(((a[0]-b[0]) ** 2) + ((a[1]-b[1]) ** 2)) 
    return (dist)

def Corner_Detection(img):
        img = np.float32(img)
        corners = cv2.goodFeaturesToTrack(img,9,0.01,100)
        corners = np.int0(corners)
        return corners

def order_points(points):
    # print("Points isn ppoints", points)
    rect = rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis = 1) 
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points,axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def AR_corners(img):
    corners = Corner_Detection(img)
    points = []
    corners = corners.reshape((-1,2))
    y = np.sort(corners[:,1])
    y_max = y[-1]
    y_min = y[0]
    
    corners = np.delete(corners,np.where(corners[:,1] == y_max)[0],0)
    corners = np.delete(corners,np.where(corners[:,1] == y_min)[0],0)

    x = np.sort(corners[:,0])
    x_max = x[-1]
    x_min = x[0]

    corners = np.delete(corners,np.where(corners[:,0] == x_max)[0],0)
    corners = np.delete(corners,np.where(corners[:,0] == x_min)[0],0)
    
    y = np.sort(corners[:,1])
    y_max = y[-1]
    y_min = y[0]

    points.append(corners[np.where(corners[:,1] == y_max)[0]][0])    
    points.append(corners[np.where(corners[:,1] == y_min)[0]][0])
    
    x = np.sort(corners[:,0])
    x_max = x[-1]
    x_min = x[0]
    
    points.append(corners[np.where(corners[:,0] == x_max)[0]][0])    
    points.append(corners[np.where(corners[:,0] == x_min)[0]][0])
    points = np.array(points)
    points = points.reshape((-1,2))
    points = order_points(points)
    return np.array(points)

K =np.array([[1406.08415449821,0,0],
   [ 2.20679787308599, 1417.99930662800,0],
   [ 1014.13643417416, 566.347754321696,1]])
K = K.T
vid = cv2.VideoCapture('./AR_TAG_Projects/Input_data/1tagvideo.mp4')
final_vid = []
count = 0
size = 256
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('./AR_TAG_Projects/Output_video/cube.avi',fourcc, 20, (1920,1080))
while vid.isOpened():
    print("Processing Frame: ",count)
    ret,frame = vid.read()
    if ret==False:
            print("Video Read Completely")
            break
    frame1 = frame.copy()
    try:
        if count%2 ==0:
            blur = cv2.GaussianBlur(frame,(51,51),0)
            gray= cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
            _,threshold= cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
            blur2 = cv2.GaussianBlur(threshold,(71,71),0)
            points = AR_corners(blur2)
            trial_corners = np.array([[0,0],[size,0],[size,size],[0,size]])
            H_2 = findhomography(np.array(points,np.float32),trial_corners)
            P = projectionMatrix(H_2, K)
            XY = cube_cor(P)
            frame2 = draw_cube(frame1, XY)
            # cv2.imshow("final",frame2)
            # cv2.waitKey(1)
            for i in points:
                x,y = i.ravel()
                frame2 = cv2.circle(frame2,(int(x),int(y)), 5, (0,0,255),-1)
            final_vid.append(frame2)
            out.write(frame2)
    except:
        final_vid.append(frame2)
        out.write(frame2)
    count = count+1
out.release()
