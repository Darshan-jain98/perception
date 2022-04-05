import cv2
import numpy as np
import imutils

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

def distanceCalc(a,b): 
    dist = np.sqrt(((a[0]-b[0]) ** 2) + ((a[1]-b[1]) ** 2)) 
    return (dist)

def determinePoints(out):
    tl = out[0]
    tr = out[1]
    br = out[2]
    bl = out[3]
    width_1 = distanceCalc(br,bl)
    width_2 = distanceCalc(tr,tl)
    max_w = max(int(width_1), int(width_2))

    height_1 = distanceCalc(br,tr)
    height_2 = distanceCalc(bl,tl)
    max_h = max(int(height_1), int(height_2))

    dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]], dtype = "float32")

    return (tl,tr,br,bl,max_w,max_h,dst)

def Corner_Detection(img):
        img = np.float32(img)

        corners = cv2.goodFeaturesToTrack(img,9,0.01,100)
        corners = np.int0(corners)

        return corners

def order_points(points):
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
    rect = []
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

vid = cv2.VideoCapture('./AR_TAG_Projects/Input_data/1tagvideo.mp4')
t_img = cv2.imread("./AR_TAG_Projects/Input_data/testudo.png")
t_img = cv2.rotate(t_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
final_vid = []
count = 0
fourcc = cv2.VideoWriter_fourcc(*'MP4V')#*'XVID'
out = cv2.VideoWriter('./AR_TAG_Projects/Output_video/Testudo.avi',fourcc, 30, (1920,1080))
while vid.isOpened():
    print("Processing Frame: ",count)
    ret,frame = vid.read()
    if ret==False:
            print("Video Read Completely")
            break
    
    try:
        if count%2 ==0:
            blur = cv2.GaussianBlur(frame,(51,51),0)
            gray= cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
            _,threshold= cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
            blur1 = cv2.GaussianBlur(threshold,(71,71),0)
            points = AR_corners(blur1)
            tl,tr,br,bl,maxWidth,maxHeight,d= determinePoints(points)
            H_2 = findhomography(np.array([tl,tr,br,bl],np.float32), d)
            warped = warp(H_2,frame, maxWidth, maxHeight)
            img = cv2.resize(t_img,(warped.shape[0],warped.shape[1]))
            rows, cols, channel = img.shape
            p1 = np.array([[0,0],[0,cols],[rows,cols],[rows,0]])
            p2 = np.array([points[0],points[1],points[2],points[3]])
            h_new = findhomography(p2,p1)
            h_new_inv = np.linalg.inv(h_new)

            for i in range(0,warped.shape[1]):
                for j in range(0,warped.shape[0]):
                    x_testudo = np.array(np.matmul(h_new_inv,[i,j,1]))[0][0]
                    y_testudo = np.array(np.matmul(h_new_inv,[i,j,1]))[0][1]
                    z_testudo = np.array(np.matmul(h_new_inv,[i,j,1]))[0][2]
                    frame[int(y_testudo/z_testudo)][int(x_testudo/z_testudo)] = img[i][j]
            final_vid.append(frame)
    except:
        final_vid.append(frame)
    count = count+1

for i in final_vid:
    out.write(i)

out.release()
cv2.destroyAllWindows()