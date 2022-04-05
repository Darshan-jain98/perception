import cv2
import matplotlib.pylab as plt
import numpy as np
# from scipy.optimize import curve_fit

#Function to be used with Curve_Fit MEthod
def objective(x,a,b,c):
    return a*x*x + b*x + c

#Read the video file
# cap = cv2.VideoCapture("ball_video1.mp4")
cap = cv2.VideoCapture("./Curve_fitting/ball_video2.mp4")
X_pos= []
Y_pos = []

while cap.isOpened():

    #Get each Frame from video to process the frame
    ret, frame = cap.read()
    if ret == False:
        print("Video read completely")
        break

    #convert the image from BGR to HSV for color detection
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Define the threshold for red color
    lower_red = np.array([160,50,50])
    upper_red = np.array([180,255,255])

    # Threshold the HSV image using inRange function to get only red colors
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    #get the lower and upper X-Y coordinates of the ball and find there mean
    X=np.where(res !=0)[1]
    Y=np.where(res !=0)[0]
    X_pos.append((np.min(X)+np.max(X))/2)
    Y_pos.append((np.min(Y)+np.max(Y))/2)

#Mathematical calculation method
X = np.array([np.square(X_pos),np.array(X_pos),np.ones(len(X_pos))])
X = X.T
Y = np.array(Y_pos)
B = np.linalg.inv(X.T @ X) @ X.T @ Y
y_fit = []
for i in X_pos:
    y_fit.append(B[0]*i*i + B[1]*i + B[2])
plt.scatter(X_pos,Y_pos)
plt.plot(X_pos,y_fit,color = 'red')
plt.xlabel('X_position')
plt.ylabel('Y_position')
plt.title("Least square fit for videos")
plt.axis([min(X_pos)-50, max(X_pos)+50, max(Y_pos)+50, min(Y_pos)-50])
plt.show()


#Curve_fit Method
# popt, pcov = curve_fit(objective, X_pos, Y_pos)
# a,b,c = popt
# x_line = np.arange(min(X_pos), max(X_pos), 1)
# y_line = objective(x_line,a,b,c)
# plt.scatter(X_pos,Y_pos)
# plt.plot(x_line,y_line,color = 'red')
# plt.scatter(X_pos,Y_pos)
# plt.show()

#destroy all windows
cap.release()
cv2.destroyAllWindows()