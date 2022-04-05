import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy import fft

def Edge_detector(image,frame):
    fft_shift_img = fft.fftshift(fft.fft2(image))
    mag_fft_shift_img = 20 * np.log(np.abs(fft_shift_img))
    rows, cols = np.shape(mag_fft_shift_img)
    crow, ccol = rows//2 , cols//2
    mask = np.ones((rows,cols))
    mask = mask * 255
    mask = cv2.circle(mask,(int(ccol),int(crow)),50,(0,0,0),-1)
    masked_fft_shift_img = fft_shift_img * mask
    mag_mfft = np.log(np.abs(masked_fft_shift_img))
    ifft_img = fft.ifft2(fft.ifftshift(masked_fft_shift_img))
    ifft_img = np.abs(ifft_img)

    plt.subplot(221),plt.imshow(frame,cmap = 'gray')
    plt.title("original Image")
    plt.subplot(222),plt.imshow(mag_fft_shift_img,cmap = 'gray')
    plt.title("Image after fft shift")
    plt.subplot(223),plt.imshow(mag_mfft,cmap = 'gray')
    plt.title("Image after applying mask")
    plt.subplot(224),plt.imshow(ifft_img,cmap = 'gray')
    plt.title("Image after ifft")
    plt.show()




#Read the video file
cap = cv2.VideoCapture("./AR_TAG_Projects/Input_data/1tagvideo.mp4")
print(cap.isOpened())
counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if counter ==1:
        break
    if ret == False:
        print("Video read completely")
        break
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame,(5,5),0)
    retval, threshold_frame = cv2.threshold(blurred_frame,200,255,cv2.THRESH_BINARY)
    Edge_detector(threshold_frame,frame)
    counter = counter +1
