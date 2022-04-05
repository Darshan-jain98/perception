import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy import fft

def tag(image):
    tag = []
    no = 0
    bit = 3
    grid_size_row = np.shape(image)[0]//8
    grid_size_col = np.shape(image)[1]//8
    img2 = np.copy(image[3*grid_size_row:5*grid_size_row,3*grid_size_col:5*grid_size_col,:])
    white = (img2[:,:,0] >200) | (img2[:,:,1] >200) | (img2[:,:,2] >200)
    black = (img2[:,:,0] <200) | (img2[:,:,1] <200) | (img2[:,:,2] <200)
    img2[white] = [255,255,255]
    img2[black] = [0,0,0]
    half_Col = np.shape(img2)[1]//2
    half_row = np.shape(img2)[0]//2
    for i in range(4):
        val = 0
        avg = []
        if i == 0:
            small_s = img2[0:half_row,0:half_Col]
        elif i == 1:
            small_s = img2[0:half_row,half_Col:np.shape(img2)[1]]
        elif i == 2:
            small_s = img2[half_row:np.shape(img2)[0],half_Col:np.shape(img2)[1]]
        else:
            small_s = img2[half_row:np.shape(img2)[0],0:half_Col]
        for i in range(np.shape(small_s)[0]):
                for j in range(np.shape(small_s)[1]):
                    avg.append(small_s[i][j][0])
        white_count = avg.count(255)
        black_count = avg.count(0)
        if white_count > 2*black_count:
            tag.append(1)
            val = 1
        else:
            tag.append(0)
            val = 0
        no = no + (val*2)**bit
        bit = bit -1
    return(tag,no)

def draw8Grid(img): 
    
    quart_x = int(img.shape[0]/8)
    half_x = int(img.shape[0]/2)
    full_x = int(img.shape[0])
    quart_y = int(img.shape[1]/8)
    half_y = int(img.shape[1]/2)
    full_y = int(img.shape[1])
    
    #drawing vertical lines across the tag
    for i in range(1,9):
        cv2.line(img, (quart_x*i,0), (quart_x*i, full_x), (125, 0, 0), 1, 1) 
    #drawing horizontal lines across the tag
    for i in range(1,9):
        cv2.line(img, (0,quart_y*i), (full_y,quart_y*i), (125, 0, 0), 1, 1) 
    return(img)


img = cv2.imread('./AR_TAG_Projects/Input_data/ARtag.png')
# cv2.imshow("image",img)
row = np.shape(img)[0]
col = np.shape(img)[1]
img2 = img[0:row-1,0:col-1,:]
tag_id,no = tag(img2)
print("tag_id ",tag_id)
print("number =", no )
