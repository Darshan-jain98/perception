import cv2
from matplotlib.pyplot import hist
import numpy as np

# Perform histogram equalization mathemathically
def hist_eq(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    pix_dict = {}
    h,s,v = cv2.split(img_hsv)
    for x in range(v.shape[0]):
        for y in range(v.shape[1]):
            if v[x][y] in pix_dict:
                pix_dict[v[x][y]].append(tuple([x,y]))
            else:
                pix_dict[v[x][y]] = [tuple([x,y])]
    
    pdf_dict = {}
    num_pix = v.shape[0]*v.shape[1]
    for i in range(256):
        try:
            pdf_dict[i] = len(pix_dict[i])/num_pix
        except:
            pdf_dict[i] = 0
    
    sum = 0
    cdf_dict = {}
    for i in range(256):
        cdf_dict[i] = sum + pdf_dict[i]
        sum = cdf_dict[i]

    eq_dict = {}
    for i in range(256):
        eq_dict[i] = round(cdf_dict[i]*255)
    
    hist_eq = np.copy(img_hsv)
    for i in range(256):
        if i in pix_dict:
            for j in pix_dict[i]:
                hist_eq[j[0]][j[1]] = (h[j[0]][j[1]],s[j[0]][j[1]],eq_dict[i])
    rgbimg = cv2.cvtColor(hist_eq, cv2.COLOR_HSV2BGR)
    return rgbimg

# Break the image into desired number of tiles and perform histogram equalization on each tile individually
def adaptive_hist_eq(img,n):
    grid_height = round(img.shape[0]/n[0])
    grid_width = round(img.shape[1]/n[1])
    for i in range(n[1]):
        for j in range(n[0]):
            img_copy = np.copy(img)
            img_copy = img[i*grid_height:(i+1)*grid_height,j*grid_width:(j+1)*grid_width]
            eq_img_copy = hist_eq(img_copy)
            img[i*grid_height:(i+1)*grid_height,j*grid_width:(j+1)*grid_width] = eq_img_copy
    blurred = cv2.bilateralFilter(img,11,25,9)
    return blurred

# output files to save the frames
out1 = cv2.VideoWriter('./output_video/Hist_equ.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1224,370))
out2 = cv2.VideoWriter('./output_video/adap_hist_equ.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1224,370))
count = 0

for i in range(25):
    if i < 10:
        image = "000000000" + str(count)
        file = "./Histogram_equalization/Input_data/adaptive_hist_data/" + image + ".png"
        print(file)
        count = count +1
        img_og = cv2.imread(file)
    else:
        image = "00000000" + str(count)
        file = "./Histogram_equalization/Input_data/adaptive_hist_data/" + image + ".png"
        print(file)
        count = count +1
        img_og = cv2.imread(file)
    
    eq_gray = hist_eq(img_og)
    adaptive_eq_gray = adaptive_hist_eq(img_og,[4,4])
   
    out1.write(eq_gray)
    out2.write(adaptive_eq_gray)

out1.release()
out2.release()