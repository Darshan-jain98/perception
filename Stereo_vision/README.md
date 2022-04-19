# Stereo vision for depth Estimation

## Requirements
 - Python 3.xx
 - Opencv 3.xx

## steps to run the program
1. The file [Stereo_vision.py](./Stereo_vision.py) contains the code in which we try to compute the following:
 - fundamental matrix, Essential Matrix, rotation and translational Matrix.
 - Homography matrix for the two images and perform perspective transformation.
 - Disparity map.
 - Depth map.
2. Various output of the functions are saved in the output folder.
3. The generate the output simply run the [Stereo_vision.py](./Stereo_vision.py) file.

## Notes
### specific editing required for selecting the dataset
 - For selecting the **curule dataset** uncomment the lines 206-212 and comment the lines for the other dataset below it.
 - Perform similar action for the other 2 dataset.