import cv2
import os
import numpy as np
import glob
import yaml

desiredFrames=20
vidcap = cv2.VideoCapture('data\calibration.mp4')
success,image = vidcap.read()
fps = vidcap.get(cv2.CAP_PROP_FPS)
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

#calcolo quanti frame saltare tra un salvataggio e l'altro
step = int(frame_count/desiredFrames)

# +++ decommentare per estrarre i frame dal video +++
#if not os.path.exists("frames"):
#    os.makedirs("frames")
#for i in range(0,desiredFrames):
#    vidcap.set(1, i*step)
#    ret, frame = vidcap.read()
#    cv2.imwrite(r"frames\frame%d.jpg" %i, frame)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)   #6x7 -> 9x6
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(r"frames\*.jpg")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        #cv2.drawChessboardCorners(img, (6,9), corners2, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
#cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(type(rvecs))

data = {'ret': ret,
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'rvecs': np.asarray(rvecs).tolist(),
        'tvecs': np.asarray(tvecs).tolist()}

# salvataggio dei parametri in un file yaml
with open("calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)