import cv2
import numpy as np
import glob

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


vidcap = cv2.VideoCapture('data\obj01.mp4')
vidcap.set(1, 50)
ret, frame = vidcap.read()
#cv2.imshow('frame 50',frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

mtx = np.load("mtx.pkl", allow_pickle=True)
dist = np.load("dist.pkl", allow_pickle=True)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

img = frame
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (6,9),None)
if ret == True:
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img,corners2,imgpts)
    cv2.imshow('img',img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'):
        cv2.imwrite(frame[:6]+'.png', img)
else:
    print("fail")

#for fname in glob.glob(r"frames\*.jpg"):
#    img = cv2.imread(fname)
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    ret, corners = cv2.findChessboardCorners(gray, (6,9),None)
#    if ret == True:
#        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
#        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
#        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
#        img = draw(img,corners2,imgpts)
#        cv2.imshow('img',img)
#        k = cv2.waitKey(0) & 0xFF
#        if k == ord('s'):
#            cv2.imwrite(fname[:6]+'.png', img)
cv2.destroyAllWindows()