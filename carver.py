import cv2
import numpy as np
import glob
import pickle

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 10)
    return img

def drawBoxes(img, corners, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


def estimatePose(image, mtx, dist):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((24*17,3), np.float32)
    objp[:,:2] = np.mgrid[0:24,0:17].T.reshape(-1,2)
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    pts = np.float32([[0,0], [0,50],
                      [50,0], [50,50]])
    #ret, corners = cv2.findChessboardCorners(gray, (24,17),None)
    
    #corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    ret, rvecs, tvecs = cv2.solvePnP(objp, pts, mtx, dist)
    imgpts, jac = cv2.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)

    img = drawBoxes(image, pts, imgpts)
    cv2.imshow('img',img)
    return

def estraiSilhouette(image):
    image = image[240:850,420:930]
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Perform color-segmentation to get the binary mask
    lwr = np.array([0, 0, 0])
    upr = np.array([179, 255, 146])
    msk = cv2.inRange(hsv, lwr, upr)

    # Extracting the rod using binary-mask
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dlt = cv2.dilate(msk, krn, iterations = 1)
    dlt = cv2.erode(dlt, krn, iterations = 10)

    res = 255 - cv2.bitwise_and(dlt, msk)

    # Display
    cv2.imshow("original", image)
    cv2.imshow("res", res)

def disegnaBordiDisco(image):
    image = image[:,1020:1800]
    disegnaBordi(image)

def disegnaBordi(image):
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    
    pts1 = np.float32([[0, 120], [880, 120],
                       [0, 600], [880, 600]])
    pts2 = np.float32([[0, 0], [400, 0],
                        [0, 640], [400, 640]])
        
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (500, 600))
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = result.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # Display
    cv2.imshow("original", image)
    cv2.imshow("borders", image_copy)


vidcap = cv2.VideoCapture('data\obj01.mp4')
with open('mtx.pkl', 'rb') as f:
    mtx = pickle.load(f)
with open('dist.pkl', 'rb') as f:
    dist = pickle.load(f)

while True:    
    ret, image = vidcap.read()
    if not ret:
        break
    estraiSilhouette(image)
    #disegnaBordiDisco(image)
    #estimatePose(image,mtx,dist)

    k = cv2.waitKey(1) & 0xff    
    # Check if 'q' key is pressed.
    if k == ord('q'):
        break

# Release the VideoCapture Object.
vidcap.release()

cv2.destroyAllWindows()
